# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Speech in: microphone → voice-activity detection → text (PROVIDED).

The workshop's microphone front end. The live pipeline hears through Gemma 4
instead (``vla_pipeline.llm.voice``).

- :class:`Segmenter` cuts a 30 ms block stream into utterances. A pre-roll
  ring keeps the first syllable that arrives while the detector is still
  deciding; ``hangover_s`` of detector-silence ends an utterance. Anything
  shorter than ``min_speech_s`` (a click, a cough) is dropped, and
  :class:`MicListener` reports it through ``on_drop``. ``webrtcvad``
  decides speech when installed, else an energy detector riding 3x above the
  adaptive noise floor.
- :class:`MicListener` runs that on a thread and hands every transcript to a
  callback. The caller supplies the function that turns audio into text.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK = 480  # 30 ms - webrtcvad's frame size


def _webrtc_vad(aggressiveness: int):
    try:
        import webrtcvad
    except ImportError:
        return None
    return webrtcvad.Vad(int(aggressiveness))


class Segmenter:
    """Block in, utterance out: :meth:`push` returns audio once speech ends."""

    def __init__(self, cfg: dict):
        a = cfg.get("audio") or {}
        blocks = lambda s: max(1, int(float(s) * SAMPLE_RATE / BLOCK))  # noqa: E731
        self.floor = float(a.get("silence_threshold", 0.015))
        self.hangover = blocks(a.get("hangover_s", 0.7))
        self.min_speech = blocks(a.get("min_speech_s", 0.25))
        self.max_blocks = blocks(a.get("max_utterance_s", 12.0))
        self._pre: deque = deque(maxlen=blocks(a.get("pre_roll_s", 0.3)))
        self._vad = _webrtc_vad(a.get("vad_aggressiveness", 2))
        self._noise = self.floor
        self._speech: list = []
        self._lead = 0  # pre-roll blocks the current utterance started with
        self._silent = 0
        self.in_speech = False
        self.dropped_s = 0.0  # speech in a segment the last push() dropped as too short, else 0

    def _is_speech(self, block: np.ndarray, rms: float) -> bool:
        if self._vad is not None:
            pcm = np.clip(block * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            return self._vad.is_speech(pcm, SAMPLE_RATE)
        if rms >= max(self.floor, 3.0 * self._noise):
            return True
        self._noise = 0.95 * self._noise + 0.05 * rms  # track the floor in silence only
        return False

    def push(self, block: np.ndarray, rms: float | None = None) -> np.ndarray | None:
        self.dropped_s = 0.0
        if rms is None:
            rms = float(np.sqrt(np.mean(np.square(block))))
        speechy = self._is_speech(block, rms)
        if not self.in_speech:
            self._pre.append(block.copy())
            if speechy:
                self.in_speech, self._speech, self._silent = True, list(self._pre), 0
                # The ring restarts empty after each utterance, so speech soon
                # after one starts with less pre-roll than its maximum.
                self._lead = len(self._pre)
            return None
        self._speech.append(block.copy())
        self._silent = 0 if speechy else self._silent + 1
        if self._silent < self.hangover and len(self._speech) < self.max_blocks:
            return None
        audio = np.concatenate(self._speech)
        voiced = len(self._speech) - self._silent - self._lead
        self._speech, self._silent, self.in_speech = [], 0, False
        self._pre.clear()
        if voiced >= self.min_speech:
            return audio
        self.dropped_s = max(voiced, 0) * BLOCK / SAMPLE_RATE
        return None


class MicListener:
    """Microphone thread: each utterance → ``on_text(text)``.

    ``on_level(rms, speaking)`` gets every block (for a level meter) and
    ``transcribing`` is True while the transcript function runs. ``on_drop(reason)`` hears
    about every utterance that never becomes text: speech too short to count,
    a transcript with no words, and microphone audio lost because the CPU was
    too busy to read it. ``opened`` is set once the microphone was tried;
    ``error`` then holds the reason if it could not be opened.
    """

    OVERFLOW_REPORT_S = 2.0  # at most one "audio lost" report per this many seconds

    def __init__(self, cfg: dict, transcribe, on_text, on_level=None, on_drop=None):
        a = cfg.get("audio") or {}
        self.cfg = cfg
        self.device = a.get("mic_device")
        self.grace_s = float(a.get("startup_grace_s", 1.5))
        self.transcribe = transcribe
        self.on_text = on_text
        self.on_level = on_level
        self.on_drop = on_drop
        self.transcribing = False
        self.error: str | None = None
        self.opened = threading.Event()
        self.overflows = 0  # microphone blocks lost since start
        self._blocks: queue.Queue = queue.Queue()  # microphone blocks; None ends the listener
        self._jobs: queue.Queue = queue.Queue()  # utterances for the transcript function; None ends it
        self._thread = threading.Thread(target=self._run, daemon=True, name="mic")

    def _dropped(self, reason: str) -> None:
        logger.info("utterance dropped: %s", reason)
        if self.on_drop is not None:
            try:
                self.on_drop(reason)
            except Exception:
                logger.exception("on_drop failed")

    def _audio_in(self, indata, frames, time_info, status) -> None:
        """sounddevice callback: queue the block; count blocks the device had to throw away."""
        if status.input_overflow:
            self.overflows += 1
        self._blocks.put(indata[:, 0].copy())

    def start(self) -> "MicListener":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._blocks.put(None)
        self._thread.join(timeout=2.0)

    def _decode(self) -> None:
        while (audio := self._jobs.get()) is not None:
            self.transcribing = True
            try:
                t = time.perf_counter()
                text = self.transcribe(audio)
                logger.info(
                    "ASR %.1fs of audio in %.0f ms: %r",
                    len(audio) / SAMPLE_RATE,
                    (time.perf_counter() - t) * 1e3,
                    text,
                )
            except Exception as e:
                logger.exception("transcription failed")
                self._dropped(f"transcription failed ({e})")
                continue
            finally:
                self.transcribing = False
            if text:
                self.on_text(text)
            else:
                self._dropped("heard no words")

    def _run(self) -> None:
        import sounddevice as sd

        seg = Segmenter(self.cfg)
        t = time.perf_counter()
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK,
                device=self.device,
                callback=self._audio_in,
            )
            stream.start()
            logger.info("microphone opened in %.3f s", time.perf_counter() - t)
        except Exception as e:
            self.error = str(e)
            logger.error("microphone unavailable: %s", e)
            return
        finally:
            self.opened.set()
        threading.Thread(target=self._decode, daemon=True, name="asr").start()
        t0 = time.monotonic()
        reported, reported_at = 0, 0.0
        try:
            while (block := self._blocks.get()) is not None:
                rms = float(np.sqrt(np.mean(np.square(block))))
                audio = seg.push(block, rms)
                if self.on_level is not None:
                    self.on_level(rms, seg.in_speech)
                now = time.monotonic()
                if self.overflows > reported and now - reported_at >= self.OVERFLOW_REPORT_S:
                    self._dropped(f"microphone audio lost {self.overflows - reported} times (CPU busy)")
                    reported, reported_at = self.overflows, now
                if now - t0 < self.grace_s:
                    continue
                if seg.dropped_s:
                    self._dropped(f"too short to hear ({seg.dropped_s:.2f} s of speech)")
                if audio is not None:
                    self._jobs.put(audio)
        finally:
            stream.stop()
            stream.close()
            self._jobs.put(None)
