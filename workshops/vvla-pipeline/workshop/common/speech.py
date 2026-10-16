# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Speech in: microphone → voice-activity detection → Whisper → text (PROVIDED).

The real pipeline's speech front end, self-contained for the workshop:

- :class:`Segmenter` cuts a 30 ms block stream into utterances. A pre-roll
  ring keeps the first syllable that arrives while the detector is still
  deciding; ``hangover_s`` of detector-silence ends an utterance. Anything
  shorter than ``min_speech_s`` (a click, a cough) is dropped. ``webrtcvad``
  decides speech when installed, else an energy detector riding 3x above the
  adaptive noise floor.
- :class:`Whisper` is the ONNX encoder/decoder pair (``whisper:`` in the
  config), greedy-decoded on the CPU.
- :class:`MicListener` runs both on a thread and hands every transcript to a
  callback - in the notebooks, straight into your ``handle_transcript``.

Whisper hallucinates on silence ("Thank you.", "[BLANK_AUDIO]"); the noise
gate in :func:`common.intents.is_noise` drops those before the LLM.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

from common.config import resolve

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
        self._silent = 0
        self.in_speech = False

    def _is_speech(self, block: np.ndarray, rms: float) -> bool:
        if self._vad is not None:
            pcm = np.clip(block * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            return self._vad.is_speech(pcm, SAMPLE_RATE)
        if rms >= max(self.floor, 3.0 * self._noise):
            return True
        self._noise = 0.95 * self._noise + 0.05 * rms  # track the floor in silence only
        return False

    def push(self, block: np.ndarray, rms: float | None = None) -> np.ndarray | None:
        if rms is None:
            rms = float(np.sqrt(np.mean(np.square(block))))
        speechy = self._is_speech(block, rms)
        if not self.in_speech:
            self._pre.append(block.copy())
            if speechy:
                self.in_speech, self._speech, self._silent = True, list(self._pre), 0
            return None
        self._speech.append(block.copy())
        self._silent = 0 if speechy else self._silent + 1
        if self._silent < self.hangover and len(self._speech) < self.max_blocks:
            return None
        audio = np.concatenate(self._speech)
        voiced = len(self._speech) - self.hangover - self._pre.maxlen
        self._speech, self._silent, self.in_speech = [], 0, False
        self._pre.clear()
        return audio if voiced >= self.min_speech else None


def _pretrained(cls, model_id: str):
    """Load a public Hugging Face processor from the local cache first.

    A saved login whose token has expired makes the Hub answer 401, and that
    error hides a complete cache. Download only when the cache does not have it.
    """
    try:
        return cls.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return cls.from_pretrained(model_id)


class Whisper:
    """Whisper ONNX encoder + decoder, greedy decoding (one utterance at a time)."""

    def __init__(self, cfg: dict):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        w = cfg["whisper"]
        cpu = ["CPUExecutionProvider"]
        # Spinning threads keep cores busy between ops without making a
        # transcription faster.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(w.get("cpu_threads", 4))
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.encoder = ort.InferenceSession(str(resolve(w["encoder_onnx"])), opts, providers=cpu)
        self.decoder = ort.InferenceSession(str(resolve(w["decoder_onnx"])), opts, providers=cpu)
        self.provider = "CPU"

        # The tokenizer must match the checkpoint: the base model is the
        # English-only one, so openai/whisper-base.en.
        self.features = _pretrained(WhisperFeatureExtractor, w["tokenizer"])
        self.tokenizer = _pretrained(WhisperTokenizer, w["tokenizer"])
        if w.get("language") and not w["tokenizer"].endswith(".en"):
            self.tokenizer.set_prefix_tokens(language=w["language"], task="transcribe")
        self.prefix = list(self.tokenizer.prefix_tokens)
        self.eos = self.tokenizer.eos_token_id

        ins = self.decoder.get_inputs()
        ids, enc = ins[0], ins[1]
        if ids.type != "tensor(int64)":
            ids, enc = enc, ids
        self._ids, self._enc = ids.name, enc.name
        self.max_len = min(448, int(ids.shape[1]))  # the decoder is exported fixed-length

    def transcribe(self, audio: np.ndarray) -> str:
        feats = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        enc_out = self.encoder.run(
            None, {self.encoder.get_inputs()[0].name: feats["input_features"]}
        )[0]
        tokens = list(self.prefix)
        while len(tokens) < self.max_len:
            ids = np.full((1, self.max_len), self.eos, dtype=np.int64)
            ids[0, : len(tokens)] = tokens
            logits = self.decoder.run(None, {self._ids: ids, self._enc: enc_out})[0]
            nxt = int(np.argmax(logits[0, len(tokens) - 1]))
            if nxt == self.eos:
                break
            tokens.append(nxt)
        return self.tokenizer.decode(tokens[len(self.prefix) :], skip_special_tokens=True).strip()


class MicListener:
    """Microphone thread: each utterance → ``on_text(text)``.

    ``on_level(rms, speaking)`` gets every block (for a level meter) and
    ``transcribing`` is True while Whisper runs. ``error`` holds the reason if
    the microphone could not be opened.
    """

    def __init__(self, cfg: dict, transcribe, on_text, on_level=None):
        a = cfg.get("audio") or {}
        self.cfg = cfg
        self.device = a.get("mic_device")
        self.grace_s = float(a.get("startup_grace_s", 1.5))
        self.transcribe = transcribe
        self.on_text = on_text
        self.on_level = on_level
        self.transcribing = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mic")

    def start(self) -> "MicListener":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import sounddevice as sd

        blocks: queue.Queue = queue.Queue()
        seg = Segmenter(self.cfg)
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK,
                device=self.device,
                callback=lambda indata, *_: blocks.put(indata[:, 0].copy()),
            )
        except Exception as e:
            self.error = str(e)
            logger.error("microphone unavailable: %s", e)
            return
        t0 = time.monotonic()
        jobs: queue.Queue = queue.Queue()

        def _decode() -> None:
            while not self._stop.is_set():
                try:
                    audio = jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
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
                except Exception:
                    logger.exception("transcription failed")
                    text = ""
                finally:
                    self.transcribing = False
                if text:
                    self.on_text(text)

        threading.Thread(target=_decode, daemon=True, name="asr").start()
        with stream:
            while not self._stop.is_set():
                try:
                    block = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(block))))
                audio = seg.push(block, rms)
                if self.on_level is not None:
                    self.on_level(rms, seg.in_speech)
                if audio is None or time.monotonic() - t0 < self.grace_s:
                    continue
                jobs.put(audio)
