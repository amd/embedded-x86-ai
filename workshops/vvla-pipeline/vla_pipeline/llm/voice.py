# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The one speech front end for the pipeline and the digital-twin notebook.

``voice_intent`` in ``config/pipeline.yaml`` is ``gemma4-e4b-it``. The
microphone's sound is cut into utterances; each is sent to Gemma 4 on
llama-server the moment the speaker has been silent for ``end_silence_ms``,
one request per sentence. A command Gemma finds is reported twice: the call
(command and object) the moment Gemma has written it, to ``on_text``; the
words Gemma heard when the answer is complete, to ``on_heard``.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import queue
import threading
import time
import wave

from vla_pipeline.audio.gemma_listen import FRAME_MS, OffsetRemover, Utterances, at_level, dbfs
from vla_pipeline.llm.intents import Intent, ParsedCommand, keyword_parse
from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)

VOICE_INTENTS = ("gemma4-e4b-it",)
WAITING_UTTERANCES = 4  # utterances that may wait for Gemma; the oldest goes when one more arrives
_STOP = None  # the sentinel that ends the request thread


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


class VoiceSession:
    """A parser plus the microphone thread that feeds it."""

    def __init__(self, kind: str, parse, llm=None, error: str | None = None):
        self.kind = kind
        self.parse = parse
        self.llm = llm
        self.error = error
        self.opened = threading.Event()
        self.transcribing = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self, on_text, on_level=None, on_busy=None, on_drop=None, stop_check=None, on_heard=None,
    ) -> "VoiceSession":
        """Begin listening.

        ``on_text`` receives a command as a JSON string the moment Gemma has written the call
        (command and object); the words heard are still being written, so its ``heard`` is
        empty. ``on_heard`` then receives those words when the answer is complete: once for
        every command delivered, empty when Gemma made out none. Without ``on_heard`` the
        request ends at the call.
        """
        if self.error:
            self.opened.set()
            return self
        target = self._gemma
        self._thread = threading.Thread(
            target=target, args=(on_text, on_level, on_busy, on_drop, stop_check, on_heard),
            daemon=True, name="voice",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Ask the microphone thread to exit and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def close(self) -> None:
        """Stop the microphone and the language server this session started."""
        self.stop()
        if self.llm is not None:
            self.llm.close()

    def _gemma(self, on_text, on_level, on_busy, on_drop, stop_check, on_heard) -> None:
        import sounddevice as sd

        g = self._gemma_cfg
        rate = int((self._cfg.get("audio") or {}).get("sample_rate", 16000))
        samples = rate * FRAME_MS // 1000
        utterances = Utterances(
            rate, float(g["window_s"]), int(g["end_silence_ms"]), int(g["pre_roll_ms"]), int(g["min_voice_ms"]),
        )
        offset = OffsetRemover(rate)
        sent_peak, most_gain = float(g["sent_peak"]), float(g["most_gain"])
        # Utterances that wait while Gemma answers an earlier one. Each is a sentence of its own,
        # so they are kept; bounded, the oldest goes when one more arrives.
        pending: queue.Queue = queue.Queue(maxsize=WAITING_UTTERANCES)

        def put(item) -> None:
            """Queue one item without blocking the microphone; when the queue is full the oldest goes."""
            while True:
                try:
                    pending.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        pending.get_nowait()
                    except queue.Empty:
                        pass

        try:
            vad = _Silero(str(resolve(g["vad_model"])), float(g["vad_threshold"]))
        except Exception as exc:
            self.error = str(exc)
            self.opened.set()
            logger.error("Silero VAD unavailable: %s", exc)
            return
        threshold = float(g["min_dbfs"])
        max_age = float(g["max_age_s"])

        def worker() -> None:
            while True:
                utterance = pending.get()  # blocks on the next utterance or the stop sentinel
                if utterance is _STOP:
                    return
                if time.monotonic() - utterance.ended > max_age:
                    continue
                if on_busy is not None:
                    on_busy(True)
                self.transcribing = True
                delivered = []  # the call of this utterance, when it holds a command

                def call_written(call: ParsedCommand, seconds: float) -> bool:
                    # Runs inside the request, while Gemma still writes the words it heard.
                    if call.intent is Intent.UNKNOWN:
                        return False  # no command: its words are not read, the next utterance is asked sooner
                    delivered.append(call)
                    logger.info(
                        "Command %r %s %.2f s after the end of the speech (request %.2f s to the call)",
                        call.intent.value, call.params or "", time.monotonic() - utterance.spoke_until, seconds,
                    )
                    on_text(json.dumps({"command": call.intent.value, "object": call.object_name or "", "heard": ""}))
                    return on_heard is not None  # the words are read only when someone shows them

                words = ""
                try:
                    words = self.llm.parse_audio(_wav(utterance.pcm, rate), on_call=call_written).transcript
                except Exception:
                    logger.exception("Gemma audio request failed")
                finally:
                    self.transcribing = False
                    if on_busy is not None:
                        on_busy(False)
                # Once for every command delivered, also when Gemma made out no words or the
                # request failed after its call: a listener can wait for it.
                if delivered and on_heard is not None:
                    on_heard(words)

        threading.Thread(target=worker, daemon=True, name="gemma-audio").start()
        try:
            try:
                # latency "low": the sound library's default for input is its "high" figure
                # (34.8 ms against 8.7 ms on a Strix Halo laptop's default device).
                stream = sd.RawInputStream(
                    samplerate=rate, blocksize=samples, channels=1, dtype="int16",
                    device=(self._cfg.get("audio") or {}).get("mic_device"), latency="low",
                )
            except Exception as exc:
                self.error = str(exc)
                self.opened.set()
                logger.error("Microphone unavailable: %s", exc)
                return
            self.opened.set()
            with stream:
                while not self._stop.is_set() and not (stop_check is not None and stop_check()):
                    buffer, overflow = stream.read(samples)  # blocks on sound
                    now = time.monotonic()
                    if overflow:
                        # Sound was lost between two reads: what was collected does not continue.
                        utterances.clear()
                        vad.reset()
                        offset.reset()
                        continue
                    # The offset comes out first: the level, the detector and Gemma get the sound itself.
                    pcm = offset.feed(bytes(buffer))
                    level = dbfs(pcm)
                    prob = vad.feed(pcm, rate)
                    qualified = prob >= vad.threshold and level >= threshold
                    if on_level is not None:
                        on_level(max(0.0, min(1.0, (level + 60) / 60)), qualified)
                    # The frame that ends an utterance hands it over: it goes to Gemma now, raised
                    # to a fixed level.
                    utterance = utterances.feed(pcm, qualified, now)
                    if utterance is not None:
                        put(dataclasses.replace(utterance, pcm=at_level(utterance.pcm, sent_peak, most_gain)))
        finally:
            put(_STOP)  # the request thread ends after the request under way


class _Silero:
    """Streaming Silero voice-activity detection on the CPU."""

    def __init__(self, model_path: str, threshold: float):
        import os

        import numpy as np
        import onnxruntime

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, providers=("CPUExecutionProvider",), sess_options=options,
        )
        self._outputs = tuple(item.name for item in self._session.get_outputs())
        self.threshold = threshold
        self._np = np
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._buffer = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self.prob = 0.0

    def feed(self, pcm: bytes, sample_rate: int) -> float:
        np = self._np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        stride = max(1, sample_rate // 16000)
        if stride > 1:
            samples = samples[::stride]
        self._buffer = np.concatenate((self._buffer, samples))
        while self._buffer.size >= 512:
            chunk = self._buffer[:512]
            audio = np.concatenate((self._context, chunk))[np.newaxis]
            outputs = self._session.run(
                self._outputs, {"input": audio, "state": self._state, "sr": np.array(16000, np.int64)},
            )
            self._state, self._context, self.prob = outputs[1], chunk[-64:], float(outputs[0].reshape(-1)[0])
            self._buffer = self._buffer[512:]
        return self.prob


def open_voice(cfg: dict, *, device=None) -> VoiceSession:
    """Build the speech front end selected by ``cfg['voice_intent']``."""
    kind = cfg.get("voice_intent")
    if kind not in VOICE_INTENTS:
        raise ValueError(f"voice_intent must be gemma4-e4b-it, not {kind!r}")
    try:
        from vla_pipeline.llm.llama_intent import LlamaIntentParser

        llm = LlamaIntentParser(cfg, section="gemma4-e4b-it")
    except FileNotFoundError as exc:
        logger.warning("Gemma is not ready (%s). Typed commands use the keyword match.", exc)
        session = VoiceSession(kind, keyword_parse, error=str(exc))
        return session
    # The first request pays for the instructions being read into the model: here, not at
    # the user's first command.
    logger.info("Gemma warmed up with one request, %.1f s", llm.warm_up())
    session = VoiceSession(kind, llm.parse, llm)
    session._cfg = cfg
    session._gemma_cfg = cfg["gemma4-e4b-it"]
    return session
