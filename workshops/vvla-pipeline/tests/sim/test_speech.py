# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The microphone front end: which sounds reach Whisper, and how the rest is reported.

Synthetic blocks through the energy detector (``webrtcvad`` is switched off):
a loud block is speech, a silent one is not. A fake ``sounddevice`` stream
feeds :class:`common.speech.MicListener` the same blocks, so no microphone
is needed.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

pytestmark = pytest.mark.sim

CFG = {"audio": {"hangover_s": 1.2, "min_speech_s": 0.25, "pre_roll_s": 0.3, "startup_grace_s": 0.0}}


def _blocks():
    from common.speech import BLOCK

    quiet = np.zeros(BLOCK, np.float32)
    loud = (0.2 * np.sin(np.linspace(0, 60 * np.pi, BLOCK))).astype(np.float32)
    return quiet, loud


def _utterance(ms: int) -> list:
    """``ms`` of speech, then enough silence to end it."""
    quiet, loud = _blocks()
    return [loud] * (ms // 30) + [quiet] * 45


def _segments(blocks) -> tuple[list, list]:
    """Utterances the segmenter keeps (in blocks) and the speech (s) of those it drops."""
    from common.speech import BLOCK, Segmenter

    seg = Segmenter(CFG)
    seg._vad = None
    kept, dropped = [], []
    for block in blocks:
        audio = seg.push(block)
        if audio is not None:
            kept.append(len(audio) // BLOCK)
        if seg.dropped_s:
            dropped.append(seg.dropped_s)
    return kept, dropped


def test_short_command_after_silence_is_kept():
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(270))
    assert len(kept) == 1 and not dropped


def test_short_command_right_after_an_earlier_sound_is_kept():
    """The pre-roll ring restarts empty after an utterance; a command 90 ms
    later still counts all of its speech."""
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(360) + [quiet] * 3 + _utterance(270))
    assert len(kept) == 2, f"kept {kept}, dropped {dropped}"


def test_click_is_dropped_and_measured():
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(120))
    assert not kept
    assert dropped and 0.05 < dropped[0] < 0.25, dropped


class _FakeStream:
    """``sounddevice.InputStream``: plays ``script`` blocks into the callback back to back
    (``MicListener`` queues every block, unbounded)."""

    script: list = []

    def __init__(self, callback=None, **_kwargs):
        self._callback = callback
        self._thread = threading.Thread(target=self._play, daemon=True)

    def _play(self):
        for block, overflow in self.script:
            self._callback(block[:, None], len(block), None, types.SimpleNamespace(input_overflow=overflow))

    def start(self):
        self._thread.start()

    def stop(self):
        pass

    def close(self):
        pass


def test_mic_listener_reports_what_never_becomes_text(monkeypatch):
    from common import speech

    quiet, _ = _blocks()
    script = [(b, False) for b in [quiet] * 20 + _utterance(120) + _utterance(450)]
    script[5] = (quiet, True)  # the device lost a block while the CPU was busy
    _FakeStream.script = script
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_FakeStream))
    monkeypatch.setattr(speech, "_webrtc_vad", lambda _aggressiveness: None)
    monkeypatch.setattr(speech.MicListener, "OVERFLOW_REPORT_S", 0.0)

    texts, drops = [], []
    three = threading.Event()

    def dropped(reason):
        drops.append(reason)
        if len(drops) >= 3:
            three.set()

    mic = speech.MicListener(CFG, lambda audio: "", texts.append, on_drop=dropped).start()
    mic.opened.wait()
    assert mic.error is None, mic.error
    three.wait()  # the listener's own reports: too short, no words, audio lost
    mic.stop()
    assert not texts
    assert any(d.startswith("too short to hear") for d in drops), drops
    assert "Whisper heard no words" in drops, drops
    assert any(d.startswith("microphone audio lost") for d in drops), drops
