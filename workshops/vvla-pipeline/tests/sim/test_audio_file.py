# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``--audio-file``: recorded WAV commands reach the dispatcher in order, with no microphone.

A fake ``transcribe`` stands in for Whisper, so no model is loaded. The WAVs
are written at 8 kHz, so the resampling to 16 kHz runs too.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import soundfile as sf

from vla_pipeline.audio.file_source import feed_audio_files
from vla_pipeline.audio.whisper_npu import SAMPLE_RATE

pytestmark = pytest.mark.sim


def _wav(path, seconds: float = 0.5, rate: int = 8000):
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    return str(path)


def test_files_are_dispatched_in_order_and_empty_text_is_skipped(tmp_path):
    paths = [_wav(tmp_path / "a.wav"), _wav(tmp_path / "silence.wav"), _wav(tmp_path / "b.wav")]
    replies = iter(["pick up the ball", "  ", "go home"])
    lengths, heard, idle = [], [], []

    def transcribe(audio):
        lengths.append(len(audio))
        return next(replies)

    texts = feed_audio_files(paths, transcribe, heard.append, lambda t: idle.append(t) or True, gap_s=0.0)

    assert texts == ["pick up the ball", "", "go home"]
    assert heard == ["pick up the ball", "go home"]
    assert idle == [None, None, None]
    assert lengths == [SAMPLE_RATE // 2] * 3  # 0.5 s at 8 kHz, resampled to 16 kHz


def test_shutdown_stops_before_the_next_file(tmp_path):
    paths = [_wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")]
    shutdown = threading.Event()
    heard = []

    def on_transcript(text):
        heard.append(text)
        shutdown.set()

    texts = feed_audio_files(paths, lambda a: "pick up the ball", on_transcript, shutdown=shutdown, gap_s=0.0)

    assert texts == ["pick up the ball"]
    assert heard == ["pick up the ball"]
