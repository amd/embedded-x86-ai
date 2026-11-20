# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Recorded WAV commands through Whisper, for machines with no microphone.

``python -m vla_pipeline.main --audio-file cmd.wav`` uses this instead of the
VAD listener: each file is transcribed, its text goes to the dispatcher, and
the next file waits until that command's behavior has ended.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from vla_pipeline.audio.whisper_npu import _load_wav

logger = logging.getLogger(__name__)


def feed_audio_files(
    paths: list[str],
    transcribe: Callable,
    on_transcript: Callable[[str], None],
    wait_idle: Callable[[float | None], bool] | None = None,
    shutdown: threading.Event | None = None,
    gap_s: float = 1.0,
) -> list[str]:
    """Transcribe each WAV in order and hand its text to ``on_transcript``.

    After each file, wait for ``wait_idle(None)`` (the behavior has ended) and
    then ``gap_s``. Empty transcripts are logged but not dispatched. Stops early
    when ``shutdown`` is set. Returns the transcripts, empty ones included.
    """
    texts = []
    for path in paths:
        if shutdown is not None and shutdown.is_set():
            break
        text = transcribe(_load_wav(path)).strip()
        logger.info("[audio-file] %s -> %s", Path(path).name, text)
        texts.append(text)
        if text:
            on_transcript(text)
        if wait_idle is not None:
            wait_idle(None)
        if shutdown is not None:
            shutdown.wait(gap_s)
        else:
            time.sleep(gap_s)
    return texts
