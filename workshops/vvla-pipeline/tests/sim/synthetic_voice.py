# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A synthetic voice (espeak-ng) as a quiet built-in microphone delivers it.

The tests that speak to the listener use this: ``tests/sim/test_voice_live.py``
(the speech front end) and ``tests/sim/test_twin_notebook_voice.py`` (the
notebook console). Needs ``libespeak-ng1``; a missing library fails the test.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ESPEAK_ONCE = Path(__file__).with_name("espeak_once.py")
RATE = 16000
OFFSET = 1028       # the constant a quiet built-in microphone was measured to add to its samples
SPEECH_PEAK = 0.1   # of full scale: quiet speech


def spoken(text: str) -> np.ndarray:
    """The sentence in espeak-ng's voice at 16 kHz, trimmed, at SPEECH_PEAK. A process per sentence:
    espeak-ng keeps state between sentences, and the same sentence must be the same sound every time."""
    ran = subprocess.run(
        [sys.executable, str(ESPEAK_ONCE), "libespeak-ng.so.1", "en-us", "150", text],
        capture_output=True, timeout=60,
    )
    assert ran.returncode == 0 and len(ran.stdout) > 4, (
        "espeak-ng gave no sound (sudo apt install libespeak-ng1): " + ran.stderr.decode("utf-8", "replace")
    )
    rate = struct.unpack("<I", ran.stdout[:4])[0]
    sound = np.frombuffer(ran.stdout[4:], "<i2").astype(np.float64)
    loud = np.flatnonzero(np.abs(sound) > 0.01 * np.abs(sound).max())
    sound = resample_poly(sound[loud[0]:loud[-1] + 1], RATE, rate)
    return sound * (SPEECH_PEAK * 32767 / np.abs(sound).max())


def microphone_bytes(sound: np.ndarray) -> bytes:
    """``sound`` as the microphone's 16-bit samples: its constant offset is on every one."""
    return np.round(sound + OFFSET).astype("<i2").tobytes()
