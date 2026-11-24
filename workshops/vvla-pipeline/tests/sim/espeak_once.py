# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One sentence from espeak-ng, in a process of its own: the samples to standard output.

    python espeak_once.py <library> <voice> <words per minute> <text>

Standard output: the sample rate as four bytes (little endian), then the sentence
as 16-bit samples. Run as a file by ``test_voice_live.py``; standard library
only, so it starts in a few hundredths of a second.

Why a process per sentence: espeak-ng keeps state from one sentence to the next
(measured 2026-11-23, libespeak-ng 1.51: the same sentence asked for twice from
one process differs in 45,347 of its 96,000 samples), and it does not come back
from ``espeak_Terminate`` in the same process. A fresh process gives the same
samples for the same sentence every time.
"""

from __future__ import annotations

import ctypes
import struct
import sys


def synthesise(library: str, voice: str, words_per_minute: int, text: str) -> tuple[int, bytes]:
    """(sample rate, 16-bit samples) of the sentence; SystemExit with the reason when espeak-ng refuses."""
    lib = ctypes.CDLL(library)
    lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.espeak_Initialize.restype = ctypes.c_int
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    lib.espeak_SetVoiceByName.restype = ctypes.c_int
    lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.espeak_SetParameter.restype = ctypes.c_int
    lib.espeak_Synth.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
    ]
    lib.espeak_Synth.restype = ctypes.c_int
    lib.espeak_Synchronize.restype = ctypes.c_int
    rate = lib.espeak_Initialize(2, 0, None, 0)  # 2 = AUDIO_OUTPUT_SYNCHRONOUS: samples go to the callback
    if rate <= 0:
        raise SystemExit(f"espeak_Initialize returned {rate}")
    if lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise SystemExit(f"espeak-ng has no voice {voice!r}")
    if lib.espeak_SetParameter(1, words_per_minute, 0) != 0:  # 1 = espeakRATE
        raise SystemExit("espeak-ng did not take the speaking rate")
    chunks = []

    def collect(samples, count, _events):
        if samples and count > 0:
            chunks.append(ctypes.string_at(samples, count * 2))
        return 0

    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
    callback = callback_type(collect)
    lib.espeak_SetSynthCallback(callback)
    data = text.encode("utf-8")
    # position 0, counted in characters (1), no end position, text is UTF-8 (flag 1)
    if lib.espeak_Synth(data, len(data) + 1, 0, 1, 0, 1, None, None) != 0:
        raise SystemExit(f"espeak-ng could not speak {text!r}")
    lib.espeak_Synchronize()
    if not chunks:
        raise SystemExit(f"espeak-ng gave no sound for {text!r}")
    return rate, b"".join(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    sample_rate, samples = synthesise(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    sys.stdout.buffer.write(struct.pack("<I", sample_rate) + samples)
