# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The microphone's frames cut into utterances for Gemma.

Frames of 20 ms arrive from the microphone. A constant offset is taken out of
them first. An utterance is the sound from the first frame that counts as
speech (with a little before it) to the moment the speaker has been silent for
``end_silence_ms``: the frame that completes that silence hands the utterance
over, so a sentence is sent the moment it has ended. Gemma takes a whole clip,
not sound as it comes, so the end of the speech is the earliest moment a
sentence can be sent. Nothing here sleeps to discover speech.
"""

from __future__ import annotations

import math
from array import array
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, lfilter_zi

FRAME_MS = 20
OFFSET_CORNER_HZ = 13  # corner of the high-pass that removes the microphone's offset; far below speech


def dbfs(pcm: bytes) -> float:
    """Level of one int16 PCM frame, in decibels relative to full scale."""
    values = array("h", pcm)
    rms = math.sqrt(sum(x * x for x in values) / max(1, len(values)))
    return 20 * math.log10(max(rms / 32768, 1e-6))


class OffsetRemover:
    """Take a constant offset out of the microphone's samples, frame after frame.

    Some microphones deliver samples around a constant instead of around 0
    (measured on a Strix Halo laptop's built-in microphone: +1028, which alone
    reads -30 dBFS while the room is at -60). With the offset in, every frame
    reads about that level whatever is said, and the offset is in the sound the
    speech detector and Gemma get. A first-order high-pass with its corner at
    ``OFFSET_CORNER_HZ``; it starts settled on the first sample, so there is no
    thump.
    """

    def __init__(self, sample_rate: int) -> None:
        self._b = (1.0, -1.0)
        self._a = (1.0, -(1.0 - 2 * math.pi * OFFSET_CORNER_HZ / sample_rate))
        self._state = None

    def reset(self) -> None:
        """Settle anew on the next first sample, after sound was lost."""
        self._state = None

    def feed(self, pcm: bytes) -> bytes:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        if self._state is None:
            self._state = lfilter_zi(self._b, self._a) * samples[0]
        cleaned, self._state = lfilter(self._b, self._a, samples, zi=self._state)
        return np.clip(np.rint(cleaned), -32768, 32767).astype("<i2").tobytes()


def at_level(pcm: bytes, peak: float, most_gain: float) -> bytes:
    """The sound scaled so that its loud samples reach ``peak`` of full scale, by ``most_gain`` at most.

    Gemma hears quiet sound worse (measured: at a tenth of a normal level it
    took one colour word for another). "Loud" is the level one sample in a
    thousand exceeds, so a click does not set the scale; what exceeds full
    scale afterwards is clipped. Silence stays silence: the gain has a ceiling.
    """
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    loud = float(np.percentile(np.abs(samples), 99.9)) if samples.size else 0.0
    if loud <= 0.0:
        return pcm
    gain = min(most_gain, peak * 32767 / loud)
    return np.clip(np.rint(samples * gain), -32768, 32767).astype("<i2").tobytes()


@dataclass(frozen=True)
class Utterance:
    """One sentence's sound, with the capture times that say how long ago it was said."""

    pcm: bytes
    spoke_until: float  # when its last frame of speech was captured: the end of the speech
    ended: float  # when the frame that completed the end silence was captured: it was handed over then


class Utterances:
    """Cut the microphone's frames into utterances; hand each one over on the frame that ends it.

    An utterance begins at the first frame that counts as speech, with
    ``pre_roll_ms`` of the sound before it (the detector answers a little after
    a word begins, and Gemma hears a sentence badly that starts at a clip's
    first sample). It has ended when ``end_silence_ms`` of frames in a row did
    not count as speech: that frame returns it, so nothing waits for a clock.
    One that grows to ``max_s`` is handed over as it is and the next begins.
    One with less than ``min_voice_ms`` of speech in it (a click, a cough) is
    dropped.
    """

    def __init__(
        self, sample_rate: int, max_s: float, end_silence_ms: int, pre_roll_ms: int, min_voice_ms: int
    ) -> None:
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self.max_frames = round(max_s * 1000 / FRAME_MS)
        self.end_silence_frames = max(1, round(end_silence_ms / FRAME_MS))
        self.min_voice_frames = max(1, round(min_voice_ms / FRAME_MS))
        self.before: deque[bytes] = deque(maxlen=max(1, round(pre_roll_ms / FRAME_MS)))
        self.clear()

    def clear(self) -> None:
        """Forget the sound collected so far, after sound was lost."""
        self.before.clear()
        self.frames: list[bytes] = []  # the utterance under way; empty between utterances
        self.voiced = 0  # its frames that counted as speech
        self.silent = 0  # its frames in a row, at its end, that did not
        self.spoke_until = 0.0  # when its newest frame of speech was captured

    @property
    def speaking(self) -> bool:
        """Whether an utterance is under way."""
        return bool(self.frames)

    def feed(self, pcm: bytes, qualified: bool, captured_at: float) -> Utterance | None:
        """Take one 20 ms frame. Return the utterance on the frame that ends it."""
        if len(pcm) != self.frame_bytes:
            raise ValueError("Invalid PCM frame length")
        if not self.frames:
            if not qualified:
                self.before.append(pcm)
                return None
            self.frames = [*self.before, pcm]
            self.before.clear()
            self.voiced, self.silent, self.spoke_until = 1, 0, captured_at
            return None
        self.frames.append(pcm)
        if qualified:
            self.voiced += 1
            self.silent = 0
            self.spoke_until = captured_at
        else:
            self.silent += 1
        if self.silent < self.end_silence_frames and len(self.frames) < self.max_frames:
            return None
        ended, voiced, spoke_until = self.frames, self.voiced, self.spoke_until
        self.frames, self.voiced, self.silent = [], 0, 0
        # The end of this utterance is the sound before the next one.
        self.before.extend(ended[-self.before.maxlen:])
        if voiced < self.min_voice_frames:
            return None
        return Utterance(b"".join(ended), spoke_until, captured_at)
