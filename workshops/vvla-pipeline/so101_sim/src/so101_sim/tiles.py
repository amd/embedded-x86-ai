# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Captions on the twin console tiles.

The notebook console and ``run_pipeline.sh`` both draw these. A processor
name, when there is one, is added by :func:`with_processor`.
"""

from __future__ import annotations

POSE = "mount: YOLOv26s-pose"
HANDS = "mount: MediaPipe hands"
DETECT = "arm: YOLOv26s-detect"
VIEW = "digital twin simulated in MuJoCo"
VOICE = (
    "Voice: gemma4-e4b-it transcription",
    "and intent-to-command translation",
)


def with_processor(caption: str, processor: str) -> str:
    """``caption`` plus the processor it is running on, in brackets."""
    return f"{caption} [{processor}]"
