# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The console tile captions live in ``so101_sim.tiles`` and nowhere else."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONSOLES = (
    REPO / "workshop/common/twin_console.py",
    REPO / "vla_pipeline/console.py",
)
CAPTIONS = (
    "mount: YOLOv26s-pose",
    "mount: MediaPipe hands",
    "arm: YOLOv26s-detect",
    "digital twin simulated in MuJoCo",
    "Voice: gemma4-e4b-it transcription",
)


def test_tile_captions_are_only_in_the_shared_module():
    shared = (REPO / "so101_sim/src/so101_sim/tiles.py").read_text()
    for caption in CAPTIONS:
        assert caption in shared
        for path in CONSOLES:
            assert caption not in path.read_text(), path.name
            assert "so101_sim.tiles" in path.read_text()
