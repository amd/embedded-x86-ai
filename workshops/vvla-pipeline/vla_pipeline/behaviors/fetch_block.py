# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Fetch the taught block. The body is :func:`vla_pipeline.behaviors.manipulation.run_fetch_block`."""

from __future__ import annotations

from vla_pipeline.behaviors._ctx import Ctx
from vla_pipeline.behaviors.manipulation import park_open, run_fetch_block as _run_fetch_block

__all__ = ["park_open", "run_fetch_block"]


def run_fetch_block(
    cfg: dict,
    arm,
    pose,
    mount_camera,
    stop_check=None,
    feedback=None,
    stop_event=None,
) -> bool:
    """Grasp the block at its taught pose and hand it over."""
    ctx = Ctx(
        cfg,
        arm,
        say=feedback,
        stop_check=stop_check,
        stop_event=stop_event,
        camera=mount_camera,
        pose=pose,
    )
    return _run_fetch_block(ctx)
