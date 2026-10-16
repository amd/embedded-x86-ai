# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Grip and release. The bodies are :mod:`common.manipulation`."""

from __future__ import annotations

from vla_pipeline.behaviors._from_workshop import Ctx
from vla_pipeline.robot.arm_interface import ensure_common_path

ensure_common_path()

from common.manipulation import (  # noqa: E402
    PRESENT_POSE,
    close_with_feedback,
    ease_grip,
    holding,
    run_grip as _run_grip,
    run_release as _run_release,
)

__all__ = [
    "PRESENT_POSE",
    "close_with_feedback",
    "ease_grip",
    "holding",
    "run_grip",
    "run_release",
]


def run_grip(
    cfg: dict,
    arm,
    shared_state: dict | None = None,
    stop_check=None,
    wait_s: float = 4.0,
    arm_camera=None,
    feedback=None,
) -> bool:
    """Take an object from the user. Delegates to :func:`common.manipulation.run_grip`."""
    ctx = Ctx(
        cfg,
        arm,
        say=feedback,
        stop_check=stop_check,
        shared_state=shared_state,
        arm_camera=arm_camera,
    )
    return _run_grip(ctx, None, wait_s=wait_s)


def run_release(arm, shared_state: dict | None = None, stop_check=None, feedback=None) -> bool:
    """Open the jaws where they are. Delegates to :func:`common.manipulation.run_release`."""
    ctx = Ctx({}, arm, say=feedback, stop_check=stop_check, shared_state=shared_state)
    return _run_release(ctx)
