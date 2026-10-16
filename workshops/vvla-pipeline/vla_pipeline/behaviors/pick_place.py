# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pick and place. The body is :func:`common.manipulation.run_pick_place`."""

from __future__ import annotations

from types import SimpleNamespace

from vla_pipeline.behaviors._from_workshop import Ctx, command, flatten_cfg
from vla_pipeline.robot.arm_interface import ensure_common_path

ensure_common_path()

from common.manipulation import (  # noqa: E402
    _GraspPlan,
    _box_usable,
    _descent_poses,
    _follow,
    _offer,
    _path_within,
    _search_views,
    _shape_of,
    _sweep_views,
    run_pick_place as _run_pick_place,
)

__all__ = [
    "_GraspPlan",
    "_box_usable",
    "_descent_poses",
    "_follow",
    "_offer",
    "_path_within",
    "_plan_from_view",
    "_search_views",
    "_shape_of",
    "_sweep_views",
    "run_pick_place",
]


def _plan_from_view(cfg, view):
    """``cfg`` is a pipeline config dict or a workshop context."""
    from common.manipulation import _plan_from_view as _workshop_plan

    if isinstance(cfg, dict):
        cfg = SimpleNamespace(cfg=flatten_cfg(cfg))
    return _workshop_plan(cfg, view)


def run_pick_place(
    cfg: dict,
    arm,
    detector,
    pose,
    arm_camera,
    operator_camera,
    object_name: str = "",
    shared_state: dict | None = None,
    stop_check=None,
    feedback=None,
) -> bool:
    """Pick the named object. Delegates to :func:`common.manipulation.run_pick_place`."""
    ctx = Ctx(
        cfg,
        arm,
        say=feedback,
        stop_check=stop_check,
        shared_state=shared_state,
        detector=detector,
        arm_camera=arm_camera,
        camera=operator_camera,
        pose=pose,
    )
    return _run_pick_place(ctx, command(object_name))
