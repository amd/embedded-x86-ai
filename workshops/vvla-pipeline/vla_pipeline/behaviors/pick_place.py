# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pick and place. The body is :func:`vla_pipeline.behaviors.manipulation.run_pick_place`."""

from __future__ import annotations

from types import SimpleNamespace

from vla_pipeline.behaviors._ctx import Ctx, command, flatten_cfg
from vla_pipeline.behaviors.manipulation import (
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
    """``cfg`` is a pipeline config dict or a behavior context."""
    from vla_pipeline.behaviors.manipulation import _plan_from_view as _ctx_plan

    if isinstance(cfg, dict):
        cfg = SimpleNamespace(cfg=flatten_cfg(cfg))
    return _ctx_plan(cfg, view)


def run_pick_place(
    cfg: dict,
    arm,
    detector,
    pose,
    arm_camera,
    mount_camera,
    object_name: str = "",
    shared_state: dict | None = None,
    stop_check=None,
    feedback=None,
) -> bool:
    """Pick the named object. Delegates to :func:`vla_pipeline.behaviors.manipulation.run_pick_place`."""
    ctx = Ctx(
        cfg,
        arm,
        say=feedback,
        stop_check=stop_check,
        shared_state=shared_state,
        detector=detector,
        arm_camera=arm_camera,
        camera=mount_camera,
        pose=pose,
    )
    return _run_pick_place(ctx, command(object_name))
