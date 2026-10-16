# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pipeline gesture mimic uses :class:`common.motion.MimicMapper`.

:meth:`CameraMimicMapper.step` accepts the keyword names
:mod:`vla_pipeline.behaviors.gesture_mimic` already passes.
"""

from __future__ import annotations

from vla_pipeline.robot.arm_interface import ensure_common_path

ensure_common_path()

from common.motion import MimicMapper  # noqa: E402


class CameraMimicMapper(MimicMapper):
    """:class:`MimicMapper` with the pipeline step() keyword names."""

    def step(self, wrist_xy, frame_w, frame_h, **kwargs):
        if "hand_roll_deg" in kwargs:
            kwargs.setdefault("roll_fallback", kwargs.pop("hand_roll_deg"))
        if "roll_ti_deg" in kwargs:
            kwargs["roll_ti"] = kwargs.pop("roll_ti_deg")
        if "roll_ref_deg" in kwargs:
            kwargs["roll_ref"] = kwargs.pop("roll_ref_deg")
        return super().step(wrist_xy, frame_w, frame_h, **kwargs)


def mapper_from_config(cfg: dict) -> CameraMimicMapper:
    """Build a mapper from ``behaviors.gesture_mimic`` or a top-level ``mimic``."""
    behaviors = cfg.get("behaviors") or {}
    section = behaviors.get("gesture_mimic") or cfg.get("mimic") or {}
    return CameraMimicMapper(section)
