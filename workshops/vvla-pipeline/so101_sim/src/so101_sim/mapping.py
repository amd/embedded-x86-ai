# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Calibrated joint space (what LeRobot reports) <-> MuJoCo joint radians.

Body joints run in LeRobot DEGREES mode: 0 deg is the calibration midpoint,
which on the SO-101 is the homing L-pose - also the MJCF zero. The gripper
runs in 0-100 RANGE mode: the calibrated jaw floor (``closed_cal``) maps to
jaws touching, 95 to fully open. The map is linear in both cases.

Raw encoder counts (``normalize=False`` reads/writes) follow LeRobot's
normalization: body ``deg = (raw - 2047.5) * 360 / 4095``; gripper
``0..100`` spans ``counts_span`` counts.
"""

from __future__ import annotations

import math

from so101_sim import MOTOR_NAMES

BODY_JOINTS = MOTOR_NAMES[:5]
_RES = 4096
_MID = (_RES - 1) / 2.0
_DEG_PER_COUNT = 360.0 / (_RES - 1)


class JointMap:
    """Bidirectional calibrated-unit / radian conversion for all six motors."""

    def __init__(self, joint_map: dict, gripper_map: dict):
        self._sign = {}
        self._off = {}
        for name in BODY_JOINTS:
            m = joint_map.get(name, {})
            self._sign[name] = float(m.get("sign", 1.0))
            self._off[name] = float(m.get("offset_deg", 0.0))
        g = gripper_map
        self.g_closed_cal = float(g["closed_cal"])
        self.g_open_cal = float(g["open_cal"])
        self.g_closed_rad = float(g["closed_rad"])
        self.g_open_rad = float(g["open_rad"])
        self.g_counts_span = float(g.get("counts_span", 2271))
        self._g_rad_per_unit = (self.g_open_rad - self.g_closed_rad) / (
            self.g_open_cal - self.g_closed_cal
        )

    # -- calibrated <-> radians ------------------------------------------------
    def to_rad(self, name: str, cal: float) -> float:
        if name == "gripper":
            return self.g_closed_rad + (float(cal) - self.g_closed_cal) * (
                self._g_rad_per_unit
            )
        return self._sign[name] * math.radians(float(cal) - self._off[name])

    def to_cal(self, name: str, rad: float) -> float:
        if name == "gripper":
            return self.g_closed_cal + (float(rad) - self.g_closed_rad) / (
                self._g_rad_per_unit
            )
        return math.degrees(float(rad)) * self._sign[name] + self._off[name]

    # -- calibrated <-> raw encoder counts (normalize=False) ------------------
    def cal_to_raw(self, name: str, cal: float) -> int:
        if name == "gripper":
            return int(round(_MID + (float(cal) - 50.0) / 100.0 * self.g_counts_span))
        return int(round(_MID + float(cal) / _DEG_PER_COUNT))

    def raw_to_cal(self, name: str, raw: float) -> float:
        if name == "gripper":
            return 50.0 + (float(raw) - _MID) / self.g_counts_span * 100.0
        return (float(raw) - _MID) * _DEG_PER_COUNT

    def quantize(self, name: str, cal: float) -> float:
        """Snap to the encoder resolution, as Present_Position reads do."""
        return self.raw_to_cal(name, self.cal_to_raw(name, cal))

    def steps_to_rad(self, name: str, steps: float) -> float:
        """Encoder steps (servo speed/accel units) -> joint radians."""
        if name == "gripper":
            # gripper steps are motor steps too; convert through the 0-100 span
            cal = float(steps) / self.g_counts_span * 100.0
            return cal * abs(self._g_rad_per_unit)
        return math.radians(float(steps) * 360.0 / _RES)
