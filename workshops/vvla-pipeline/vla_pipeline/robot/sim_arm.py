# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``LeRobotArm`` on the MuJoCo digital twin instead of the Feetech bus.

:class:`SimArmClient` is that transport with a :class:`so101_sim.SimFollower`
in place of ``SO101Follower``. The slew clamp, direct-goal profile writes and
effort probe are the ones on :class:`common.motion.LeRobotArm`.

Selected with ``robot.backend: sim`` (see ``make_arm`` and the ROS server
node). The world is per process (``so101_sim.shared_world``) so the arm and
an in-process sim arm camera see the same physics.
"""

from __future__ import annotations

import logging

from vla_pipeline.robot.arm_interface import (
    DIRECT_MODE_ACCEL_RAW,
    DIRECT_MODE_SPEED_RAW,
    GRIPPER_TORQUE_LIMIT,
    REST_POSE,
    LeRobotArm,
    sim_world,
)

logger = logging.getLogger(__name__)

__all__ = ["REST_POSE", "SimArmClient", "sim_world"]


class SimArmClient(LeRobotArm):
    """:class:`LeRobotArm` driving the MuJoCo digital twin."""

    def __init__(self, cfg: dict, world=None):
        from so101_sim import SimFollower

        r = cfg.get("robot", {})
        max_relative_target = r.get("max_relative_target", 8.0)
        self.world = world or sim_world(cfg)
        self._robot = SimFollower(
            self.world,
            max_relative_target=max_relative_target,
            robot_id=r.get("robot_id", "sim"),
        )
        self._robot.config.disable_torque_on_disconnect = False
        self._robot.connect()
        self._effort_register = False
        self._coordinated_joints = []
        logger.info("SO101 digital twin connected (MuJoCo, %s)", self.world.scene_path.name)
        self._direct_goal = max_relative_target is None
        if self._direct_goal:
            self.set_motion_profile(
                speed=DIRECT_MODE_SPEED_RAW, acceleration=DIRECT_MODE_ACCEL_RAW,
            )
        self.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)

    def grip_report(self) -> str:
        return self.world.grip_report()
