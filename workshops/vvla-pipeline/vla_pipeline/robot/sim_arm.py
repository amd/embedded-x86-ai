# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``DirectArm`` on the MuJoCo digital twin instead of the Feetech bus.

:class:`SimArmClient` is that transport with a :class:`so101_sim.SimFollower`
in place of ``SO101Follower``. The slew clamp, the per-move servo profile
writes, the motion-state reads and the effort probe are
:class:`~vla_pipeline.robot.arm_interface.DirectArm`'s.

Selected with ``robot.backend: sim`` (see ``make_arm`` and the ROS server
node). The world is per process (``so101_sim.shared_world``) so the arm and
an in-process sim arm camera see the same physics. The workshop keeps its
own copy (``workshop/common/motion.py``); ``tests/sim/test_copies.py`` fails when the two drift apart.
"""

from __future__ import annotations

import logging

from vla_pipeline.robot.arm_interface import (
    GRIPPER_CLOSED_FLOOR,
    REST_POSE,
    DirectArm,
    MotionTimeout,
    settled,
)

logger = logging.getLogger(__name__)


def sim_world(cfg: dict):
    """The process-wide MuJoCo digital twin, set up for the pipeline's arm."""
    from so101_sim import shared_world

    sim = dict(cfg.get("sim") or {})
    gm = dict(sim.get("gripper_map") or {})
    gm["closed_cal"] = GRIPPER_CLOSED_FLOOR
    sim["gripper_map"] = gm
    return shared_world({"sim": sim}, rest_pose=REST_POSE)


class SimArmClient(DirectArm):
    """:class:`DirectArm` driving the MuJoCo digital twin: a
    :class:`so101_sim.SimFollower` in place of ``SO101Follower``. The slew
    clamp, the per-move servo profile writes, the motion-state reads and the
    effort probe are :class:`DirectArm`'s."""

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
        logger.info("SO101 digital twin connected (MuJoCo, %s)", self.world.scene_path.name)
        self._init_transport(max_relative_target)

    def _plan_start(self) -> dict:
        """The twin's profile starts from its setpoint, which a squeezed jaw leaves past the object."""
        return self.world.setpoint_cal()

    def _await_motion(self, joints: list, sent_at: float, stop_check) -> tuple[str, dict]:
        """The twin's physics owns the servo profiles: check them after each physics pass.

        The twin computes how long its profiles still run (:meth:`SimWorld.profile_s`,
        braking included after a goal written mid-move). That sim time, plus the
        physics step the end shows in, bounds the wait.
        """
        w = self.world
        budget = w.profile_s(joints) + w.dt
        deadline = float(w.data.time) + budget
        seen = w.passes
        while True:
            now = float(w.data.time)  # before the state: still moving after `now` is a real overrun
            state = {"joints": self.get_joints(), **self.read_motion_state(), "eta_s": self.last_eta_s}
            if stop_check is not None and stop_check():
                return "stopped", state
            if settled(state, joints):
                return "settled", state
            if now > deadline:
                raise MotionTimeout(f"servos {joints} still moving {budget:.3f} s of sim time after the goal: {state}")
            seen = w.next_step(seen)

    def state_time(self) -> float:
        """The twin's sim time."""
        return float(self.world.data.time)

    def wait_state(self, t: float, stop_check=None) -> float | None:
        """The first physics pass at or after sim time ``t``."""
        w = self.world
        seen = w.passes
        while float(w.data.time) < t:
            if stop_check is not None and stop_check():
                return None
            seen = w.next_step(seen)
        if stop_check is not None and stop_check():
            return None
        return float(w.data.time)

    def grip_report(self) -> str:
        return self.world.grip_report()

