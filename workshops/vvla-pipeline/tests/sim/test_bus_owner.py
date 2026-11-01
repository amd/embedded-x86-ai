# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Direct serial (``robot.use_ros2: false``): the bus owner in this process.

The arm under test is ``LeRobotArm``, the real-arm class, with the twin's
``SimFollower`` as its bus: the owner writes goals and reads the servos'
registers exactly as it would on the Feetech bus.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


@pytest.fixture
def world(workshop_cfg):
    from common.motion import REST_POSE
    from so101_sim.world import SimWorld

    w = SimWorld({"sim": dict(workshop_cfg.get("sim") or {})}, rest_pose=REST_POSE)
    for obj in w.objects:
        w.hide(obj)
    yield w
    w.close()


def _real_class_arm(world):
    """A ``LeRobotArm`` (direct goals, as the configs set) whose follower is the twin's bus."""
    from common.motion import LeRobotArm
    from so101_sim import SimFollower

    arm = LeRobotArm.__new__(LeRobotArm)
    arm._robot = SimFollower(world, max_relative_target=None, robot_id="sim")
    arm._robot.config.disable_torque_on_disconnect = False
    arm._robot.connect()
    arm._init_transport(None)
    return arm


def _until(world, pred) -> None:
    seen = world.next_step(world.passes)
    while not pred():
        seen = world.next_step(seen)


@pytest.fixture
def owner(world, workshop_cfg):
    from common.motion import BusOwnerArm, bus_owner

    arm = bus_owner(_real_class_arm(world), workshop_cfg)
    assert isinstance(arm, BusOwnerArm)
    yield arm
    arm.close()


def test_only_the_real_bus_arm_gets_an_owner(world, workshop_cfg):
    from common.motion import DryRunArm, SimArmClient, bus_owner

    twin = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
    assert bus_owner(twin, workshop_cfg) is twin, "the twin's physics owns its servos"
    dry = DryRunArm()
    assert bus_owner(dry, workshop_cfg) is dry


def test_a_direct_move_arrives_on_the_owners_report(owner, world):
    assert owner.move_to({"shoulder_pan": 25.0, "elbow_flex": 70.0}) == "arrived"
    here = world.joint_cal()
    assert abs(here["shoulder_pan"] - 25.0) < 2.5 and abs(here["elbow_flex"] - 70.0) < 2.5, here
    assert not any(owner.read_motion_state()["moving"].values()), "a servo still runs after the move"
    assert owner.move_to({"gripper": 80.0}) == "arrived"


def test_a_stop_halts_a_direct_move(owner, world):
    from rigs import wait_joint_rest

    start = owner.get_joints()["shoulder_pan"]
    how = owner.move_to({"shoulder_pan": 60.0}, stop_check=lambda: owner.get_joints()["shoulder_pan"] > start + 10.0)
    assert how == "stopped"
    wait_joint_rest(world, "shoulder_pan")
    assert world.joint_cal()["shoulder_pan"] < 45.0, "the arm carried on after the stop"


def test_the_torque_limit_and_current_pass_through_the_owner(owner, world):
    owner.set_gripper_torque_limit(240)
    _until(world, lambda: world.bus.read("Torque_Limit", "gripper", normalize=False) == 240)
    t = owner.wait_state(owner.state_time() + 0.2)
    assert t is not None and owner.get_gripper_load() is not None


def test_the_owner_ticks_and_parks_on_close(world, workshop_cfg):
    from common.motion import REST_POSE, bus_owner

    arm = bus_owner(_real_class_arm(world), workshop_cfg)
    t0 = arm.state_time()
    assert arm.wait_state(t0 + 0.05) >= t0 + 0.05
    assert arm.wait_state(t0 + 10.0, stop_check=lambda: True) is None
    assert arm.move_to({"shoulder_pan": 30.0}) == "arrived"
    arm.close()
    _until(world, lambda: not any(world.bus.sync_read("Moving").values()))
    assert abs(world.joint_cal()["shoulder_pan"] - REST_POSE["shoulder_pan"]) < 2.5, "close did not park"
