# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The twin's 7.4 V STS3215: current readings, torque limit, overload protection."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.sim

# Jaws at table height, closing sideways: how the pickup grasps the ball.
GRASP = {"shoulder_pan": 0.0, "shoulder_lift": 9.0, "elbow_flex": 20.0, "wrist_flex": 54.0, "wrist_roll": 0.0}
GRIPPER = 5


@pytest.mark.parametrize("torque_nm,moving,counts", [
    (0.0, False, 1),  # 6 mA idle
    (0.0, True, 23),  # 150 mA running free
    (0.49, False, 100),  # 650 mA at the rated 5 kg·cm
    (1.91, False, 385),  # 2.5 A at the 19.5 kg·cm stall
    (0.955, False, 193),  # stalled at Torque_Limit 500
    (0.382, False, 78),  # stalled at Torque_Limit 200
])
def test_current_follows_the_datasheet(torque_nm, moving, counts):
    from so101_sim.config import DEFAULTS
    from so101_sim.servo import CurrentModel

    cm = CurrentModel({**DEFAULTS["current_model"], "noise_units": 0})
    cm.update(np.full(6, torque_nm), dt=1.0, moving=np.full(6, moving))
    assert int(cm.present_current()[GRIPPER]) == counts


@pytest.fixture(scope="module")
def world():
    from so101_sim.world import SimWorld

    w = SimWorld(start=False)
    yield w
    w.close()


def _gripper_cap(world) -> float:
    return float(world.model.actuator_forcerange[world._act[GRIPPER], 1])


def test_torque_limit_is_live_and_max_torque_limit_waits_for_power_on(world):
    from so101_sim.arm import SimBus

    world.reset()
    bus = SimBus(world)
    full = float(world.force_full[GRIPPER, 1])
    assert full == pytest.approx(1.91)
    assert _gripper_cap(world) == pytest.approx(full)
    bus.write("Max_Torque_Limit", "gripper", 500)
    assert bus.read("Max_Torque_Limit", "gripper") == 500
    assert bus.read("Torque_Limit", "gripper") == 1000
    assert _gripper_cap(world) == pytest.approx(full), "an EEPROM write changed the live limit"
    bus.write("Torque_Limit", "gripper", 240)
    assert bus.read("Torque_Limit", "gripper") == 240
    assert _gripper_cap(world) == pytest.approx(0.24 * full)


def _squeeze_ball(world, limit: int):
    from common.motion import GRIPPER_CLOSED_FLOOR
    from so101_sim.arm import SimBus

    world.reset({**GRASP, "gripper": 85.0})
    for obj in world.objects:
        world.hide(obj)
    world.stage_object("sports_ball", GRASP)
    world.advance(0.3)
    bus = SimBus(world)
    bus.write("Overload_Torque", "gripper", 25)  # LeRobot's gripper configure
    bus.write("Torque_Limit", "gripper", limit)
    bus.write("Goal_Position", "gripper", GRIPPER_CLOSED_FLOOR)
    return bus


def test_overload_drops_the_grip_after_protection_time_until_the_next_goal(world):
    from common.motion import GRIPPER_CLOSED_FLOOR

    bus = _squeeze_ball(world, 500)
    full = float(world.force_full[GRIPPER, 1])
    world.advance(1.0)  # the jaws reach the ball
    assert world.is_grasped("sports_ball")
    assert bus.read("Present_Load", "gripper", normalize=False) >= 450, "the jaws are not loaded"
    for _ in range(10):  # the pipeline streams goals; that must not reset the timer
        world.advance(0.1)
        bus.write("Goal_Position", "gripper", GRIPPER_CLOSED_FLOOR)
    assert bus.read("Status", "gripper") == 0
    assert _gripper_cap(world) == pytest.approx(0.5 * full)
    world.advance(1.0)
    assert bus.read("Status", "gripper") & 0x20, "no overload after 2 s above 25 %"
    assert _gripper_cap(world) == pytest.approx(0.2 * full)
    world.advance(0.3)
    assert bus.read("Present_Load", "gripper", normalize=False) <= 205
    bus.write("Goal_Position", "gripper", GRIPPER_CLOSED_FLOOR)
    assert bus.read("Status", "gripper") == 0
    assert _gripper_cap(world) == pytest.approx(0.5 * full)


def test_reset_reloads_the_gripper_torque_limit_and_clears_a_trip(world):
    """A grasp's hold limit and a tripped servo do not survive a reset."""
    from so101_sim.arm import SimBus

    bus = SimBus(world)
    bus.write("Max_Torque_Limit", "gripper", 500)
    bus.write("Torque_Limit", "gripper", 240)
    bus.write("Goal_Position", "gripper", 40.0)
    world.protection.tripped[GRIPPER] = True
    world.protection._over_s[GRIPPER] = 5.0
    world._apply_torque_caps()
    world.reset()
    assert bus.read("Torque_Limit", "gripper", normalize=False) == 500
    assert world.protection.limit[GRIPPER] == pytest.approx(0.5)
    assert not world.protection.tripped[GRIPPER]
    assert bus.read("Status", "gripper", normalize=False) == 0
    assert _gripper_cap(world) == pytest.approx(0.5 * float(world.force_full[GRIPPER, 1]))


def test_every_dance_keyframe_holds_without_an_overload_trip(world):
    """No dance keyframe drives the arm into itself or the table. A keyframe
    that does stalls its servos until the overload protection cuts their
    torque, and the arm then crawls to its rest pose after the dance."""
    from common.motion import DANCE_MOVES, REST_POSE, clamp_joints
    from so101_sim import MOTOR_NAMES
    from so101_sim.arm import SimBus

    for move, keyframes in DANCE_MOVES.items():
        world.reset(REST_POSE)
        bus = SimBus(world)
        bus.sync_write("Goal_Velocity", 1000, normalize=False)  # the arm's direct-goal profile
        bus.sync_write("Acceleration", 50, normalize=False)
        pose = dict(REST_POSE)
        for keyframe in keyframes:
            pose = clamp_joints({**pose, **keyframe})
            for motor in keyframe:
                bus.write("Goal_Position", motor, pose[motor])
            world.advance(2.5)  # longer than Protection_Time
            tripped = [m for m, t in zip(MOTOR_NAMES, world.protection.tripped) if t]
            assert not tripped, f"{move} {keyframe}: overload trip on {tripped}"


def test_a_hold_under_the_overload_threshold_never_trips(world):
    bus = _squeeze_ball(world, 240)
    world.advance(5.0)
    assert world.is_grasped("sports_ball")
    assert bus.read("Status", "gripper") == 0
    assert 200 <= bus.read("Present_Load", "gripper", normalize=False) < 250
