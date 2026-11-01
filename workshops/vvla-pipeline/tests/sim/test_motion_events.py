# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""What a move waits on, and how fast the servos run it.

A direct-goal move is one ``Goal_Position`` write. ``common.servo_profile``
picks each joint's ``Goal_Velocity`` and ``Acceleration`` from the arm's
gravity and inertia, and the move is over when the servo bus owner reports
the goal finished (``ArmClient._await_motion``). The twin runs the same
registers, so its ``Moving`` flag must drop when the planner says.
In-process, the twin's physics thread is that owner.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.sim

STALL_NM = 1.91
DAMPING, FRICTION, INERTIA = 0.6, 0.052, 0.05
STEP_RAD = math.tau / 4096  # one encoder step of a body joint


def _profile(gravity_nm: float) -> tuple[int, int]:
    from common.servo_profile import profile_for

    return profile_for(
        gravity_nm, INERTIA, DAMPING, FRICTION, rad_per_step=STEP_RAD, stall_nm=STALL_NM,
        cruise_frac=0.7, peak_frac=0.95, max_speed_raw=3073.0, accel_unit_steps=100.0,
    )


def test_planned_torque_stays_inside_the_cruise_and_ramp_budgets():
    accel_step_rad = 100.0 * STEP_RAD
    for gravity in (0.0, 0.3, 0.8, 1.1):
        speed_raw, accel_raw = _profile(gravity)
        cruise = gravity + FRICTION + DAMPING * speed_raw * STEP_RAD
        ramp = cruise + INERTIA * accel_raw * accel_step_rad
        # Rounding to whole register steps may add up to one step of each.
        assert cruise <= 0.7 * STALL_NM + DAMPING * STEP_RAD, (gravity, speed_raw)
        assert ramp <= 0.95 * STALL_NM + INERTIA * accel_step_rad, (gravity, accel_raw)


def test_every_joint_converts_servo_steps_through_the_twin_joint_map(world):
    from common.servo_profile import shared_profiler
    from so101_sim import MOTOR_NAMES

    profiler = shared_profiler()
    assert profiler.rad_per_step == {m: world.jmap.steps_to_rad(m, 1.0) for m in MOTOR_NAMES}


def test_more_gravity_never_plans_a_faster_profile():
    profiles = [_profile(g) for g in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)]
    assert all(a[0] >= b[0] and a[1] >= b[1] for a, b in zip(profiles, profiles[1:])), profiles


def test_a_joint_whose_budget_gravity_uses_up_still_creeps():
    from common.servo_profile import MIN_ACCEL_RAW, MIN_SPEED_RAW

    speed_raw, accel_raw = _profile(1.5)
    assert speed_raw == MIN_SPEED_RAW
    assert accel_raw >= MIN_ACCEL_RAW


@pytest.mark.parametrize("distance, speed, accel, seconds", [
    (0.0, 1.0, 4.0, 0.0),
    (2.0, 1.0, 4.0, 2.25),  # cruises: 2 rad at 1 rad/s plus one ramp's worth
    (0.1, 1.0, 4.0, 2.0 * (0.1 / 4.0) ** 0.5),  # never reaches cruise: a triangle
])
def test_trapezoid_duration(distance, speed, accel, seconds):
    from common.servo_profile import trapezoid_s

    assert trapezoid_s(distance, speed, accel) == pytest.approx(seconds)


@pytest.fixture(scope="module")
def world():
    from so101_sim.world import SimWorld

    w = SimWorld(start=False)
    yield w
    w.close()


def _run_planned(world, start: dict, goal: dict, lockstep: bool = False):
    """Write the planner's registers and ``goal`` on the twin's bus, then step
    physics until no servo in ``goal`` reports ``Moving``.

    Returns the plan, each joint's seconds until its flag dropped, and the
    joints that tripped overload protection on the way.
    """
    from common.servo_profile import shared_profiler
    from so101_sim import MOTOR_NAMES
    from so101_sim.arm import SimBus

    world.reset(start)
    for obj in world.objects:
        world.hide(obj)
    bus = SimBus(world)
    plan = shared_profiler().plan(start, goal, lockstep=lockstep)
    bus.sync_write("Goal_Velocity", plan.speed_raw, normalize=False)
    bus.sync_write("Acceleration", plan.accel_raw, normalize=False)
    bus.sync_write("Goal_Position", goal)
    finished: dict[str, float] = {}
    elapsed = 0.0
    while len(finished) < len(goal):
        assert elapsed <= 2.0 * plan.eta_s + 0.5, f"servos still moving {elapsed:.2f} s into a {plan.eta_s:.2f} s plan"
        for joint, moving in bus.sync_read("Moving", list(goal)).items():
            if not moving:
                finished.setdefault(joint, elapsed)
        world.advance(world.dt)
        elapsed += world.dt
    tripped = [m for m, t in zip(MOTOR_NAMES, world.protection.tripped) if t]
    return plan, finished, tripped


def test_moving_is_set_by_the_goal_write_before_physics_steps(world):
    """A state published between the write and the next physics step must not look finished."""
    from common.motion import REST_POSE
    from so101_sim.arm import SimBus

    world.reset(REST_POSE)
    bus = SimBus(world)
    assert not any(bus.sync_read("Moving").values())
    bus.write("Goal_Position", "shoulder_pan", 20.0)
    moving = bus.sync_read("Moving")
    assert moving["shoulder_pan"] == 1
    assert not any(v for k, v in moving.items() if k != "shoulder_pan"), moving


def _keyframe_moves():
    """Park, the pickup's scan sweep, a reach down to the table, the wave and every dance keyframe."""
    from common.config import load_config
    from common.motion import DANCE_MOVES, REST_POSE, WAVE_UP, clamp_joints

    pp = load_config()["pick_place"]
    wave = clamp_joints({**REST_POSE, **WAVE_UP})
    scan = clamp_joints({**REST_POSE, **pp["scan_pose"]})
    moves = [("rest to wave", REST_POSE, wave), ("park from wave", wave, dict(REST_POSE)), ("rest to scan", REST_POSE, scan)]
    pose = scan
    for pan in pp["scan_pans"][1:]:
        goal = {**pose, "shoulder_pan": float(pan)}
        moves.append((f"scan pan {pan:+.0f}", pose, goal))
        pose = goal
    over = clamp_joints({**scan, "shoulder_lift": -15.0, "elbow_flex": 20.0, "wrist_flex": 54.0})
    low = clamp_joints({**over, "shoulder_lift": 9.0})
    moves += [("reach over", scan, over), ("down to the table", over, low), ("park from the table", low, dict(REST_POSE))]
    for name, keyframes in DANCE_MOVES.items():
        pose = dict(REST_POSE)
        for i, keyframe in enumerate(keyframes):
            goal = clamp_joints({**pose, **keyframe})
            moves.append((f"{name} {i}", pose, goal))
            pose = goal
    return moves


@pytest.mark.parametrize("name, start, goal", _keyframe_moves(), ids=[m[0] for m in _keyframe_moves()])
def test_planned_moves_finish_when_predicted_without_an_overload_trip(world, name, start, goal):
    plan, finished, tripped = _run_planned(world, start, goal)
    assert not tripped, f"{name}: overload trip on {tripped}"
    took = max(finished.values())
    # The twin's generator brakes on sqrt(2 a e) one physics step at a time; the
    # planner's closed-form trapezoid is the same profile without the steps.
    assert took == pytest.approx(plan.eta_s, abs=0.05 * plan.eta_s + 5 * world.dt), (name, finished)
    assert took <= plan.eta_s + world.dt, (name, finished)  # SimArmClient's move budget relies on this


def test_lockstep_finishes_every_joint_with_the_slowest(world):
    from common.motion import REST_POSE, WAVE_UP, clamp_joints

    goal = clamp_joints({**REST_POSE, **WAVE_UP})
    # A joint moving a few degrees runs at the registers' smallest steps and ends early.
    moved = [k for k in goal if abs(goal[k] - REST_POSE[k]) >= 10.0]
    plan, finished, tripped = _run_planned(world, REST_POSE, goal, lockstep=True)
    assert not tripped
    times = [finished[k] for k in moved]
    # Lockstep scales whole register steps, so each joint can end a few percent early.
    assert max(times) - min(times) <= 0.1 * plan.eta_s + 5 * world.dt, dict(zip(moved, times))


def test_a_partial_goal_plans_only_its_joints():
    from common.motion import REST_POSE
    from common.servo_profile import shared_profiler

    plan = shared_profiler().plan(REST_POSE, {"shoulder_pan": 30.0})
    assert set(plan.speed_raw) == set(plan.accel_raw) == {"shoulder_pan"}
    assert plan.eta_s > 0.0


def test_a_twin_goal_is_planned_from_the_servo_setpoint(world):
    """The twin's profile starts at its setpoint. Another caller may have moved the
    arm since this client's last read, so the setpoint is read for every goal."""
    from common.motion import REST_POSE, SimArmClient
    from common.servo_profile import shared_profiler

    arm = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
    world.reset({**REST_POSE, "gripper": 85.0})
    arm.get_joints()
    world.reset({**REST_POSE, "gripper": 40.0})
    setpoint = world.setpoint_cal()
    arm.send_joints({"gripper": 85.0})
    assert setpoint["gripper"] == pytest.approx(40.0, abs=0.1)
    assert arm.last_eta_s == pytest.approx(shared_profiler().plan(setpoint, {"gripper": 85.0}).eta_s)


def test_a_goal_written_mid_move_is_bounded_by_the_twin_profile(world):
    """A goal written while a profile still runs brakes first. The twin's own
    dry run counts that braking, so it matches the steps the profile takes."""
    from common.motion import REST_POSE, SimArmClient

    arm = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
    world.reset(dict(REST_POSE))
    arm.send_joints({"shoulder_pan": 60.0})
    world.advance(0.3)
    arm.send_joints({"shoulder_pan": -20.0})
    budget = world.profile_s(["shoulder_pan"])
    assert budget > arm.last_eta_s, "the reverse must cost more than a profile from rest"
    took = 0.0
    while arm.read_motion_state()["moving"]["shoulder_pan"]:
        world.advance(world.dt)
        took += world.dt
    assert took == pytest.approx(budget, abs=world.dt)


def test_the_twin_arm_ends_a_move_on_the_physics_pass_its_servos_finish():
    """Past the planned profile plus one physics step the wait raises MotionTimeout."""
    from common.motion import REST_POSE, WAVE_UP, SimArmClient, clamp_joints
    from so101_sim.world import SimWorld

    world = SimWorld(rest_pose=REST_POSE)
    try:
        arm = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
        assert arm.move_to(clamp_joints({**REST_POSE, **WAVE_UP})) == "arrived"
    finally:
        world.close()


def test_a_twin_move_fails_at_once_without_a_physics_thread(world):
    from common.motion import REST_POSE, SimArmClient

    arm = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
    world.reset(dict(REST_POSE))
    with pytest.raises(RuntimeError, match="physics thread is not running"):
        arm.move_to({"shoulder_pan": 20.0})


def _reporting_arm(finish: dict, how: str = "settled"):
    """A transport whose bus owner reports every goal finished at ``finish``."""
    from common.motion import ArmClient

    class ReportingArm(ArmClient):
        def __init__(self):
            self.sent: list[dict] = []

        def send_joints(self, joints):
            self.sent.append(dict(joints))

        def get_joints(self):
            return dict(finish)

        def _await_motion(self, joints, sent_at, stop_check):
            return how, {"joints": dict(finish), "moving": {}, "overload": {"elbow_flex": True}, "eta_s": 0.4}

    return ReportingArm()


def test_move_to_reports_where_the_servos_finished():
    goal = {"shoulder_pan": 20.0, "elbow_flex": 30.0}
    assert _reporting_arm({"shoulder_pan": 19.0, "elbow_flex": 30.0}).move_to(goal) == "arrived"
    assert _reporting_arm({"shoulder_pan": 20.0, "elbow_flex": 12.0}).move_to(goal) == "blocked"
    # The jaws' open-fraction readback need not converge: a gripper move ends with its servo.
    assert _reporting_arm({"gripper": 60.0}).move_to({"gripper": 95.0}) == "arrived"


def test_a_stopped_move_halts_where_the_arm_is():
    arm = _reporting_arm({"shoulder_pan": 8.0, "elbow_flex": 30.0}, how="stopped")
    assert arm.move_to({"shoulder_pan": 20.0}) == "stopped"
    assert arm.sent == [{"shoulder_pan": 20.0}, {"shoulder_pan": 8.0, "elbow_flex": 30.0}]


def test_dry_run_arm_moves_at_once():
    from common.motion import DryRunArm

    arm = DryRunArm()
    assert arm.move_to({"shoulder_pan": 30.0}) == "arrived"
    assert arm.get_joints()["shoulder_pan"] == 30.0
