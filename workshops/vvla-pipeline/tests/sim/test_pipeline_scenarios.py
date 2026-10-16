# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier 1: ``vla_pipeline`` behaviors driven by scripted user signals.

Spoken commands go through the real ``Dispatcher`` (grace window, stop fast
path, noise gate) and the real ``LlamaIntentParser`` HTTP client against the
mock llama-server; behaviors run unchanged on ``SimArmClient`` (the
``LeRobotArm`` transport over the twin). Assertions are on the simulated
physics: joint traces, gripper current, and whether objects are held.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rigs import HOLD_WATCH_S, assert_ordered_high_scan, carry_problem, hold_problem
from scripted import Hand, Person

pytestmark = pytest.mark.sim

BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Present_Current counts held for 0.3 s: a squeeze at hold_torque_limit holds
# ~94; the free jaw closing holds 50-70 and only spikes higher.
_SQUEEZE_CURRENT = 80.0


def _rest():
    from vla_pipeline.robot.arm_interface import REST_POSE

    return REST_POSE


def _assert_at(rig, pose: dict, tol: float = 4.0, joints=BODY):
    cur = rig.joints()
    off = {k: round(cur[k] - pose[k], 1) for k in joints if abs(cur[k] - pose[k]) > tol}
    assert not off, f"not at pose (deg off): {off}"


def _assert_within_limits(rig, limits: dict, slack: float = 2.0):
    for s in rig.trace.samples:
        for k, (lo, hi) in limits.items():
            v = s.joints[k]
            assert lo - slack <= v <= hi + slack, f"{k}={v:.1f} outside [{lo}, {hi}] at t={s.t:.2f}s"


# ---------------------------------------------------------------------------
# scripted behaviors
# ---------------------------------------------------------------------------
def test_wave(pipeline_rig, mock_llm):
    rig = pipeline_rig
    rig.run([(0.3, "wave at me")], timeout=25)

    # the LLM client sent a grammar-constrained, deterministic request
    p = mock_llm.last_payload
    assert p and p.get("grammar") and p.get("temperature") == 0.0
    assert rig.event_times("behavior_start", intent="wave")
    assert rig.event_times("behavior_end", intent="wave", ok=True)

    _, elbow = rig.trace.series("elbow_flex")
    assert elbow.min() < -35.0, "the forearm was never raised"
    _, pan = rig.trace.series("shoulder_pan")
    assert pan.max() > 20.0 and pan.min() < -20.0, "the hand never swept through +/-25"
    _assert_at(rig, _rest())


def test_release_opens_the_jaws(pipeline_rig):
    rig = pipeline_rig
    rig.ctx.shared_state["grip_lock"] = True
    rig.ctx.shared_state["grip_lock_pos"] = 40.0
    rig.run([(0.3, "drop it"), (2.5, "open"), (5.0, "release")], timeout=20)

    assert len(rig.event_times("behavior_end", intent="release", ok=True)) == 3
    assert rig.said_any("Released")
    assert rig.joints()["gripper"] > 80.0
    assert not rig.ctx.shared_state.get("grip_lock")
    _assert_at(rig, _rest())


def test_home_from_anywhere(pipeline_rig):
    rig = pipeline_rig
    rig.world.reset({"shoulder_pan": 40.0, "shoulder_lift": -40.0, "elbow_flex": 50.0, "wrist_flex": 20.0,
                     "wrist_roll": 0.0})
    rig.run([(0.3, "go home")], timeout=15)
    assert rig.event_times("behavior_end", intent="home", ok=True)
    _assert_at(rig, _rest())


def test_dance_visits_moves_and_parks(pipeline_rig):
    from vla_pipeline.robot.arm_interface import JOINT_LIMITS

    rig = pipeline_rig
    rig.run([(0.3, "let's dance")], timeout=30)
    assert rig.event_times("behavior_end", intent="dance", ok=True)
    spans = {k: np.ptp(rig.trace.series(k)[1]) for k in BODY}
    assert sum(v > 20.0 for v in spans.values()) >= 2, f"dance barely moved: {spans}"
    _assert_within_limits(rig, JOINT_LIMITS)
    _assert_at(rig, _rest())


# ---------------------------------------------------------------------------
# dispatcher safety properties
# ---------------------------------------------------------------------------
def test_stop_mid_wave(pipeline_rig):
    """'stop' halts the move within a control tick; the wave never resumes, the arm parks."""
    rig = pipeline_rig
    rig.run([(0.3, "wave"), (1.6, "stop")], timeout=25)
    t_stop = rig.transcripts.sent_at("stop")
    assert rig.event_times("stop"), "stop fast path not taken"
    assert len(rig.event_times("parsed")) == 1, "stop must not reach the LLM"

    # the commanded goal leaves the wave target within 100 ms (halt = hold here)
    t, goal = rig.trace.series("elbow_flex", goal=True)
    before = goal[t < t_stop][-1]
    after = goal[(t > t_stop) & (t < t_stop + 0.15)]
    assert np.any(np.abs(after - before) > 1.0), "goal did not change within 150 ms of 'stop'"

    # no further wave strokes after the stop (shoulder_pan goal never hits +/-25)
    tr, pan_goal = rig.trace.series("shoulder_pan", t0=t_stop, goal=True)
    stroke = np.abs(np.abs(pan_goal) - 25.0) <= 2.0
    assert not np.any(stroke & (tr > t_stop + 0.15)), (
        f"wave resumed after stop: +/-25 pan goal at t-t_stop={np.round(tr[stroke] - t_stop, 3)}")
    assert rig.event_times("behavior_end", intent="wave", stopped=True)
    _assert_at(rig, _rest())


def test_noise_and_grace_window_cause_no_motion(pipeline_rig, mock_llm):
    rig = pipeline_rig
    start = rig.joints()
    rig.run([(0.2, "wave"), (1.3, "thank you."), (1.6, "you"), (1.9, "...")], timeout=10,
            startup_grace_s=1.0)
    reasons = [d["reason"] for _t, e, d in rig.events if e == "ignored"]
    assert reasons == ["grace", "noise", "noise", "noise"], reasons
    assert mock_llm.request_count == 0, "gated transcripts must never reach the LLM"
    _assert_at(rig, start, tol=1.0, joints=BODY + ("gripper",))


# ---------------------------------------------------------------------------
# grip: gripper current from contact physics
# ---------------------------------------------------------------------------
def _present():
    from vla_pipeline.behaviors.grip import PRESENT_POSE

    return PRESENT_POSE


def test_grip_object_handed_over(pipeline_rig):
    rig = pipeline_rig
    # the "user" puts the toothbrush between the jaws after the arm presents
    rig.when_near(_present(), lambda: rig.stage("toothbrush", _present()), delay=1.0)
    rig.run([(0.3, "hold this for me")], timeout=25, settle=1.5)

    assert rig.ctx.shared_state.get("grip_lock"), f"grip not reported; robot said {rig.said}"
    assert rig.world.is_grasped("toothbrush"), (
        f"toothbrush not between both jaws: {rig.world.contact_report('toothbrush')}"
    )
    assert rig.trace.grasped_seconds("toothbrush") >= 1.0, "toothbrush not held for 1 s"
    held = rig.trace.held_current()
    assert held > _SQUEEZE_CURRENT, f"squeezing the toothbrush held only {held:.0f} counts"
    assert rig.said_any("Object gripped")


def test_grip_empty_jaws_reports_nothing(pipeline_rig):
    rig = pipeline_rig
    rig.run([(0.3, "hold this")], timeout=25)
    held = rig.trace.held_current()
    assert held < _SQUEEZE_CURRENT, f"the jaws held {held:.0f} counts: something was squeezed"
    assert not rig.ctx.shared_state.get("grip_lock"), "reported a grip with empty jaws"


# ---------------------------------------------------------------------------
# pick and place / fetch (YOLO on the arm camera + scripted person)
# ---------------------------------------------------------------------------
PERSON_LEFT = Person(cx=0.3, wrist=(0.4, 0.45))


def _descend_pose(cfg, pan: float = 0.0) -> dict:
    pp = cfg["behaviors"]["pick_place"]
    return {"shoulder_pan": pan, **pp["approach_pose"], "shoulder_lift": float(pp["descend_lift"]),
            "wrist_roll": 0.0}


def _assert_others_in_place(rig, handled: str):
    moved = rig.disturbed(handled)
    assert not moved, f"the arm moved objects it was not asked to pick up (m): {moved}"


def _holding(rig) -> bool:
    from vla_pipeline.behaviors.grip import holding

    return holding(rig.ctx.arm, rig.cfg["behaviors"]["grip"], rig.ctx.shared_state["grip_lock_pos"])


def _assert_held_out(rig, body: str, spoken: str, z0: float):
    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    assert rig.said_any(f"Got the {spoken}"), f"robot said {rig.said}"
    heights = rig.trace.heights(body)
    assert heights.max() - z0 > 0.03, f"{spoken} never rose 3 cm"
    problems = [p for p in (carry_problem(rig, body, z0), hold_problem(rig, body, z0)) if p]
    assert not problems, "; ".join(problems)
    assert rig.world.is_grasped(body), f"{spoken} was dropped"
    assert _holding(rig), f"holding() says the jaws are empty while the twin has the {spoken}"
    end = rig.world.body_pos(body)
    assert abs(end[0] - 0.30) < 0.08 and abs(end[1]) < 0.08, (
        f"{spoken} ended at ({end[0]:.3f}, {end[1]:.3f}), not held out in front"
    )
    assert end[2] > 0.08, f"{spoken} was set back down ({end[2]:.3f} m)"
    _assert_others_in_place(rig, body)


@pytest.mark.slow
@pytest.mark.parametrize("spot", [
    None,  # its default spot among the other props, in reach of the fixed poses too
    (0.22, -0.04),  # closer than the fixed descend pose reaches
    (0.32, 0.0),  # farther than the fixed descend pose reaches
    (0.25, -0.10),
], ids=["default", "near", "far", "right"])
def test_pick_up_the_ball(pipeline_rig, caplog, spot):
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    if spot is None:
        rig.default_scene()  # the notebook console's scene: all four props
        z0 = rig.world.body_pos("sports_ball")[2]
    else:
        z0 = rig.put("sports_ball", [*spot, 0.015])[2]
    rig.run([(0.3, "pick up the ball")], timeout=60, settle=HOLD_WATCH_S)

    assert "IK reach: sports ball" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert_ordered_high_scan(
        caplog, rig.cfg["behaviors"]["pick_place"]["scan_pans"], allow_near=spot is not None,
    )
    if spot is None:
        assert caplog.text.count("IK locate:") == 1
        assert caplog.text.count("IK solve") == 1
    _assert_held_out(rig, "sports_ball", "sports ball", z0)
    assert rig.event_times("behavior_end", intent="pick_place", ok=True)


@pytest.mark.slow
def test_pick_up_toothbrush_aligns_wrist(pipeline_rig, caplog):
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()  # the notebook console's scene: all four props
    z0 = rig.world.body_pos("toothbrush")[2]
    rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=HOLD_WATCH_S)

    assert_ordered_high_scan(caplog, rig.cfg["behaviors"]["pick_place"]["scan_pans"])
    assert caplog.text.count("IK locate:") == 1
    assert caplog.text.count("-> solved") == 1
    _, roll = rig.trace.series("wrist_roll")
    assert abs(roll).max() > 60.0, "wrist never turned across the toothbrush"
    _assert_held_out(rig, "toothbrush", "toothbrush", z0)
    assert rig.event_times("behavior_end", intent="pick_place", ok=True)


@pytest.mark.slow
def test_pick_up_bottle_without_knocking_it_over(pipeline_rig, caplog):
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()
    z0 = rig.world.body_pos("bottle")[2]
    rig.run([(0.3, "pick up the bottle")], timeout=90, settle=HOLD_WATCH_S)

    assert "IK reach: bottle" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert_ordered_high_scan(caplog, rig.cfg["behaviors"]["pick_place"]["scan_pans"])
    assert caplog.text.count("IK locate:") == 1
    assert caplog.text.count("IK solve") == 1
    _assert_held_out(rig, "bottle", "bottle", z0)
    assert rig.event_times("behavior_end", intent="pick_place", ok=True)


@pytest.mark.slow
def test_bottle_that_slips_out_mid_carry_is_reported_dropped(pipeline_rig):
    # the bottle, because the fixed fingertips cradle the ball and hold it without friction
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()
    z0 = rig.world.body_pos("bottle")[2]
    rig.when_lifted("bottle", z0, lambda: rig.slicken("bottle"))
    rig.run([(0.3, "pick up the bottle")], timeout=90, settle=2.0)

    assert rig.event_times("slicken"), "the bottle was never lifted"
    assert hold_problem(rig, "bottle", z0, watch_s=0.0) is not None, "the slip check missed the drop"
    assert not rig.world.is_grasped("bottle")
    assert rig.said_any("I dropped the bottle"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.event_times("behavior_end", intent="pick_place", ok=False)


@pytest.mark.slow
def test_pick_ball_out_of_fixed_reach_says_so(pipeline_rig):
    rig = pipeline_rig
    rig.cfg["behaviors"]["pick_place"]["reach"] = "fixed"
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    # in view of the arm camera but past the taught descend pose: anywhere short
    # of it the missed grab pushes the ball under the parked jaws
    z0 = rig.put("sports_ball", [0.34, 0.0, 0.015])[2]
    rig.run([(0.3, "pick up the ball")], timeout=60, settle=2.0)

    assert rig.said_any("couldn't get a grip on the sports ball"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.trace.heights("sports_ball").max() - z0 < 0.03, "ball was lifted"
    assert rig.event_times("behavior_end", intent="pick_place", ok=False)
    _assert_at(rig, _rest())


def test_pick_unknown_object_fails_safely(pipeline_rig):
    rig = pipeline_rig
    rig.stage("toothbrush", _descend_pose(rig.cfg))
    rig.run([(0.3, "pick up the pen")], timeout=25)
    assert rig.said_any("don't know what a pen looks like"), rig.said
    assert not any(s.grasped.get("toothbrush") for s in rig.trace.samples)
    assert not rig.disturbed(), f"the arm moved objects (m): {rig.disturbed()}"
    _assert_at(rig, _rest())


@pytest.mark.slow
def test_fetch_block_hands_to_person(pipeline_rig):
    rig = pipeline_rig
    fb = rig.cfg["behaviors"]["fetch_block"]
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    # Staged with the arm already there: approaching from rest sweeps a
    # pedestal-staged prop off. The ball provides a stable grasp for this
    # legacy open-loop handoff; detection-driven object tests run separately.
    rig.arm.go_to({**fb["block_pose"], "gripper": 70.0}, duration_s=2.5)
    time.sleep(0.8)
    rig.stage("sports_ball", fb["block_pose"])
    z0 = rig.world.body_pos("sports_ball")[2]
    # lift_pose is lower than this pedestal: take it away once the jaws have the ball
    rig.when(lambda: rig.world.is_grasped("sports_ball"), lambda: rig.world.park_stand("stand_a"))
    rig.run([(0.3, "bring me the block")], timeout=60)

    assert rig.event_times("behavior_end", intent="fetch_block", ok=True), rig.said
    assert rig.trace.heights("sports_ball").max() - z0 > 0.03, "prop never lifted"
    opened = next(s.t for s in rig.trace.samples if s.t > 1.0 and s.goal["gripper"] > 90.0)
    problem = hold_problem(rig, "sports_ball", z0, watch_s=0.0, until=opened - 1e-3)
    assert problem is None, problem
    assert not rig.world.is_grasped("sports_ball"), "prop not released"
    _assert_at(rig, _rest())


@pytest.mark.slow
def test_fetch_block_without_block_says_so(pipeline_rig):
    """The ball sits on the table, not at block_pose: the jaws close on air."""
    rig = pipeline_rig
    rig.run([(0.3, "bring me the block")], timeout=40)

    assert rig.said_any("couldn't get a grip"), rig.said
    assert not any(s.grasped.get("sports_ball") for s in rig.trace.samples)
    _assert_at(rig, _rest())


# ---------------------------------------------------------------------------
# gesture mimic (scripted person + hand -> mapper -> twin)
# ---------------------------------------------------------------------------
def _mimic_limits():
    from vla_pipeline.robot.arm_interface import JOINT_LIMITS

    return JOINT_LIMITS


@pytest.mark.slow
def test_mimic_sweep_pinch_roll(pipeline_rig):
    rig = pipeline_rig
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    floor = float(GRIPPER_CLOSED_FLOOR)
    rig.pose.keys = [
        (0.0, [Person(wrist=(0.5, 0.45))]),
        (2.0, [Person(wrist=(0.5, 0.45))]),
        (5.0, [Person(wrist=(0.8, 0.45))]),  # sweep to the right of the frame
        (6.0, [Person(wrist=(0.8, 0.45))]),
    ]
    rig.hands.keys = [
        (0.0, Hand(pinch=1.3, roll=0.0)),
        (6.0, Hand(pinch=1.3, roll=0.0)),
        (6.5, Hand(pinch=0.05, roll=0.0)),  # pinch -> gripper to the floor
        (8.0, Hand(pinch=0.05, roll=0.0)),
        (9.0, Hand(pinch=0.05, roll=60.0)),  # roll one way...
        (10.5, Hand(pinch=0.05, roll=60.0)),
        (11.5, Hand(pinch=0.05, roll=-60.0)),  # ...then the other
        (13.0, Hand(pinch=0.05, roll=-60.0)),
    ]
    rig.run([(0.3, "copy my movements"), (13.0, "stop")], timeout=30)

    tr = rig.trace
    _, pan_a = tr.series("shoulder_pan", 1.5, 2.0, goal=True)
    t, pan_s = tr.series("shoulder_pan", 2.0, 5.5, goal=True)
    _, pan_b = tr.series("shoulder_pan", 5.5, 6.0, goal=True)
    assert abs(pan_b.mean() - pan_a.mean()) > 25.0, "pan did not follow the wrist sweep"
    corr = np.corrcoef(t, pan_s)[0, 1]
    assert abs(corr) > 0.9, f"pan not steady through the sweep (r={corr:.2f})"

    _, grip = tr.series("gripper", 7.5, 8.0)
    assert grip.mean() < floor + 4.0, f"pinch did not close the gripper (got {grip.mean():.1f})"

    _, roll_p = tr.series("wrist_roll", 10.0, 10.5, goal=True)
    _, roll_n = tr.series("wrist_roll", 12.5, 13.0, goal=True)
    assert np.sign(roll_p.mean()) != np.sign(roll_n.mean()) and min(abs(roll_p.mean()), abs(roll_n.mean())) > 15
    _assert_within_limits(rig, _mimic_limits())
    _assert_at(rig, _rest())


def test_mimic_hands_a_held_grip_to_the_pinch(pipeline_rig):
    """A grip locked by a pickup does not survive "copy my movements": the
    pinch drives the gripper, so opening the fingers opens the jaws."""
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    rig = pipeline_rig
    rig.ctx.shared_state["grip_lock"] = True
    rig.ctx.shared_state["grip_lock_pos"] = 45.0
    rig.pose.keys = [(0.0, [Person(wrist=(0.5, 0.45))]), (10.0, [Person(wrist=(0.5, 0.45))])]
    rig.hands.keys = [
        (0.0, Hand(pinch=1.3)), (3.0, Hand(pinch=1.3)),
        (3.5, Hand(pinch=0.05)), (6.5, Hand(pinch=0.05)),
        (7.0, Hand(pinch=1.3)), (10.0, Hand(pinch=1.3)),
    ]
    rig.run([(0.3, "copy my movements"), (10.0, "stop")], timeout=30)
    assert rig.said_any("Your pinch drives the gripper"), rig.said
    assert not rig.ctx.shared_state.get("grip_lock"), "mimic left the grip locked"
    _, opened = rig.trace.series("gripper", 2.0, 3.0)
    _, pinched = rig.trace.series("gripper", 5.5, 6.5)
    _, reopened = rig.trace.series("gripper", 9.0, 10.0)
    floor = float(GRIPPER_CLOSED_FLOOR)
    assert pinched.mean() < floor + 4.0, f"the pinch did not close the gripper ({pinched.mean():.1f})"
    assert min(opened.mean(), reopened.mean()) > floor + 30.0, (
        f"open fingers did not open the gripper ({opened.mean():.1f}, {reopened.mean():.1f})"
    )


def test_mimic_keeps_a_held_ball_until_the_pinch_is_seen(pipeline_rig):
    """Mimic does not command the gripper before MediaPipe reports a pinch: the
    pose model sees the wrist first, the pinched hand a second later, and the
    ball the pickup holds stays held."""
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()
    rig.run([(0.3, "pick up the ball")], timeout=60, settle=HOLD_WATCH_S)
    assert rig.world.is_grasped("sports_ball"), f"the pickup failed: {rig.said}"

    pinched = Hand(pinch=0.05)
    rig.pose.keys = [(0.0, [Person(wrist=(0.5, 0.45))]), (10.0, [Person(wrist=(0.5, 0.45))])]
    rig.hands.keys = [(0.0, None), (1.0, None), (1.01, pinched), (10.0, pinched)]
    first = len(rig.trace.samples)  # the trace runs on across runs; the clock restarts
    rig.run([(0.3, "copy my movements"), (6.0, "stop")], timeout=60)

    assert rig.said_any("Your pinch drives the gripper"), rig.said
    goals = [s.goal["gripper"] for s in rig.trace.samples[first:]]
    assert max(goals) <= goals[0] + 0.5, (
        f"mimic opened the gripper to {max(goals):.1f} (held at {goals[0]:.1f})"
    )
    assert rig.world.is_grasped("sports_ball"), "mimic dropped the held ball"


@pytest.mark.slow
def test_mimic_hand_lost_then_person_lost(pipeline_rig):
    rig = pipeline_rig
    rig.pose.keys = [
        (0.0, [Person(wrist=(0.5, 0.45))]),
        (2.0, [Person(wrist=(0.5, 0.45))]),
        (4.5, [Person(wrist=(0.25, 0.45))]),  # wrist moves with NO hand seen
        (5.5, [Person(wrist=(0.25, 0.45))]),
        (5.6, []),  # person leaves: the arm must hold its pose
        (8.0, []),
    ]
    rig.hands.keys = [(0.0, Hand(pinch=1.3)), (1.5, Hand(pinch=1.3)), (1.6, None)]
    rig.run([(0.3, "mirror my arm"), (8.0, "stop")], timeout=20)

    tr = rig.trace
    _, a = tr.series("shoulder_pan", 1.6, 2.0, goal=True)
    _, b = tr.series("shoulder_pan", 5.0, 5.5, goal=True)
    assert abs(b.mean() - a.mean()) > 20.0, "YOLO wrist fallback did not drive pan"
    t, held = tr.series("shoulder_pan", 6.2, 7.9, goal=True)
    assert np.ptp(held) < 1.5, f"arm drifted with nobody in frame ({np.ptp(held):.1f} deg)"


def test_mimic_follows_nearest_person(pipeline_rig):
    rig = pipeline_rig
    far_right = Person(cx=0.75, scale=0.5, wrist=(0.85, 0.45))
    near_left = Person(cx=0.35, scale=1.1, wrist=(0.2, 0.45))
    rig.pose.keys = [(0.0, [far_right, near_left])]
    rig.hands.keys = [(0.0, None)]
    rig.run([(0.3, "follow me"), (3.5, "stop")], timeout=15)
    _, pan = rig.trace.series("shoulder_pan", 2.5, 3.4, goal=True)
    # mirrored mapping: a wrist on the LEFT of the image pans the arm positive
    assert pan.mean() > 20.0, f"followed the wrong person (pan {pan.mean():.1f})"
