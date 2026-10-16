# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier 1: the workshop's ``common.manipulation`` behaviors on the twin.

Spoken commands go through the project's ``integration.handle_transcript``
and the workshop ``LlamaIntentParser`` against the mock llama-server; the
behaviors run on ``common.motion.SimArmClient`` and see through the
workshop's YOLOv26s on the twin's rendered arm camera (``--sim-eyes seg``:
ground-truth boxes from its segmentation).
"""

from __future__ import annotations

import logging

import pytest

from rigs import HOLD_WATCH_S, assert_ordered_high_scan, carry_problem, hold_problem
from scripted import Hand, Person

pytestmark = [pytest.mark.sim, pytest.mark.slow]

PERSON_LEFT = Person(cx=0.3, wrist=(0.4, 0.45))


def _pick_ball(rig, pos=None, timeout: float = 60.0):
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    if pos is None:
        rig.default_scene()  # the notebook console's scene: all four props
        z0 = rig.world.body_pos("sports_ball")[2]
    else:
        z0 = rig.put("sports_ball", pos)[2]
    rig.run([(0.3, "pick up the ball")], timeout=timeout, settle=HOLD_WATCH_S)
    return z0


def _assert_others_in_place(rig, handled: str):
    moved = rig.disturbed(handled)
    assert not moved, f"the arm moved objects it was not asked to pick up (m): {moved}"


def _holding(rig) -> bool:
    from common.manipulation import holding

    return holding(rig.ctx.arm, rig.cfg["grip"], rig.ctx.shared_state["grip_lock_pos"])


def _assert_held_out(rig, body: str, spoken: str, z0: float):
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


def test_workshop_release_opens_the_jaws(workshop_rig):
    from common.motion import REST_POSE

    rig = workshop_rig
    rig.ctx.shared_state["grip_lock"] = True
    rig.ctx.shared_state["grip_lock_pos"] = 40.0
    rig.run([(0.3, "drop it")], timeout=20)

    from common.fixtures import offline_parse_text

    for phrase in ("release", "open", "drop it", "let go"):
        assert offline_parse_text(phrase)["command"] == "release", phrase
    assert offline_parse_text("pick up the ball")["command"] == "pick_place"
    parsed = [data for _, event, data in rig.events if event == "parsed" and data.get("intent") == "release"]
    assert len(parsed) == 1, parsed
    assert rig.said_any("Released")
    assert rig.joints()["gripper"] > 80.0
    assert not rig.ctx.shared_state.get("grip_lock")
    cur = rig.joints()
    off = {
        k: round(cur[k] - REST_POSE[k], 1)
        for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        if abs(cur[k] - REST_POSE[k]) > 4.0
    }
    assert not off, off


def test_workshop_mimic_hands_a_held_grip_to_the_pinch(workshop_rig):
    """A grip locked by a pickup does not survive "mirror my arm": the pinch
    drives the gripper, so opening the fingers opens the jaws."""
    from common.motion import GRIPPER_CLOSED_FLOOR

    rig = workshop_rig
    rig.ctx.shared_state["grip_lock"] = True
    rig.ctx.shared_state["grip_lock_pos"] = 45.0
    rig.pose.keys = [(0.0, [Person(wrist=(0.5, 0.45))]), (10.0, [Person(wrist=(0.5, 0.45))])]
    rig.hands.keys = [
        (0.0, Hand(pinch=1.3)), (3.0, Hand(pinch=1.3)),
        (3.5, Hand(pinch=0.05)), (6.5, Hand(pinch=0.05)),
        (7.0, Hand(pinch=1.3)), (10.0, Hand(pinch=1.3)),
    ]
    rig.run([(0.3, "mirror my arm"), (10.0, "stop")], timeout=30)
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


def test_workshop_mimic_keeps_a_held_ball_until_the_pinch_is_seen(workshop_rig):
    """Mimic does not command the gripper before MediaPipe reports a pinch: the
    pose model sees the wrist first, the pinched hand a second later, and the
    ball the pickup holds stays held."""
    rig = workshop_rig
    _pick_ball(rig)
    assert rig.world.is_grasped("sports_ball"), f"the pickup failed: {rig.said}"

    pinched = Hand(pinch=0.05)
    rig.pose.keys = [(0.0, [Person(wrist=(0.5, 0.45))]), (10.0, [Person(wrist=(0.5, 0.45))])]
    rig.hands.keys = [(0.0, None), (1.0, None), (1.01, pinched), (10.0, pinched)]
    first = len(rig.trace.samples)  # the trace runs on across runs; the clock restarts
    rig.run([(0.3, "mirror my arm"), (6.0, "stop")], timeout=60)

    assert rig.said_any("Your pinch drives the gripper"), rig.said
    goals = [s.goal["gripper"] for s in rig.trace.samples[first:]]
    assert max(goals) <= goals[0] + 0.5, (
        f"mimic opened the gripper to {max(goals):.1f} (held at {goals[0]:.1f})"
    )
    assert rig.world.is_grasped("sports_ball"), "mimic dropped the held ball"


@pytest.mark.parametrize("spot", [None, (0.22, -0.04)], ids=["default", "near"])
def test_workshop_pick_up_the_ball(workshop_rig, caplog, spot):
    rig = workshop_rig
    z0 = _pick_ball(rig, None if spot is None else [*spot, 0.015])

    assert "IK reach: sports ball" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert_ordered_high_scan(
        caplog, rig.cfg["pick_place"]["scan_pans"], allow_near=spot is not None,
    )
    if spot is None:
        assert caplog.text.count("IK locate:") == 1
        assert caplog.text.count("IK solve") == 1
    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    _assert_held_out(rig, "sports_ball", "sports ball", z0)


def test_workshop_bottle_that_slips_out_mid_carry_is_reported_dropped(workshop_rig):
    # the bottle, because the jaws cradle the ball and hold it without friction
    rig = workshop_rig
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


def test_workshop_pick_up_toothbrush_aligns_wrist(workshop_rig, caplog):
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()  # the notebook console's scene: all four props
    z0 = rig.world.body_pos("toothbrush")[2]
    rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=HOLD_WATCH_S)

    assert_ordered_high_scan(caplog, rig.cfg["pick_place"]["scan_pans"])
    assert caplog.text.count("IK locate:") == 1
    assert caplog.text.count("-> solved") == 1
    _, roll = rig.trace.series("wrist_roll")
    assert abs(roll).max() > 60.0, "wrist never turned across the toothbrush"
    _assert_held_out(rig, "toothbrush", "toothbrush", z0)


def test_workshop_pick_up_the_bottle(workshop_rig, caplog):
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()
    z0 = rig.world.body_pos("bottle")[2]
    rig.run([(0.3, "pick up the bottle")], timeout=90, settle=HOLD_WATCH_S)

    assert "IK reach: bottle" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert_ordered_high_scan(caplog, rig.cfg["pick_place"]["scan_pans"])
    assert caplog.text.count("IK locate:") == 1
    assert caplog.text.count("IK solve") == 1
    _assert_held_out(rig, "bottle", "bottle", z0)


def test_workshop_pick_ball_out_of_fixed_reach_says_so(workshop_rig):
    rig = workshop_rig
    rig.cfg["pick_place"]["reach"] = "fixed"
    z0 = _pick_ball(rig, [0.19, 0.0, 0.015])

    assert rig.said_any("couldn't get a grip on the sports ball"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.trace.heights("sports_ball").max() - z0 < 0.03, "ball was lifted"


_TRACE_PREFIXES = (
    "reset:",
    "scan sighting: toothbrush",
    "IK reach: toothbrush",
    "grasp:",
    "grasp missed:",
)


def _postmortem(caplog, said) -> str:
    """The stack lines that show what reset restored and how each grasp landed."""
    lines = [
        rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(_TRACE_PREFIXES)
    ]
    speech = " | ".join(said) if said else "(nothing)"
    body = "\n".join(lines) if lines else "(no stack trace)"
    return f"robot said: {speech}\n{body}"


def _said_since(rig, start: int, phrase: str) -> bool:
    return any(phrase.lower() in message.lower() for message in rig.said[start:])


def _press_reset(rig) -> None:
    """The notebook reset button, once the behavior has already stopped."""
    from common.motion import restore_twin

    rig.ctx.shared_state.pop("grip_lock", None)
    rig.ctx.shared_state.pop("grip_lock_pos", None)
    restore_twin(rig.arm)


def test_workshop_reset_then_pick_the_toothbrush(workshop_rig, caplog):
    """The GUI sequence, three times: pick up the toothbrush, drop it, reset, pick it up.

    The lesson lays the brush straight with ``place()``. Each reset restores
    the compiled pose. The failure text is the stack trace of every attempt.
    """
    caplog.set_level(logging.INFO, logger="common.manipulation")
    caplog.set_level(logging.INFO, logger="so101_sim")
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()

    rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=HOLD_WATCH_S)
    assert rig.said_any("Got the toothbrush"), _postmortem(caplog, rig.said)
    assert rig.world.is_grasped("toothbrush"), _postmortem(caplog, rig.said)

    for n in (1, 2, 3):
        spoken = len(rig.said)
        rig.run([(0.3, "drop it")], timeout=20)
        assert _said_since(rig, spoken, "Released"), _postmortem(caplog, rig.said)
        _press_reset(rig)
        spoken = len(rig.said)
        rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=HOLD_WATCH_S)
        report = f"after reset {n}\n{_postmortem(caplog, rig.said)}"
        assert _said_since(rig, spoken, "Got the toothbrush"), report
        assert rig.world.is_grasped("toothbrush"), report
