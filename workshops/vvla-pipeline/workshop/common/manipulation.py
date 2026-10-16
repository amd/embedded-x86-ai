# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Manipulation behaviors: grip, pick_place, fetch_block (PROVIDED).

The pipeline's remaining voice commands, in the workshop's handler shape
``(ctx, cmd)`` so the registry can dispatch them like dance or wave:

- **grip** - present the open jaws, close until contact (gripper current, or
  a position stall), confirm with the arm camera, lock the grip. "release",
  "open", or "drop it" opens the jaws; gesture_mimic hands them to the pinch.
- **pick_place** - ground the spoken object name to a COCO class, walk the
  scan until one settled frame holds a usable box, locate that frame once on
  the table and reach it with IK
  (``so101_sim.kinematics``; the taught poses only with ``reach: fixed`` or
  the arm camera offline), grasp, lift it above the other objects, and hold
  it out. The jaws stay
  closed until "release", "open", or "drop it".
- **fetch_block** - fetch the block from its taught pose, then hand it over.

The handover finds you with YOLO-pose on the operator camera, aims the pan
joint at you, extends, opens the jaws and parks. The same sequence the
pipeline runs (``vla_pipeline/behaviors/``); the workshop is standalone, so
it lives here rather than being imported. Settings: ``grip``, ``pick_place``
and ``fetch_block`` in ``config/workshop.yaml``.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from so101_sim.kinematics import OPEN_GRIPPER

from common.imaging import KP
from common.intents import Intent
from common.motion import GRASP_LIMITS, GRIPPER_CLOSED_FLOOR, GRIPPER_TORQUE_LIMIT, JOINT_LIMITS, REST_POSE

logger = logging.getLogger(__name__)

# Pose presenting the gripper toward the user, jaws open.
PRESENT_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.0,
    "elbow_flex": 30.0,
    "wrist_flex": 10.0,
    "wrist_roll": 0.0,
    "gripper": 80.0,
}

TICK_S = 0.05

# ----------------------------------------------------------------------------
# Grasping: contact from current or position stall, confirmed by the camera
# ----------------------------------------------------------------------------


def _jaw_roi(frame: np.ndarray, roi) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def object_in_jaws(reference: np.ndarray, frame: np.ndarray, roi, threshold: float) -> bool:
    """True if the jaw region differs enough from the empty-jaws reference."""
    a = cv2.cvtColor(_jaw_roi(reference, roi), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(_jaw_roi(frame, roi), cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape or a.size == 0:
        return False
    diff = float(np.mean(cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0))))
    logger.debug("[grip] jaw ROI diff %.3f (threshold %.3f)", diff / 255.0, threshold)
    return diff / 255.0 >= threshold


def close_with_feedback(arm, g: dict, stop_check=None, start_from: float | None = None, ease: bool = True):
    """Close the gripper until contact. Returns ``(gripped, hold_position)``.

    Contact is sustained current above ``load_threshold`` (when the arm reports
    it) or a stall: the jaws stopped short of the command. After the
    command reaches the closed floor, sampling continues for ``stall_grace_s``
    so a thin object (a pen) is still caught.

    The close runs at :data:`GRIPPER_TORQUE_LIMIT`. After contact the goal
    trails ``hold_past_contact`` below the jaw for ``reseat_s`` while both
    fingers seat, then :func:`ease_grip` drops the limit for the hold. The
    goal stays past the object, so the torque limit sets the grip force.

    ``ease=False`` keeps the close limit for an object that still rests on
    the table: there the hold force lets it settle against one jaw and the
    table, and the lift leaves it behind. The caller eases once the first
    lift step is done.
    """
    step = float(g["close_step_deg"])
    eps = float(g["stall_eps_deg"])
    need_ticks = int(g["stall_ticks"])
    load_threshold = float(g.get("load_threshold", 120.0))
    load_ticks_needed = int(g.get("load_ticks", 3))
    grace_s = float(g.get("stall_grace_s", 0.6))
    floor = GRIPPER_CLOSED_FLOOR  # the arm clamps below this; a lower command would read as a stall

    arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    use_load = arm.get_gripper_load() is not None
    logger.info("[grip] closing with %s feedback", "current" if use_load else "position-stall")

    commanded = (
        start_from if start_from is not None else float(arm.get_joints().get("gripper", PRESENT_POSE["gripper"]))
    )
    stall_count = load_count = 0
    gripped = False
    grace_deadline = None
    prev = float(arm.get_joints().get("gripper", commanded))
    while True:
        if stop_check and stop_check():
            break
        if commanded > floor:
            commanded = max(commanded - step, floor)
            arm.send_joints({"gripper": commanded})
            if commanded == floor:
                grace_deadline = time.time() + grace_s
        elif grace_deadline is not None and time.time() >= grace_deadline:
            break  # fully closed and the grace window is over: nothing between the jaws
        time.sleep(TICK_S)

        if use_load:
            load = arm.get_gripper_load()
            if load is None:  # current readback died mid-grasp: fall back to the stall
                use_load = False
            elif load >= load_threshold:
                load_count += 1
                if load_count >= load_ticks_needed:
                    gripped = True
                    break
            else:
                load_count = 0

        measured = float(arm.get_joints().get("gripper", commanded))
        # A servo trails a moving goal, so lagging alone isn't contact: the jaws must also have stopped.
        stopped_moving = prev - measured < eps
        prev = measured
        if measured - commanded > eps and stopped_moving:
            stall_count += 1
            if stall_count >= need_ticks:
                gripped = True
                break
        else:
            stall_count = 0

    if not gripped:
        return False, floor
    hold = _reseat(arm, g, stop_check)
    if ease:
        ease_grip(arm, g)
    return True, hold


def ease_grip(arm, g: dict) -> None:
    """Drop to ``hold_torque_limit``: below the 25 % ``Overload_Torque``
    LeRobot sets for the gripper, which would otherwise cut the grip to 20 %
    about 2 s into a squeeze at the close limit."""
    arm.set_gripper_torque_limit(int(g.get("hold_torque_limit", 240)))


def _reseat(arm, g: dict, stop_check=None) -> float:
    """Keep the goal ``hold_past_contact`` below the jaw for ``reseat_s``,
    then wait for the jaw to settle against the last goal (:func:`_settle_jaw`).
    Contact can register while one jaw still pushes the object; the trailing
    goal lets both fingers seat. Returns the last goal."""
    past = float(g.get("hold_past_contact", 6.0))
    deadline = time.time() + float(g.get("reseat_s", 0.6))
    while True:
        jaw = float(arm.get_joints()["gripper"])
        hold = max(jaw - past, GRIPPER_CLOSED_FLOOR)
        arm.send_joints({"gripper": hold})
        if time.time() >= deadline or (stop_check and stop_check()):
            break
        time.sleep(TICK_S)
    _settle_jaw(arm, g, hold, stop_check)
    return hold


def _settle_jaw(arm, g: dict, hold: float, stop_check=None, timeout_s: float = 1.0) -> None:
    """Wait until the jaw reaches ``hold`` or stops short of it.

    The last trailing goal is set while an empty jaw may still be closing.
    Read at that moment, the jaw sits ``hold_past_contact`` above its goal
    exactly as if an object held it open, so :func:`holding` would see a
    grip in empty jaws.
    """
    eps = float(g["stall_eps_deg"])
    still_needed = int(g["stall_ticks"])
    target = max(hold, GRIPPER_CLOSED_FLOOR)
    deadline = time.time() + timeout_s
    prev = float(arm.get_joints()["gripper"])
    still = 0
    while time.time() < deadline and not (stop_check and stop_check()):
        time.sleep(TICK_S)
        jaw = float(arm.get_joints()["gripper"])
        if jaw - target <= eps:  # closed onto the goal: nothing between the jaws
            return
        still = still + 1 if abs(prev - jaw) < eps else 0
        if still >= still_needed:  # stopped short of the goal: something is
            return
        prev = jaw


def holding(arm, g: dict, hold: float) -> bool:
    """The jaws stopped more than ``stall_eps_deg`` short of the held goal.
    The goal is past the object, so an empty jaw closes onto it. Only
    meaningful once the jaw has settled against ``hold``:
    :func:`close_with_feedback` returns after :func:`_settle_jaw`. Current
    is not used: at the hold limit it reads below the close's contact level."""
    target = max(hold, GRIPPER_CLOSED_FLOOR)
    return float(arm.get_joints().get("gripper", target)) - target > float(g["stall_eps_deg"])


def run_grip(ctx, cmd=None, wait_s: float = 4.0) -> bool:
    """Take an object from the user and lock the grip; "grip" again releases."""
    g = ctx.cfg["grip"]
    state = ctx.shared_state
    roi = g.get("jaw_roi", [0.30, 0.45, 0.70, 0.95])
    diff_thr = float(g.get("jaw_diff_threshold", 0.06))
    cam = ctx.arm_camera

    if state.get("grip_lock"):
        ctx.say("I'm already holding something. Say release to let go.")
        return False

    ctx.say("Place the object between the jaws.")
    ctx.arm.go_to(PRESENT_POSE, duration_s=2.5, stop_check=ctx.stop_check)

    reference = None  # the empty jaws, before the user reaches in
    if cam is not None:
        ok, ref = cam.read()
        reference = ref if ok else None

    for remaining in range(int(wait_s), 0, -1):
        if ctx.stop_check():
            ctx.arm.go_to_rest()
            return False
        logger.info("[grip] closing in %d...", remaining)
        time.sleep(1.0)

    if reference is not None:
        ok, now = cam.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            ctx.say("I don't see anything between the jaws yet - closing carefully.")

    gripped, hold = close_with_feedback(ctx.arm, g, stop_check=ctx.stop_check)

    if gripped and reference is not None:
        ok, now = cam.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            logger.warning("[grip] contact detected but the arm camera sees empty jaws")
            gripped = False

    if gripped:
        state["grip_lock"] = True
        state["grip_lock_pos"] = hold
        ctx.say(
            f"Object gripped at {hold:.1f} degrees - grip locked. "
            "Say 'gesture mimic' to use it; say 'release' to let go."
        )
    else:
        ctx.say("No object detected.")
        ctx.arm.go_to({"gripper": GRIPPER_CLOSED_FLOOR}, duration_s=0.8)
    return gripped

# ----------------------------------------------------------------------------
# Handover: find the person, extend, release, park
# ----------------------------------------------------------------------------


def person_pan_target(pose, frame: np.ndarray) -> float | None:
    """The pan angle facing the nearest person's torso, or None."""
    person = pose.primary_person(frame)
    if person is None:
        return None
    kpts, _ = person
    xs = [kpts[i, 0] for i in (KP["l_shoulder"], KP["r_shoulder"], KP["nose"]) if kpts[i, 2] > pose.kpt_threshold]
    if not xs:
        return None
    cx = (float(np.mean(xs)) / frame.shape[1]) * 2.0 - 1.0  # [-1, 1]
    return float(np.clip(-cx * 60.0, -80.0, 80.0))  # mirrored: person on the left -> pan left


def find_person_pan(ctx, seek_s: float = 4.0) -> tuple[float, bool]:
    """Track the person for ``seek_s`` and aim the pan joint at them."""
    pan, found = 0.0, False
    if ctx.pose is None or ctx.camera is None:
        logger.info("[handoff] no operator camera - handing over straight ahead")
        return pan, found
    deadline = time.time() + seek_s
    while time.time() < deadline and not ctx.stop_check():
        ok, frame = ctx.camera.read()
        if not ok:
            time.sleep(TICK_S)
            continue
        target = person_pan_target(ctx.pose, frame)
        if target is not None:
            pan, found = target, True
            ctx.arm.send_joints({"shoulder_pan": pan})
        time.sleep(TICK_S)
    logger.info("[handoff] person %s (pan=%.1f)", "found" if found else "NOT found", pan)
    return pan, found


def hand_over(ctx, section: dict, lift_pose: dict, still_holding=lambda: True) -> bool:
    """Find the person, extend toward them, open the jaws, park.

    Returns False, with the jaws left closed, when ``still_holding()`` finds
    them empty after the extend: there is nothing to hand over.
    """
    pan, _ = find_person_pan(ctx)
    logger.info("[handoff] extend toward the person")
    ctx.arm.go_to(
        {
            "shoulder_pan": pan,
            "shoulder_lift": float(lift_pose["shoulder_lift"]) + float(section["handoff_reach"]),
            "elbow_flex": max(float(lift_pose["elbow_flex"]) - 40.0, 0.0),
            "wrist_flex": 20.0,
        },
        duration_s=2.5,
        stop_check=ctx.stop_check,
    )
    if not still_holding():
        return False
    logger.info("[handoff] release - take it!")
    time.sleep(min(float(section["release_timeout_s"]), 6.0) * 0.5)
    ctx.arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    ctx.arm.go_to({"gripper": JOINT_LIMITS["gripper"][1]}, duration_s=1.0)
    time.sleep(1.0)
    park_open(ctx.arm)
    return True


def park_open(arm, stop_check=None) -> None:
    """Go to the rest pose with the jaws wide open. Closing them on the way
    home can catch the object just let go of and drag it across the table."""
    arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    arm.go_to({**REST_POSE, "gripper": JOINT_LIMITS["gripper"][1]}, duration_s=3.0, stop_check=stop_check)

# ----------------------------------------------------------------------------
# fetch_block and pick_place
# ----------------------------------------------------------------------------


# The lift runs this long at the close limit, so the block hangs from the
# fingers before the grip eases to the hold limit.
_BREAKAWAY_S = 0.3


def run_fetch_block(ctx, cmd=None) -> bool:
    """Grasp the block at its taught pose, lift it, hand it over.

    The pose is open loop - nothing looks for the block - so the grasp is
    checked twice, after closing and after the lift; if the block isn't held,
    the arm says so and parks instead of handing over empty jaws.
    """
    fb, g = ctx.cfg["fetch_block"], ctx.cfg["grip"]
    block_pose, lift_pose = dict(fb["block_pose"]), dict(fb["lift_pose"])

    logger.info("[fetch] approach the block (jaws open)")
    ctx.arm.go_to({**block_pose, "gripper": 70.0}, duration_s=2.5, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    logger.info("[fetch] grasp")
    gripped, hold = close_with_feedback(ctx.arm, g, stop_check=ctx.stop_check, start_from=70.0, ease=False)
    if ctx.stop_check():
        return False
    if not (gripped and holding(ctx.arm, g, hold)):
        ctx.say("I couldn't get a grip on the block - is it in its usual spot?")
        ctx.arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
        ctx.arm.go_to({"gripper": 70.0}, duration_s=0.8)
        ctx.arm.go_to_rest()
        return False
    logger.info("[fetch] lift")
    ctx.arm.send_joints({**lift_pose, "gripper": hold})
    time.sleep(_BREAKAWAY_S)
    ease_grip(ctx.arm, g)
    ctx.arm.go_to({**lift_pose, "gripper": hold}, duration_s=2.5, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    time.sleep(0.3)
    if not holding(ctx.arm, g, hold):
        ctx.say("I dropped the block.")
        park_open(ctx.arm)
        return False
    if not hand_over(ctx, fb, lift_pose, lambda: holding(ctx.arm, g, hold)):
        if ctx.stop_check():
            return False
        ctx.say("I dropped the block.")
        park_open(ctx.arm)
        return False
    return True

# Grasp point high enough that a carried ball (its centre is the grasp
# point) clears the bottle, the tallest prop on the table.
_CARRY_Z = 0.12
# Out toward the operator, who stands in front of the arm.
_OFFER_XY = (0.30, 0.0)
_OFFER_ERR_M = 0.015


def _log_grip(arm, label: str, hold: float, plan=None) -> None:
    """Log the jaw against its held goal, the current, how far the hand landed
    from ``plan``'s grasp pose, and what the fingers touch (the twin only)."""
    here = arm.get_joints()
    jaw = float(here.get("gripper", hold))
    load = arm.get_gripper_load()
    parts = [
        f"jaw {jaw:.1f}, goal {hold:.1f} (margin {jaw - hold:+.1f})",
        "current n/a" if load is None else f"current {load:.0f}",
    ]
    if plan is not None:
        try:
            from so101_sim.kinematics import shared_kinematics

            kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
            want = {**here, **plan.grasp}
            got = {**here, "gripper": want["gripper"]}
            got_pt = kin.world.grasp_point(got, 0.04)
            want_pt = kin.world.grasp_point(want, 0.04)
            parts.append(
                f"fingers {1000 * got_pt[2]:.0f} mm high, planned {1000 * want_pt[2]:.0f} mm, "
                f"object centre {1000 * plan.xyz[2]:.0f} mm"
            )
        except Exception as e:
            logger.debug("grasp offset unavailable: %s", e)
    report = arm.grip_report()
    if report:
        parts.append(report)
    logger.info("%s: %s", label, "; ".join(parts))


def _offer(arm, hold, stop_check, limits, say, resolved, still_holding, ease=lambda: None) -> str:
    """Lift above the other props and hold the object out. The jaws stay closed.

    Two joint-space legs from the pose that is holding the object, keeping
    its wrist roll. ``ease()`` runs after the first step, once the object
    hangs from the fingers. Returns ``offered``, ``dropped``, ``stopped``, or
    ``unreachable``.
    """
    try:
        from so101_sim.kinematics import joint_line, shared_kinematics
    except Exception as e:
        logger.warning("offer unavailable: %s", e)
        return "unreachable"

    def stopped() -> bool:
        return bool(stop_check and stop_check())

    pending = [ease]

    def follow(steps, duration_s: float) -> bool:
        for step in steps:
            arm.go_to(
                step,
                duration_s=duration_s / len(steps),
                stop_check=stop_check,
                limits=limits,
            )
            while pending:
                pending.pop()()
            if stopped():
                return False
        return True

    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    here = arm.get_joints()
    roll = float(here.get("wrist_roll", 0.0))
    origin = kin.world.grasp_point(here, 0.04)
    targets = (
        np.array([origin[0], origin[1], _CARRY_Z]),
        np.array([_OFFER_XY[0], _OFFER_XY[1], _CARRY_Z]),
    )
    logger.info(
        "offer: lift to %.2f m, then out to (%.2f, %.2f)",
        _CARRY_Z, _OFFER_XY[0], _OFFER_XY[1],
    )
    seed = here
    for leg, (target, duration_s) in enumerate(zip(targets, (2.5, 2.0)), 1):
        pose, err, _tilt = kin.ik(
            target, 0.04, hold, seed, roll, limits=limits, one_branch=True,
        )
        if err > _OFFER_ERR_M:
            logger.info("offer: no pose within 15 mm (err %.0f mm)", 1000 * err)
            return "unreachable"
        # From the measured jaw, the first steps would let go of the squeeze.
        if not follow(joint_line({**seed, "gripper": hold}, {**pose, "gripper": hold}), duration_s):
            return "stopped"
        time.sleep(0.3)
        _log_grip(arm, f"offer check {leg}", hold)
        if not still_holding():
            return "dropped"
        seed = arm.get_joints()
    say(f"Got the {resolved}.")
    return "offered"


_DESCENT_LATERAL_M = 0.005


def _read_settled(camera, settled_at: float):
    """One frame captured after the arm finished moving at ``settled_at``."""
    read_after = getattr(camera, "read_after", None)
    if callable(read_after):
        return read_after(settled_at)
    period = 1.0 / 30.0
    ok, frame = False, None
    for _ in range(8):
        started = time.perf_counter()
        ok, frame = camera.read()
        if not ok or frame is None:
            return False, None
        if time.perf_counter() - started >= 0.5 * period:
            return True, frame
    return bool(ok), frame


def _shape_of(resolved: str) -> str:
    from so101_sim.kinematics import FLAT_CLASSES, ROUND_CLASSES

    if resolved in ROUND_CLASSES:
        return "sphere"
    if resolved in FLAT_CLASSES:
        return "flat"
    if resolved == "bottle":
        return "bottle"
    return "upright"


def _box_usable(box, shape: str, width: int, height: int) -> bool:
    """True when ``locate_on_table`` will accept this box's edges."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return False
    if shape == "flat" and y1 <= 1:
        return False
    return True


def _rank_views(views: list) -> list:
    """Largest usable box first, then the earliest pan."""
    return sorted(views, key=lambda view: (-view["area"], view["order"]))


class _GraspPlan:
    def __init__(self, poses, xyz, beside=None):
        self.poses = poses
        self.xyz = np.asarray(xyz, float)
        self.beside = beside  # the last pose moved sideways to the fixed finger's gap, or None

    @property
    def grasp(self) -> dict:
        """The pose the jaw closes at."""
        return self.beside if self.beside is not None else self.poses[-1]


def _log_sighting(pp, cam, frame, det, resolved, joints) -> None:
    """Log one detection with its table fix and the joints that took the frame."""
    height, width = frame.shape[:2]
    shape = _shape_of(resolved)
    if det.area / float(height * width) < float(pp["min_box_area_frac"]):
        keep = "too small"
    elif not _box_usable(det.box, shape, width, height):
        keep = "clipped"
    else:
        keep = "kept"
    fix = "no table fix"
    try:
        from so101_sim.kinematics import shared_kinematics

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
        K, dist = kin.intrinsics((cam or {}).get("intrinsics"), width, height)
        loc = kin.locate_on_table(
            det.box, joints, K, dist, (cam or {}).get("mount"),
            float(pp.get("table_z", -0.005)), (width, height), shape=shape,
        )
        if loc is not None:
            fix = "xyz (%.3f, %.3f, %.3f) m, %.0f mm wide, %.0f mm tall" % (
                *loc["xyz"], 1000 * loc["width"], 1000 * loc["height"],
            )
    except Exception as exc:
        fix = f"locate failed: {exc}"
    logger.info(
        "scan sighting: %s box %s score %.2f %s | %s | joints pan %.1f lift %.1f elbow %.1f flex %.1f roll %.1f",
        resolved, tuple(round(float(v), 1) for v in det.box), float(det.score), keep, fix,
        float(joints.get("shoulder_pan", 0.0)), float(joints.get("shoulder_lift", 0.0)),
        float(joints.get("elbow_flex", 0.0)), float(joints.get("wrist_flex", 0.0)),
        float(joints.get("wrist_roll", 0.0)),
    )


def _sweep_views(arm, camera, find, name, pose, pans, pp, stop_check, *, near=False, first=True, cam=None, accept=None):
    """Walk the pans until one settled frame can be grasped.

    A miss, a box cut by the image edge, or a box whose grasp does not solve
    does not end the sweep. ``accept(batch)`` returns true when that pan's
    boxes produced a grasp. With no ``accept``, the first usable box stops
    the sweep. Returns ``(views, status, saw_small, saw_large)``.
    """
    views, saw_small, saw_large = [], False, False
    min_frac = float(pp["min_box_area_frac"])
    for i, pan in enumerate(pans):
        arm.go_to(
            {**pose, "shoulder_pan": float(pan)},
            duration_s=2.5 if first and i == 0 else 1.2,
            stop_check=stop_check,
        )
        if stop_check and stop_check():
            return views, "stopped", saw_small, saw_large
        logger.info("scan near pan %.1f" if near else "scan pan %.1f", float(pan))
        settled_at = time.time()
        ok, frame = _read_settled(camera, settled_at)
        joints = dict(arm.get_joints())
        if not ok or frame is None:
            continue
        dets, resolved = find(frame, name)
        if resolved is None:
            return views, "unknown", saw_small, saw_large
        if not dets:
            continue
        shape = _shape_of(resolved)
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        batch = []
        for det in dets:
            _log_sighting(pp, cam, frame, det, resolved, joints)
            if det.area / frame_area < min_frac:
                saw_small = True
                continue
            saw_large = True
            if not _box_usable(det.box, shape, width, height):
                continue
            batch.append({
                "frame": frame,
                "det": det,
                "dets": list(dets),
                "resolved": resolved,
                "joints": joints,
                "area": float(det.area),
                "order": i,
                "settled_at": settled_at,
            })
        if not batch:
            continue
        if accept is None or accept(batch):
            views.extend(batch)
            break
        logger.info("scan sighting not graspable at pan %.1f, continuing", float(pan))
    return views, "ok", saw_small, saw_large


def _search_views(arm, camera, find, name, pp, stop_check, arm_camera, cam=None, accept=None):
    """High sweep first. The near pose runs only when that sweep stored nothing."""
    base = dict(pp["scan_pose"])
    pans = pp.get("scan_pans") if camera is arm_camera else None
    pans = list(pans or [base.get("shoulder_pan", 0.0)])
    views, status, saw_small, saw_large = _sweep_views(
        arm, camera, find, name, base, pans, pp, stop_check, cam=cam, accept=accept,
    )
    if status != "ok" or views or camera is not arm_camera or not pp.get("scan_near_pose"):
        return views, status, saw_small, saw_large
    near = dict(pp["scan_near_pose"])
    near_pans = list(pp.get("scan_near_pans") or [near.get("shoulder_pan", 0.0)])
    more, status, small, large = _sweep_views(
        arm, camera, find, name, near, near_pans, pp, stop_check,
        near=True, first=False, cam=cam, accept=accept,
    )
    return views + more, status, saw_small or small, saw_large or large


def _path_within(kin, pre, path, width, table_z) -> bool:
    top = kin.world.grasp_point(pre, width)
    for pose in path:
        point = kin.world.grasp_point(pose, width)
        if float(np.linalg.norm(point[:2] - top[:2])) > _DESCENT_LATERAL_M:
            return False
        if kin.lowest_point(pose) < table_z:
            return False
    return True


def _descent_poses(kin, pre, grasp, width, table_z, limits=None):
    """Joint-space steps from ``pre`` to ``grasp``.

    When those steps leave the vertical by more than 5 mm, use the Cartesian
    waypoints so the bound still holds. ``None`` when neither path stays above
    the table inside that band.
    """
    from so101_sim.kinematics import joint_line

    path = joint_line(pre, grasp)
    if _path_within(kin, pre, path, width, table_z):
        return path
    cart = kin.waypoints(pre, grasp, width, limits=limits)
    if cart and len(cart) >= 2 and _path_within(kin, pre, cart, width, table_z):
        logger.info("IK reach: joint line leaves the 5 mm band, descending on Cartesian steps")
        return cart
    logger.info("IK reach: descent leaves the vertical")
    return None


def _plan_from_view(ctx, view):
    """One table locate and one grasp solve from a stored frame. ``None`` if either fails."""
    pp = ctx.cfg["pick_place"]
    try:
        from so101_sim.kinematics import principal_axis_in_box, shared_kinematics

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    except Exception as e:
        logger.warning("IK reach unavailable: %s", e)
        return None
    frame, det, resolved, joints = view["frame"], view["det"], view["resolved"], view["joints"]
    height, width = frame.shape[:2]
    cam = (ctx.cfg.get("cameras") or {}).get("arm") or {}
    K, dist = kin.intrinsics(cam.get("intrinsics"), width, height)
    table_z = float(pp.get("table_z", -0.005))
    shape = _shape_of(resolved)
    loc = kin.locate_on_table(
        det.box, joints, K, dist, cam.get("mount"), table_z, (width, height), shape=shape,
    )
    if loc is None:
        logger.info("IK reach: no table fix for the %s box %s", resolved, det.box)
        return None
    logger.info("IK locate: %s", resolved)
    p = loc["xyz"].copy()
    p[2] = min(p[2], table_z + float(pp.get("max_grasp_height_m", 0.03)))
    obstacle = (
        (*loc["xyz"][:2], 0.5 * loc["width"], table_z + loc["height"])
        if shape in ("bottle", "upright")
        else None
    )
    oriented = resolved in pp.get("wrist_align_classes", [])
    object_axis = None
    if oriented:
        axis = principal_axis_in_box(
            frame, det.box, min_elongation=float(pp.get("wrist_axis_min_elongation", 1.6)),
        )
        if axis is not None:
            found = kin.table_axis(
                axis["center"], axis["direction"], axis["length_px"], joints, K, dist,
                cam.get("mount"), table_z,
            )
            if found is not None:
                centre, object_axis = found
                offset = (
                    float(pp.get("wrist_align_center_offset_m", 0.018))
                    if abs(float(joints.get("wrist_roll", 0.0))) < 30.0
                    else 0.0
                )
                p[:2] = centre[:2] + np.asarray(object_axis[:2], float) * offset
        p[2] = max(p[2], table_z + 0.011)
    grasp_width = (
        float(pp.get("wrist_align_grasp_width_m", 0.024)) if oriented else loc["width"]
    )
    above_m = float(
        pp.get("wrist_align_pregrasp_above_m", 0.10) if oriented else pp.get("pregrasp_above_m", 0.06)
    )
    # A standing grasp, and a rolled toothbrush, need the pipeline backstop.
    # The workshop's 67.5° wrist cap has no pose that stays outside the bottle
    # or turns the jaws across the brush.
    limits = GRASP_LIMITS if (obstacle or oriented) else None
    logger.info("IK solve")
    solved = kin.solve_grasp(
        p, above_m=above_m, width=grasp_width, table_z=table_z, wrist_roll=0.0,
        limits=limits, obstacle=obstacle, object_axis=object_axis,
    )
    logger.info(
        "IK reach: %s at (%.3f, %.3f, %.3f) m, %.0f mm wide -> %s",
        resolved, *p, 1000 * grasp_width, "solved" if solved else "out of reach",
    )
    if solved is None:
        return None
    pregrasp, grasp = solved
    descent = _descent_poses(kin, pregrasp, grasp, grasp_width, table_z, limits)
    if descent is None:
        return None
    beside = None
    if not obstacle:
        beside = kin.grasp_beside(
            grasp, p, float(pp.get("fixed_jaw_gap_m", 0.005)), grasp_width, table_z=table_z, limits=limits,
        )
    return _GraspPlan([pregrasp, *descent], p, beside)


def _view_here(arm, camera, find, name, pp, cam=None):
    """One fresh frame at the pose the arm is already holding, or ``None``."""
    settled_at = time.time()
    ok, frame = _read_settled(camera, settled_at)
    if not ok or frame is None:
        return None
    joints = dict(arm.get_joints())
    dets, resolved = find(frame, name)
    if resolved is None or not dets:
        return None
    for det in dets:
        _log_sighting(pp, cam, frame, det, resolved, joints)
    shape = _shape_of(resolved)
    height, width = frame.shape[:2]
    frame_area = float(height * width)
    min_frac = float(pp["min_box_area_frac"])
    usable = [
        det for det in dets
        if det.area / frame_area >= min_frac and _box_usable(det.box, shape, width, height)
    ]
    if not usable:
        return None
    det = max(usable, key=lambda item: item.area)
    return {
        "frame": frame, "det": det, "dets": list(dets), "resolved": resolved,
        "joints": joints, "area": float(det.area), "order": 0, "settled_at": settled_at,
    }


def _follow(arm, poses, stop_check, limits, duration_s: float) -> bool:
    if not poses:
        return True
    step = duration_s / len(poses)
    for pose in poses:
        arm.go_to(pose, duration_s=step, stop_check=stop_check, limits=limits)
        if stop_check and stop_check():
            return False
    return True


def _execute_plan(arm, plan, stop_check, limits) -> bool:
    """Move to the pose just above the object, then straight down, then
    sideways. Each leg starts once the arm has arrived at the end of the last,
    so a lagging joint does not cut the corner onto the object. The jaw stays
    open."""
    from so101_sim.kinematics import joint_line

    legs = [(joint_line(arm.get_joints(), plan.poses[0]), 1.5), (plan.poses[1:], 1.5)]
    if plan.beside is not None:
        legs.append(([plan.beside], 0.5))
    for poses, duration_s in legs:
        if not _follow(arm, poses, stop_check, limits, duration_s):
            return False
        # 0.5° is about 2 mm at the jaws. With the default 1° of progress the
        # last degree never counts as moving, and the goal is resent until the timeout.
        arm.move_to(poses[-1], tolerance_deg=0.5, progress_eps_deg=0.1, stop_check=stop_check, limits=limits)
        if stop_check and stop_check():
            return False
    return True


def _return_along(arm, plan, stop_check, limits) -> bool:
    """Walk the descent back to the pre-grasp. The gripper is already open."""
    back = list(reversed(plan.poses))
    return _follow(arm, back if plan.beside is not None else back[1:], stop_check, limits, 1.5)


def run_pick_place(ctx, cmd) -> bool:
    """Find the named object, grasp it, and hold it out. The jaws stay closed."""
    pp, g = ctx.cfg["pick_place"], ctx.cfg["grip"]
    name = (cmd.object_name if cmd is not None else "") or ""
    if ctx.detector is None:
        ctx.say("Object detection isn't available - can't do pick and place.")
        return False
    if not name:
        ctx.say("What should I pick up? Try: pick up the ball.")
        return False
    camera = ctx.arm_camera or ctx.camera
    if camera is None:
        ctx.say("I can't see anything - both cameras are offline.")
        return False
    if camera is not ctx.arm_camera:
        ctx.say("Arm camera is offline - using the operator camera (coarser aim).")

    ctx.say(f"Looking for a {name}.")
    cams = ctx.cfg.get("cameras") or {}
    cam = cams.get("arm") if camera is ctx.arm_camera else cams.get("operator")
    use_ik = (
        pp.get("reach", "fixed") == "ik"
        and ctx.arm_camera is not None
        and camera is ctx.arm_camera
    )
    plan = None
    saw_ungraspable = False

    def accept(batch):
        nonlocal plan, saw_ungraspable
        saw_ungraspable = True
        for view in _rank_views(batch):
            plan = _plan_from_view(ctx, view)
            if plan is not None:
                return True
        plan = None
        return False

    views, status, saw_small, saw_large = _search_views(
        ctx.arm, camera, ctx.detector.find, name, pp, ctx.stop_check, ctx.arm_camera, cam,
        accept if use_ik else None,
    )
    if status == "stopped" or ctx.stop_check():
        return False
    if status == "unknown":
        ctx.say(f"Sorry - I don't know what a {name} looks like.")
        ctx.arm.go_to_rest()
        return False
    if not views:
        if saw_ungraspable:
            ctx.say(f"The {name} is out of reach.")
        elif saw_small and not saw_large:
            ctx.say(f"The {name} I can see is too small to grab safely.")
        else:
            ctx.say(f"I can't see a {name} in the workspace.")
        ctx.arm.go_to_rest()
        return False
    ranked = _rank_views(views)
    winner = ranked[0]
    resolved = winner["resolved"]
    by_score = sorted(winner["dets"], key=lambda det: det.score, reverse=True)
    if len(by_score) > 1 and by_score[0].score / max(by_score[1].score, 1e-6) < float(pp["ambiguous_ratio"]):
        ctx.say(f"I see {len(by_score)} of those - taking the clearest one.")

    gripped, hold = False, 0.0
    attempts = 1 + int(pp.get("retries", 1))
    for attempt in range(attempts):
        if ctx.stop_check():
            return False
        if use_ik:
            # The pipeline backstop, as the pipeline sends it. The workshop's
            # 67.5° wrist cap holds even a ball's pre-grasp ~24° off vertical,
            # and the fingers then swing in on the way down.
            if not _execute_plan(ctx.arm, plan, ctx.stop_check, GRASP_LIMITS):
                return False
        else:
            _reach_fixed(ctx)
        if ctx.stop_check():
            return False
        gripped, hold = close_with_feedback(
            ctx.arm, g, stop_check=ctx.stop_check, start_from=OPEN_GRIPPER, ease=False,
        )
        if ctx.stop_check():
            return False
        if gripped and holding(ctx.arm, g, hold):
            _log_grip(ctx.arm, "grasp", hold, plan)
            break
        _log_grip(ctx.arm, "grasp missed", hold, plan)
        gripped = False
        ctx.arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
        ctx.arm.go_to({"gripper": OPEN_GRIPPER}, duration_s=0.6, stop_check=ctx.stop_check)
        if attempt + 1 >= attempts or not use_ik:
            continue
        ctx.say("The grasp slipped - trying once more.")
        if not _return_along(ctx.arm, plan, ctx.stop_check, GRASP_LIMITS):
            return False
        fresh = _view_here(ctx.arm, camera, ctx.detector.find, name, pp, cam)
        if fresh is None:
            logger.info("retry: no usable box from the pre-grasp - giving up")
            break
        plan = _plan_from_view(ctx, fresh)
        if plan is None:
            logger.info("retry: the fresh box gives no grasp - giving up")
            break
    if ctx.stop_check():
        return False
    if not gripped:
        ctx.say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        park_open(ctx.arm)
        return False

    # The grasp limits allow the wrist to 90°, which the high carry needs
    # for every object. The tighter workshop cap drops the hand back down.
    status = _offer(
        ctx.arm, hold, ctx.stop_check, GRASP_LIMITS, ctx.say, resolved,
        lambda: holding(ctx.arm, g, hold), lambda: ease_grip(ctx.arm, g),
    )
    ease_grip(ctx.arm, g)
    if status == "stopped":
        return False
    if status == "dropped":
        ctx.say(f"I dropped the {resolved}.")
        park_open(ctx.arm)
        return False
    if status != "offered":
        ctx.say(f"I can't lift the {resolved} clear of the other objects.")
        return False
    ctx.shared_state["grip_lock"] = True
    ctx.shared_state["grip_lock_pos"] = hold
    ctx.say("Done.")
    return True


def _reach_fixed(ctx) -> None:
    """Jaws open over the object at the configured reach, then down onto it."""
    pp = ctx.cfg["pick_place"]
    above = {**pp["approach_pose"], "shoulder_pan": ctx.arm.get_joints()["shoulder_pan"]}
    ctx.arm.go_to({**above, "gripper": OPEN_GRIPPER}, duration_s=1.5, stop_check=ctx.stop_check)
    ctx.arm.go_to({"shoulder_lift": float(pp["descend_lift"])}, duration_s=1.5, stop_check=ctx.stop_check)


def run_release(ctx, cmd=None) -> bool:
    """Open the jaws where they are. The arm stays put."""
    ctx.shared_state.pop("grip_lock", None)
    ctx.shared_state.pop("grip_lock_pos", None)
    ctx.say("Released.")
    ctx.arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    ctx.arm.go_to(
        {"gripper": JOINT_LIMITS["gripper"][1]}, duration_s=0.8, stop_check=ctx.stop_check,
    )
    return not ctx.stop_check()


def register_manipulation(registry) -> None:
    """Add grip, release, pick_place and fetch_block to ``registry``."""
    registry.add(Intent.GRIP, run_grip)
    registry.add(Intent.RELEASE, run_release)
    registry.add(Intent.PICK_PLACE, run_pick_place)
    registry.add(Intent.FETCH_BLOCK, run_fetch_block)
