# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gesture mimic - the workshop's flagship composition.

One tick of mimic is the whole pipeline in miniature:

    camera frame ──► YOLO-pose (NPU) ──► wrist x/y + forearm ─┐
                └──► MediaPipe (CPU) ──► size/roll/pinch ─────┴──► mapper ──► arm
                                                                 (provided)  (ROS 2)

Your pinch always drives the gripper: a grip locked by pick-and-place or
grip is released when mimic starts, so a held object drops when you open
your fingers.

On a REAL arm the mimic session runs against LIVE_MIMIC_LIMITS - the base
envelope extended 25% (gripper excluded) - and restores the base
JOINT_LIMITS on exit; dry-run/synthetic stays on the base limits.

Everything around the tick - the loop, the preview window, the HUD that
attributes each joint to the TODO that feeds it - is provided. The tick
itself, the line where model outputs become robot motion, is yours.

Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Harness:  ``python app.py --dry-run --synthetic``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from common.imaging import (
    annotate_lines,
    draw_arm_panel,
    draw_skeleton,
    pick_arm,
    pick_nearest,
)
from common.motion import LIVE_MIMIC_LIMITS, set_active_limits

logger = logging.getLogger(__name__)


@dataclass
class MimicSignals:
    """One tick's worth of evidence for the HUD (PROVIDED)."""

    joints: dict = None  # what was commanded (None = nothing sent)
    wrist_xy: tuple = None  # pose model's wrist pick
    elbow_xy: tuple = None  # same arm's elbow (forearm → wrist_flex)
    boxes: object = None  # raw pose outputs, for the skeleton overlay
    scores: object = None
    kpts: object = None
    obs: object = None  # HandObservation (or None)
    raw: object = None  # raw MediaPipe results, for the hand overlay


def mimic_tick(ctx, frame) -> MimicSignals:
    """Turn ONE camera frame into ONE arm command.

    The composition, in order:

    1. ``ctx.pose.infer(frame)`` → boxes/scores/kpts; keep only the NEAREST
       person (largest box) via :func:`common.imaging.pick_nearest`, then pick
       a wrist AND its elbow with :func:`common.imaging.pick_arm` (threshold:
       ``ctx.pose.kpt_threshold``).
    2. ``ctx.hands.process(frame)`` → a HandObservation (or None) carrying
       hand_size / roll_ti / roll_ref / roll_ti_weight / roll / pinch.
    3. ``ctx.mapper.step(...)`` with everything from 1-2 (pass the elbow as
       ``elbow_xy=`` - wrist_flex measures your hand's direction against it
       to read the WRIST BEND; without it the forearm is assumed vertical)
       → a clamped joint dict - then ``ctx.arm.send_joints(joints)``.
       If NEITHER a wrist NOR a hand was seen, send nothing (hold pose).

    Return a :class:`MimicSignals` with everything you used - the HUD shows
    your work. Keep the RAW results from ``ctx.hands.process`` too (the
    ``raw`` field) so the UI can draw the MediaPipe hand overlay.
    """
    # >>> TODO 3.3: mimic_tick - notebooks/03_integration/01_behavior.ipynb
    boxes, scores, kpts = ctx.pose.infer(frame)
    wrist_xy = None
    elbow_xy = None
    if len(boxes) > 0:
        person_kpts = kpts[pick_nearest(boxes)]  # nearest person only
        wrist_xy, elbow_xy = pick_arm(person_kpts, ctx.pose.kpt_threshold)

    obs, raw = ctx.hands.process(frame) if ctx.hands else (None, None)

    joints = None
    if wrist_xy is not None or obs is not None:
        h, w = frame.shape[:2]
        joints = ctx.mapper.step(
            wrist_xy,
            w,
            h,
            elbow_xy=elbow_xy,
            hand_size=obs.hand_size if obs else None,
            roll_ti=obs.roll_ti if obs else None,
            roll_ref=obs.roll_ref if obs else None,
            roll_ti_weight=obs.roll_ti_weight if obs else 0.0,
            roll_fallback=obs.roll if obs else None,
            pinch=obs.pinch if obs else None,
        )
        ctx.arm.send_joints(joints)

    return MimicSignals(
        joints=joints,
        wrist_xy=wrist_xy,
        elbow_xy=elbow_xy,
        boxes=boxes,
        scores=scores,
        kpts=kpts,
        obs=obs,
        raw=raw,
    )
    # <<< TODO 3.3


# =============================================================================
# PROVIDED: the loop, the preview, and the HUD around your tick
# =============================================================================

HUD_ATTRIBUTION = [
    "wrist x  -> shoulder_pan            [TODO 1.2 session -> TODO 3.3 tick]",
    "wrist y  -> lift+elbow (locked)     [TODO 1.2 session -> TODO 3.3 tick]",
    "wrist bend -> wrist_flex            [TODO 1.2+1.4     -> TODO 3.3 tick]",
    "hand size-> fwd reach (lift+elbow)  [TODO 1.4 hands   -> TODO 3.3 tick]",
    "thumb-index angle -> wrist_roll     [TODO 1.4 hands   -> TODO 3.3 tick]",
    "pinch    -> gripper                 [TODO 1.4 hands   -> TODO 3.3 tick]",
    "joints   -> /so101/joint_command    [TODO 2.7/2.8 arm client]",
]


def run_mimic(ctx, cmd=None) -> None:
    """PROVIDED: mimic loop + live UI. Runs until 'q', stop, or max_frames."""
    import cv2

    provider = getattr(ctx.pose, "provider", "?")
    # LIVE ARM ONLY: mimic runs against the 25%-extended envelope (gripper
    # kept at its calibrated jaw span). Restored in the finally below, so
    # scripted behaviors - and any dry-run/synthetic session - stay on the
    # base JOINT_LIMITS.
    live = bool(getattr(ctx.arm, "is_live", False))
    prev_limits = set_active_limits(LIVE_MIMIC_LIMITS) if live else None
    if ctx.shared_state.pop("grip_lock", None):
        ctx.shared_state.pop("grip_lock_pos", None)
        ctx.say("Your pinch drives the gripper now. Open your fingers to let go.")
    logger.info(
        "Mimic running (pose on %s%s) - move your hand; 'q' quits.",
        provider,
        ", live +25% envelope" if live else "",
    )
    frames = 0
    missing = 0  # consecutive frames with no person (→ reset to avoid a jump)
    last_signals = MimicSignals()
    try:
        while not ctx.stop_check():
            ok, frame = ctx.camera.read()
            if not ok:
                time.sleep(0.05)
                continue
            signals = mimic_tick(ctx, frame)
            last_signals = signals
            frames += 1

            # Person gone a while → drop stale smoothing/roll-center state so
            # the arm doesn't lurch when they reappear (same as the pipeline).
            if signals.kpts is None or len(signals.kpts) == 0:
                missing += 1
                if missing == 5:
                    ctx.mapper.reset()
            else:
                missing = 0

            if not ctx.headless or ctx.max_frames:
                vis = frame
                if signals.kpts is not None and len(signals.kpts) > 0:
                    vis = draw_skeleton(
                        frame,
                        signals.boxes,
                        signals.scores,
                        signals.kpts,
                        getattr(ctx.pose, "kpt_threshold", 0.3),
                    )
                else:
                    vis = frame.copy()
                # MediaPipe hand landmarks + pinch line (same overlay the
                # real pipeline shows).
                if ctx.hands is not None and hasattr(ctx.hands, "draw"):
                    vis = ctx.hands.draw(vis, signals.obs, signals.raw)
                if signals.wrist_xy is not None:
                    cv2.circle(
                        vis,
                        (int(signals.wrist_xy[0]), int(signals.wrist_xy[1])),
                        9,
                        (0, 255, 0),
                        2,
                    )
                    if signals.elbow_xy is not None:  # the forearm wrist_flex follows
                        e = (int(signals.elbow_xy[0]), int(signals.elbow_xy[1]))
                        cv2.line(
                            vis,
                            e,
                            (int(signals.wrist_xy[0]), int(signals.wrist_xy[1])),
                            (0, 255, 255),
                            2,
                        )
                        cv2.circle(vis, e, 6, (0, 255, 255), 2)
                live_tag = "  [LIVE +25% RANGE]" if live else ""
                annotate_lines(
                    vis,
                    [f"gesture mimic - pose on {provider}{live_tag}"]
                    + HUD_ATTRIBUTION,
                    scale=0.5,
                )
                panel = draw_arm_panel(
                    {**ctx.arm.get_joints(), **(signals.joints or {})},
                    width=300,
                    height=vis.shape[0],
                )
                canvas = np.hstack([vis, panel])
                if ctx.headless:
                    ctx.shared_state["last_ui_frame"] = canvas
                else:
                    cv2.imshow("workshop: gesture mimic", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if ctx.max_frames and frames >= ctx.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if prev_limits is not None:  # back to the base envelope BEFORE
            set_active_limits(prev_limits)  # parking / scripted behaviors
        if not ctx.headless:
            try:
                cv2.destroyWindow("workshop: gesture mimic")
            except Exception:
                pass
        logger.info("Mimic stopped after %d frames - parking the arm.", frames)
        ctx.arm.go_to_rest(duration_s=1.5)
    return last_signals
