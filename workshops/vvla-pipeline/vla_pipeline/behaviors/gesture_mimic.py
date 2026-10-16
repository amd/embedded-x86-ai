# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gesture-mimic behavior: the robot follows the human hand around the camera
frame, via **direct camera-frame motion mapping** (see
:mod:`vla_pipeline.utils.gesture_map`).

The image is treated as a control surface and its signals are mapped straight
onto SO-101 joints - the same absolute-target, limit-clamped commands that
``test_ros2_arm_motion`` issues - rather than solving the human arm's geometry:

- **YOLOv26s-pose (NPU)** → the nearest person's right wrist position in the
  frame: horizontal → left/right (shoulder_pan), vertical → up/down
  (shoulder_lift and elbow_flex as a locked pair). The right elbow gives the
  forearm direction for wrist_flex. The YOLO wrist is also the **fallback
  controller**: when MediaPipe loses the hand the robot keeps following the
  wrist for x/y until the hand returns.
- **MediaPipe Hands (CPU)** →
    * hand **size** (how near the hand is) → forward/back (z): reaching forward
      extends shoulder_lift with elbow_flex negative; the opposite pulls back,
    * wrist→knuckle direction measured against the forearm (the wrist bend)
      → wrist_flex,
    * thumb↔index **line** (with the knuckle-line reference, then the wrist→knuckle
      angle, as reliability-weighted fallbacks) → wrist_roll (forward-most = the
      calibration midpoint, 0),
    * thumb↔index **distance** → gripper, as a continuous ramp: closed for the
      first ``grip_close_frac`` of the calibrated min/max range, then opening
      linearly up to ~90%.

The pinch always drives the gripper: a grip locked by *grip* or *pick_place*
is released when mimic starts (``vla_pipeline.main``), so a held object drops
when the fingers open. The gripper holds its last position whenever the hand
is unseen and only moves when a new thumb↔index distance is measured.

Depth neutral (optional, one-time per camera position):
    python -m vla_pipeline.behaviors.gesture_mimic --calibrate-depth
Hold your hand at a comfortable mid-distance; paste the printed
``hand_size_neutral`` into ``behaviors.gesture_mimic`` in ``config/pipeline.yaml``.
With it left at 0 the neutral auto-calibrates from the first few frames.

Gripper distance (recommended, one-time per operator):
    python -m vla_pipeline.behaviors.gesture_mimic --calibrate-grip
Pinch your thumb & index together, then spread them wide when prompted; paste
the printed ``grip_min`` / ``grip_max`` into ``config/pipeline.yaml``.

Independent test
----------------
    python -m vla_pipeline.behaviors.gesture_mimic --dry-run     # no robot
Press 'q' in the preview window (or Ctrl-C) to stop.
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, make_arm
from vla_pipeline.utils.config import load_config
from vla_pipeline.utils.gesture_map import CameraMimicMapper, mapper_from_config
from vla_pipeline.vision.mediapipe_hands import HandTracker
from vla_pipeline.vision.yolo_pose_npu import KP, PoseEstimator

logger = logging.getLogger(__name__)

def _primary_person(pose: PoseEstimator, frame):
    """Largest-box detection → (kpts[17,3], boxes, scores, kpts_all) or Nones.

    Runs inference once and reuses it for both control and the overlay so the
    NPU isn't hit twice per frame.
    """
    boxes, scores, kpts_all = pose.infer(frame)
    if len(boxes) == 0:
        return None, boxes, scores, kpts_all
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return kpts_all[int(np.argmax(areas))], boxes, scores, kpts_all


def _pick_arm(kpts, thr: float, side: str = "r"):
    """(wrist_xy, elbow_xy) of one fixed arm (default right), like the workshop.

    wrist_xy is None when that wrist is below ``thr`` (and then so is elbow_xy);
    elbow_xy is None when only the elbow is below it.
    """
    w, e = KP[f"{side}_wrist"], KP[f"{side}_elbow"]
    if kpts[w, 2] < thr:
        return None, None
    wrist = (float(kpts[w, 0]), float(kpts[w, 1]))
    elbow = (float(kpts[e, 0]), float(kpts[e, 1])) if kpts[e, 2] >= thr else None
    return wrist, elbow


def run_gesture_mimic(
    cfg: dict,
    arm: ArmClient,
    pose: PoseEstimator,
    hands: HandTracker,
    camera,  # CameraClient (operator role)
    stop_check=None,
    show_preview: bool = True,
) -> None:
    """Run the mimic loop until 'q', Ctrl-C, or ``stop_check()`` is True."""
    mapper = mapper_from_config(cfg)
    thr = pose.kpt_threshold
    missing = 0  # consecutive frames with no person (→ reset to avoid a jump)

    logger.info(
        "Gesture mimic running (direct camera-frame mapping) - show your "
        "hand to the operator camera. Move it around the frame; reach in/out "
        "for forward/back; tilt to roll; pinch to grip."
    )
    try:
        while True:
            if stop_check and stop_check():
                break
            ok, frame = camera.read()
            if not ok:
                logger.warning("Operator camera frame unavailable - holding pose.")
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]

            # --- Hand location from YOLO pose (NPU) ---
            kpts, boxes, scores, kpts_all = _primary_person(pose, frame)
            wrist_xy, elbow_xy = (
                _pick_arm(kpts, thr) if kpts is not None else (None, None)
            )
            if kpts is None:
                missing += 1
                if missing == 5:
                    mapper.reset()  # person gone a while → drop stale smoothing
            else:
                missing = 0

            # --- Depth / roll / gripper from MediaPipe (CPU) ---
            obs, mp_results = hands.process(frame)
            hand_size = obs.hand_size if obs is not None else None
            hand_roll = obs.roll if obs is not None else None
            roll_ti = obs.roll_ti if obs is not None else None
            roll_ref = obs.roll_ref if obs is not None else None
            roll_ti_w = obs.roll_ti_weight if obs is not None else 0.0
            pinch = obs.pinch if obs is not None else None

            # --- Map → command (skip while we have neither a hand nor a wrist) ---
            if wrist_xy is not None or obs is not None:
                joints = mapper.step(
                    wrist_xy,
                    w,
                    h,
                    hand_size=hand_size,
                    hand_roll_deg=hand_roll,
                    roll_ti_deg=roll_ti,
                    roll_ref_deg=roll_ref,
                    roll_ti_weight=roll_ti_w,
                    pinch=pinch,
                    elbow_xy=elbow_xy,
                )
                arm.send_joints(joints)
            else:
                joints = None

            if show_preview:
                vis = frame
                if kpts is not None:
                    vis = pose.draw(frame, boxes, scores, kpts_all)
                vis = hands.draw(vis, obs, mp_results)
                _hud(vis, joints, wrist_xy, obs)
                cv2.imshow("gesture_mimic", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show_preview:
            try:
                cv2.destroyWindow("gesture_mimic")
            except cv2.error:
                pass
        logger.info("Gesture mimic stopped - returning to rest pose.")
        # Park home as ONE smooth move. resend_s=None disables move_to's resend
        # loop (re-writing the goal restarts the servo profile, which looked like
        # the arm repeatedly lurching toward rest on a stop). We deliberately do
        # NOT pass stop_check here: on a voice "stop" the stop event is already
        # set, and honoring it would abort this park instantly and leave the arm
        # wherever it was - we want it to complete the single trip home.
        arm.go_to_rest(resend_s=None)


def _hud(vis, joints, wrist_xy, obs) -> None:
    """Draw a compact status line + the commanded joints for the operator."""
    cv2.putText(
        vis,
        "gesture mimic [direct] - 'q' to stop",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    if wrist_xy is not None:
        cv2.circle(vis, (int(wrist_xy[0]), int(wrist_xy[1])), 8, (0, 255, 0), 2)
    if joints:
        grip_deg = joints.get("gripper")  # absent until MediaPipe reports a pinch
        grip = "hold" if grip_deg is None else f"{grip_deg:3.0f} ({_grip_label(grip_deg)})"
        line = (
            f"pan {joints['shoulder_pan']:+5.0f}  lift {joints['shoulder_lift']:+5.0f}  "
            f"elb {joints['elbow_flex']:+5.0f}  wflex {joints['wrist_flex']:+5.0f}  "
            f"roll {joints['wrist_roll']:+5.0f}  grip {grip}"
        )
        cv2.putText(
            vis,
            line,
            (12, vis.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def _grip_label(grip_deg: float) -> str:
    """Describe the continuous gripper opening for the HUD."""
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    if grip_deg <= GRIPPER_CLOSED_FLOOR + 2.0:
        return "closed"
    if grip_deg >= 85.0:
        return "open"
    return f"{grip_deg:.0f}%"


# =============================================================================
# Depth-neutral calibration mode
# =============================================================================


def calibrate_depth(
    cfg: dict, hands: HandTracker, camera, sample_s: float = 2.5
) -> None:
    """Measure the neutral hand size at a comfortable mid-distance and print the
    ``hand_size_neutral`` value to paste into ``config/pipeline.yaml``."""
    print(
        "\nHold your open hand toward the operator camera at a comfortable "
        "mid-distance (the pose you want to be 'neither forward nor back')."
    )
    input("Press Enter when in position... ")
    samples: list[float] = []
    deadline = time.time() + sample_s
    while time.time() < deadline:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.03)
            continue
        obs, _ = hands.process(frame)
        if obs is not None and obs.hand_size > 0:
            samples.append(obs.hand_size)
    if len(samples) < 5:
        raise SystemExit("Calibration failed - hand not visible long enough.")
    neutral = float(np.median(samples))
    print(
        f"\nMeasured neutral hand size: {neutral:.4f} "
        f"(n={len(samples)}, std {np.std(samples):.4f})"
    )
    print("Paste into config/pipeline.yaml under behaviors.gesture_mimic:\n")
    print(f"    hand_size_neutral: {neutral:.4f}")


def calibrate_grip(
    cfg: dict, hands: HandTracker, camera, sample_s: float = 2.5
) -> None:
    """Measure the thumb↔index ``pinch`` at fully-closed and fully-open and print
    the ``grip_min`` / ``grip_max`` values to paste into ``config/pipeline.yaml``.

    These bound the normalized pinch the continuous gripper ramp uses, so a closed
    pinch maps to fully closed and a wide spread to fully open.
    """

    def _sample(prompt: str, reducer) -> float:
        """Prompt, sample obs.pinch for sample_s seconds, and reduce the readings to one value."""
        print(prompt)
        input("Press Enter when in position... ")
        vals: list[float] = []
        deadline = time.time() + sample_s
        while time.time() < deadline:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.03)
                continue
            obs, _ = hands.process(frame)
            if obs is not None and obs.pinch >= 0:
                vals.append(obs.pinch)
        if len(vals) < 5:
            raise SystemExit("Calibration failed - hand not visible long enough.")
        return float(reducer(vals))

    print("\nGripper distance calibration - thumb↔index distance min/max.")
    grip_min = _sample(
        "\n1/2: Pinch your thumb and index finger together (gripper CLOSED).", np.median
    )
    grip_max = _sample(
        "\n2/2: Spread your thumb and index finger as wide apart as is "
        "comfortable (gripper WIDE OPEN).",
        np.median,
    )
    if grip_max <= grip_min:
        raise SystemExit(
            f"Calibration failed - max ({grip_max:.3f}) not greater "
            f"than min ({grip_min:.3f}). Re-run and spread wider."
        )
    print(f"\nMeasured grip_min={grip_min:.4f}, grip_max={grip_max:.4f}")
    print("Paste into config/pipeline.yaml under behaviors.gesture_mimic:\n")
    print(f"    grip_min: {grip_min:.4f}")
    print(f"    grip_max: {grip_max:.4f}")


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run gesture mimic, or one of the calibration modes."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Gesture mimic behavior - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true", help="no robot hardware")
    parser.add_argument(
        "--calibrate-depth",
        action="store_true",
        help="measure hand_size_neutral at a comfortable distance and exit",
    )
    parser.add_argument(
        "--calibrate-grip",
        action="store_true",
        help="measure thumb↔index grip_min/grip_max and exit",
    )
    parser.add_argument(
        "--device", choices=["npu", "cpu"], default=None, help="YOLO device override"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    from vla_pipeline.vision.camera import make_camera

    hands = HandTracker(cfg)
    camera = make_camera(cfg, "operator")
    if args.calibrate_depth or args.calibrate_grip:
        try:
            if args.calibrate_depth:
                calibrate_depth(cfg, hands, camera)
            if args.calibrate_grip:
                calibrate_grip(cfg, hands, camera)
        finally:
            camera.close()
            hands.close()
        return

    pose = PoseEstimator(cfg, device=args.device)
    arm: ArmClient = DryRunArm(verbose=False) if args.dry_run else make_arm(cfg)
    try:
        run_gesture_mimic(cfg, arm, pose, hands, camera)
    finally:
        camera.close()
        hands.close()
        arm.close()


if __name__ == "__main__":
    main()
