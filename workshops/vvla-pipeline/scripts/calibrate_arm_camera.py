#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Calibrate the SO-101 wrist camera so pick_place can locate objects (``reach: ik``).

pick_place turns the YOLO box of an object into a position on the table with
the camera's intrinsics and its mount on the gripper. Until both are
calibrated it uses the digital twin's camera model, which only matches a
real arm whose camera sits exactly like the twin's. Run on the hardware with
the pipeline stopped (the script opens the arm and the camera directly):

1. ``intrinsics`` - hold a printed checkerboard in front of the arm camera
   at 12+ distances and angles, SPACE captures a view, ENTER solves. Paste
   the printed ``intrinsics`` into ``cameras.arm`` of config/pipeline.yaml
   (and workshop/*/config/workshop.yaml).
2. ``mount`` - tape the checkerboard flat on the table. Jog the arm (keys
   below) to ~8 poses that see the whole board from different angles,
   SPACE at each, ENTER solves. Paste the printed ``mount``.
3. ``check`` - put the sports ball at a spot you measured in the arm's base frame
   (x forward from the base, y to the left, metres) and compare.

    python scripts/calibrate_arm_camera.py intrinsics --board 9x6 --square-mm 24
    python scripts/calibrate_arm_camera.py mount --board 9x6 --square-mm 24
    python scripts/calibrate_arm_camera.py check --object sports_ball --at 0.28 0.06
    python scripts/calibrate_arm_camera.py check --object sports_ball --sim   # on the twin

Jog keys (mount): a/d pan, w/s shoulder lift, i/k elbow, j/l wrist flex,
u/o wrist roll, [ / ] step size; SPACE capture, ENTER solve, q quit.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOG = {
    ord("a"): ("shoulder_pan", +1), ord("d"): ("shoulder_pan", -1),
    ord("w"): ("shoulder_lift", -1), ord("s"): ("shoulder_lift", +1),
    ord("i"): ("elbow_flex", -1), ord("k"): ("elbow_flex", +1),
    ord("j"): ("wrist_flex", -1), ord("l"): ("wrist_flex", +1),
    ord("u"): ("wrist_roll", -1), ord("o"): ("wrist_roll", +1),
}
ENTER = (13, 10)


def _config(args) -> dict:
    from vla_pipeline.utils.config import load_config

    cfg = copy.deepcopy(load_config(args.config))
    cfg["robot"]["use_ros2"] = False
    cfg["cameras"]["arm"]["use_ros2"] = False
    if getattr(args, "sim", False):
        cfg["robot"]["backend"] = "sim"
        cfg["cameras"]["arm"]["source"] = "sim"
    return cfg


def _board(spec: str, square_mm: float):
    cols, rows = (int(v) for v in spec.lower().split("x"))
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
    return (cols, rows), obj


def _find(frame, size):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def _show(frame, corners, size, lines) -> int:
    import cv2

    view = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(view, size, corners, True)
    for i, text in enumerate(lines):
        cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("calibrate arm camera", view)
    return cv2.waitKey(30) & 0xFF


def _yaml_list(values, fmt="{:.6g}") -> str:
    return "[" + ", ".join(fmt.format(float(v)) for v in values) + "]"


# ---------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------
def cmd_intrinsics(args) -> int:
    import cv2

    size, objp = _board(args.board, args.square_mm)
    img_pts, shape = [], None
    if args.images:
        for path in sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg")):
            frame = cv2.imread(str(path))
            corners = _find(frame, size)
            shape = frame.shape[1::-1]
            if corners is not None:
                img_pts.append(corners)
        print(f"{len(img_pts)} images with the board in {args.images}")
    else:
        from vla_pipeline.vision.camera import make_camera

        cam = make_camera(_config(args), "arm")
        save = Path(args.save) if args.save else None
        if save:
            save.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                shape = frame.shape[1::-1]
                corners = _find(frame, size)
                key = _show(frame, corners, size, [
                    f"views {len(img_pts)}/{args.min_views}  board {'found' if corners is not None else '-'}",
                    "SPACE capture  ENTER solve  q quit",
                ])
                if key == ord("q"):
                    return 1
                if key == ord(" ") and corners is not None:
                    img_pts.append(corners)
                    if save:
                        cv2.imwrite(str(save / f"view_{len(img_pts):02d}.png"), frame)
                if key in ENTER and len(img_pts) >= args.min_views:
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()
    if len(img_pts) < 3:
        print("need at least 3 views of the board")
        return 1
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(img_pts), img_pts, shape, None, None)
    print(f"\nreprojection error {rms:.3f} px over {len(img_pts)} views (under ~0.5 px is good)\n")
    print("cameras:\n  arm:")
    print(f"    intrinsics: {{fx: {K[0, 0]:.2f}, fy: {K[1, 1]:.2f}, cx: {K[0, 2]:.2f}, cy: {K[1, 2]:.2f}, "
          f"dist: {_yaml_list(dist.ravel()[:5])}, width: {shape[0]}, height: {shape[1]}}}")
    return 0


# ---------------------------------------------------------------------------
# mount (hand-eye)
# ---------------------------------------------------------------------------
def _solve_and_print(kin, views, objp, K, dist) -> int:
    import cv2

    from so101_sim.kinematics import matrix_to_rpy, solve_mount

    grip, board = [], []
    for v in views:
        ok, rvec, tvec = cv2.solvePnP(objp, np.asarray(v["corners"], np.float32), K, dist)
        if not ok:
            continue
        grip.append(kin.gripper_pose(v["joints"]))
        board.append((cv2.Rodrigues(rvec)[0], tvec.ravel()))
    if len(grip) < 3:
        print("need at least 3 usable views")
        return 1
    R, t, spread = solve_mount(grip, board)
    R0, t0 = kin.default_mount
    angle = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
    print(f"\n{len(grip)} views; the board's position agrees to {spread * 1000:.1f} mm across them "
          "(under ~5 mm is good - otherwise add views from more varied angles)")
    print(f"vs the twin's camera model: {np.linalg.norm(t - t0) * 1000:.1f} mm, {angle:.1f} deg\n")
    print("cameras:\n  arm:")
    print(f"    mount: {{pos: {_yaml_list(t, '{:.4f}')}, rpy: {_yaml_list(matrix_to_rpy(R), '{:.2f}')}}}")
    return 0


def cmd_mount(args) -> int:
    import cv2

    from so101_sim.kinematics import shared_kinematics

    cfg = _config(args)
    size, objp = _board(args.board, args.square_mm)
    kin = shared_kinematics()
    if args.replay:
        data = json.loads(Path(args.replay).read_text())
        K, dist = np.asarray(data["K"]), (np.asarray(data["dist"]) if data["dist"] else None)
        return _solve_and_print(kin, data["views"], objp, K, dist)

    from vla_pipeline.robot.arm_interface import make_arm
    from vla_pipeline.vision.camera import make_camera

    cam_cfg = cfg["cameras"]["arm"]
    if not (cam_cfg.get("intrinsics") or {}).get("fx"):
        print("cameras.arm.intrinsics is not set - run `intrinsics` first (using the twin's model for now)")
    arm, cam = make_arm(cfg), make_camera(cfg, "arm")
    views, step = [], 3.0
    save = Path(args.save or "calibration_mount_views.json")
    try:
        arm.go_to({**cfg["behaviors"]["pick_place"]["scan_pose"]}, duration_s=2.5)
        K = dist = None
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            if K is None:
                K, dist = kin.intrinsics(cam_cfg.get("intrinsics"), frame.shape[1], frame.shape[0])
            corners = _find(frame, size)
            key = _show(frame, corners, size, [
                f"views {len(views)} (8+ recommended)  board {'found' if corners is not None else '-'}  step {step:g} deg",
                "a/d w/s i/k j/l u/o jog  [ ] step  SPACE capture  ENTER solve  q quit",
            ])
            if key == ord("q"):
                return 1
            if key in JOG:
                joint, sign = JOG[key]
                arm.go_to({joint: arm.get_joints()[joint] + sign * step}, duration_s=0.3)
            elif key == ord("["):
                step = max(0.5, step / 2)
            elif key == ord("]"):
                step = min(12.0, step * 2)
            elif key == ord(" "):
                time.sleep(0.5)  # let the arm settle before reading the joints
                cam.read()
                ok, frame = cam.read()
                corners = _find(frame, size) if ok else None
                if corners is None:
                    print("board not found - not captured")
                    continue
                views.append({"joints": arm.get_joints(), "corners": corners.reshape(-1, 2).tolist()})
                save.write_text(json.dumps({"K": K.tolist(), "dist": None if dist is None else dist.tolist(),
                                            "views": views}, indent=1))
                print(f"captured view {len(views)} (saved to {save})")
            elif key in ENTER and len(views) >= 3:
                break
        return _solve_and_print(kin, views, objp, K, dist)
    finally:
        cam.close()
        arm.go_to_rest()
        arm.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    from so101_sim.kinematics import ROUND_CLASSES, shared_kinematics
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, make_arm
    from vla_pipeline.vision.camera import make_camera
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    cfg = _config(args)
    pp = cfg["behaviors"]["pick_place"]
    cam_cfg = cfg["cameras"]["arm"]
    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    detector = ObjectDetector(cfg, device=args.device)
    arm, cam = make_arm(cfg), make_camera(cfg, "arm")
    try:
        arm.go_to({**pp["scan_pose"], "shoulder_pan": args.pan}, duration_s=2.5)
        time.sleep(0.5)
        cam.read()
        joints = arm.get_joints()
        dets, resolved = [], None
        for _ in range(20):
            ok, frame = cam.read()
            if ok:
                dets, resolved = detector.find(frame, args.object)
                if dets or resolved is None:
                    break
            time.sleep(0.1)
        if resolved is None:
            print(f"YOLO has no class for {args.object!r}")
            return 1
        if not dets:
            print(f"no {resolved} in view at pan {args.pan:g} - try --pan")
            return 1
        h, w = frame.shape[:2]
        K, dist = kin.intrinsics(cam_cfg.get("intrinsics"), w, h)
        table_z = float(pp.get("table_z", -0.005))
        loc = kin.locate_on_table(dets[0].box, joints, K, dist, cam_cfg.get("mount"), table_z, (w, h),
                                  shape="sphere" if resolved in ROUND_CLASSES else "upright")
        source = "calibrated" if (cam_cfg.get("intrinsics") or {}).get("fx") else "twin camera model"
        if loc is None:
            print(f"{resolved} box {np.round(dets[0].box).tolist()} touches the image border - try --pan")
            return 1
        x, y, z = loc["xyz"]
        print(f"{resolved}: x {x:.3f}  y {y:.3f}  z {z:.3f} m, {loc['width'] * 1000:.0f} mm wide ({source})")
        if args.at:
            err = np.hypot(x - args.at[0], y - args.at[1])
            print(f"measured x {args.at[0]:.3f}  y {args.at[1]:.3f}: off by {err * 1000:.1f} mm "
                  f"({'good' if err < 0.01 else 'recalibrate the mount' if err > 0.02 else 'marginal'})")
        reach = kin.solve_grasp(loc["xyz"], above_m=float(pp.get("pregrasp_above_m", 0.06)),
                                width=loc["width"], table_z=table_z)
        print("IK reach:", "solved" if reach else "out of reach")
        return 0
    finally:
        cam.close()
        arm.go_to_rest()
        arm.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="pipeline config (default config/pipeline.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("intrinsics", help="checkerboard -> cameras.arm.intrinsics")
    p.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    p.add_argument("--square-mm", type=float, default=24.0)
    p.add_argument("--min-views", type=int, default=12)
    p.add_argument("--images", help="calibrate from saved PNG/JPG views instead of the camera")
    p.add_argument("--save", help="directory to keep the captured views")
    p.set_defaults(fn=cmd_intrinsics)

    p = sub.add_parser("mount", help="board on the table + arm poses -> cameras.arm.mount")
    p.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    p.add_argument("--square-mm", type=float, default=24.0)
    p.add_argument("--save", help="captured views JSON (default calibration_mount_views.json)")
    p.add_argument("--replay", help="solve from a saved views JSON, no hardware")
    p.set_defaults(fn=cmd_mount)

    p = sub.add_parser("check", help="locate an object and compare with a measured spot")
    p.add_argument("--object", default="sports_ball")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), help="measured position (m, base frame)")
    p.add_argument("--pan", type=float, default=0.0, help="shoulder_pan for the look (deg)")
    p.add_argument("--device", choices=["npu", "cpu"], default=None)
    p.add_argument("--sim", action="store_true", help="run on the digital twin")
    p.set_defaults(fn=cmd_check)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
