# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Twin calibration checks: does the sim match the rig?

    python -m so101_sim.calibrate poses   [--config config/pipeline.yaml] [--out poses.png]
    python -m so101_sim.calibrate grasp   [--config ...]
    python -m so101_sim.calibrate framing [--out framing.png]
    python -m so101_sim.calibrate mirror  --port /dev/ttyACM0 --id my_awesome_follower_arm

``poses``   renders every configured behavior pose (rest, scan, block, ...) in
            the 3-D view next to the wrist camera - compare against photos of
            the real arm in the same pose to validate the joint map.
``grasp``   stages an object between the jaws of each grasp pose, closes the
            gripper with the real ``max_relative_target``/profile settings and
            reports gripper current, contact and lift - the numbers the grip
            behavior's ``load_threshold`` is tuned against.
``framing`` renders the wrist camera with an object in the open jaws (the
            arm-camera view the pick behavior servos on).
``mirror``  (rig attached) reads the REAL arm's joints with torque off and
            drives the twin to the same values, side by side, so a sign or
            offset error in ``sim.joint_map`` is obvious.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

PIPELINE_POSES = {
    "rest": {},
    "scan": dict(shoulder_pan=0.0, shoulder_lift=-55.0, elbow_flex=65.0, wrist_flex=45.0, wrist_roll=0.0, gripper=70.0),
    "approach": dict(shoulder_pan=0.0, shoulder_lift=-35.0, elbow_flex=45.0, wrist_flex=35.0, wrist_roll=0.0, gripper=75.0),
    "descend": dict(shoulder_pan=0.0, shoulder_lift=-20.0, elbow_flex=45.0, wrist_flex=35.0, wrist_roll=0.0, gripper=75.0),
    "block": dict(shoulder_pan=35.0, shoulder_lift=-45.0, elbow_flex=60.0, wrist_flex=40.0, wrist_roll=0.0, gripper=70.0),
    "present": dict(shoulder_pan=0.0, shoulder_lift=-35.0, elbow_flex=30.0, wrist_flex=10.0, wrist_roll=0.0, gripper=85.0),
}

# (object, pose, open gripper value) - how each behavior grasps
GRASPS = [
    ("toothbrush", "present", 85.0),  # grip: hand a thin object to the jaws
    ("book", "block", 70.0),  # fetch_block uses the prop at its taught spot
    ("sports_ball", "descend", 75.0),  # pick_place
]


def _poses_from_config(cfg: dict | None) -> dict:
    poses = {k: dict(v) for k, v in PIPELINE_POSES.items()}
    if not cfg:
        return poses
    b = cfg.get("behaviors") or {}
    pp = b.get("pick_place") or {}
    if pp.get("scan_pose"):
        poses["scan"] = dict(pp["scan_pose"])
    if pp.get("approach_pose"):
        poses["approach"] = {**poses["approach"], **pp["approach_pose"]}
        poses["descend"] = {**poses["approach"], "shoulder_lift": float(pp.get("descend_lift", -20.0))}
    fb = b.get("fetch_block") or {}
    if fb.get("block_pose"):
        poses["block"] = {**fb["block_pose"], "gripper": 70.0}
    return poses


def _load_cfg(path):
    if not path:
        return None
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _settle(world, arm, pose, seconds=2.5):
    arm.send_joints(pose)
    if world.running:
        time.sleep(seconds)
    else:
        world.advance(seconds)


def run_grasp(cfg=None, verbose=True) -> list[dict]:
    """Stage + close + lift for each grasp; returns one result dict per grasp."""
    from so101_sim.arm import SimArm
    from so101_sim.world import SimWorld

    poses = _poses_from_config(cfg)
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    mapped = ((cfg or {}).get("sim") or {}).get("gripper_map", {}).get("closed_cal")
    floor = GRIPPER_CLOSED_FLOOR if mapped is None else float(mapped)
    results = []
    world = SimWorld(cfg, start=False)
    try:
        for obj, pname, opened in GRASPS:
            world.reset()
            arm = SimArm(world)
            pose = {k: v for k, v in poses[pname].items() if k != "gripper"}
            _settle(world, arm, {**pose, "gripper": opened})
            staged = world.stage_object(obj, pose)
            at_stage = world.contact_report(obj)
            world.advance(0.3)
            drift = float(np.linalg.norm(world.body_pos(obj) - staged))
            arm.send_joints({"gripper": floor})
            peak, t_contact = 0.0, None
            for i in range(60):
                world.advance(0.05)
                c = arm.get_gripper_load()
                peak = max(peak, c)
                if c >= 120 and t_contact is None:
                    t_contact = round(0.05 * (i + 1), 2)
            hold_cur = arm.get_gripper_load()
            grip = arm.get_joints()["gripper"]
            grasped = world.is_grasped(obj)
            world.park_stand("stand_a")
            z0 = world.body_pos(obj)[2]
            lift = {**pose, "shoulder_lift": pose["shoulder_lift"] - 30.0}
            arm.send_joints(lift)
            world.advance(1.5)
            dz = float(world.body_pos(obj)[2] - z0)
            r = dict(
                object=obj, pose=pname, drift_mm=round(1000 * drift, 1), gripper=round(grip, 1),
                hold_current=hold_cur, peak_current=peak, t_contact_s=t_contact,
                grasped=grasped, lifted_mm=round(1000 * dz, 1), held_after_lift=world.is_grasped(obj),
            )
            results.append(r)
            if verbose:
                print(
                    f"{obj:5s} @ {pname:8s} drift={r['drift_mm']:5.1f}mm gripper={r['gripper']:5.1f} "
                    f"current hold={hold_cur:4.0f} peak={peak:4.0f} (>=120 at {t_contact}s) "
                    f"grasped={grasped} lift dz={r['lifted_mm']:6.1f}mm held={r['held_after_lift']}"
                )
                if r["drift_mm"] > 5.0:
                    print(f"      staged at {np.round(staged, 3)}; contacts then: {at_stage}")
    finally:
        world.close()
    return results


def run_poses(cfg=None, out="so101_poses.png") -> str:
    import cv2

    from so101_sim.arm import SimArm
    from so101_sim.camera import SimCamera
    from so101_sim.render import RenderService
    from so101_sim.world import SimWorld

    world = SimWorld(cfg, start=False)
    rs = RenderService(world)
    view = SimCamera(world, "view", renderer=rs, width=480, height=360, blocking=True)
    wrist = SimCamera(world, "arm", renderer=rs, width=480, height=360, blocking=True)
    rows = []
    try:
        for name, pose in _poses_from_config(cfg).items():
            world.reset(pose)
            SimArm(world)
            world.advance(0.2)
            _, a = view.read(fresh=True)
            _, b = wrist.read(fresh=True)
            for img, label in ((a, f"{name} (3-D)"), (b, f"{name} (wrist cam)")):
                cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(np.hstack([a, b]))
        cv2.imwrite(out, np.vstack(rows))
    finally:
        rs.close()
        world.close()
    print(f"wrote {out}")
    return out


def run_framing(cfg=None, out="so101_framing.png") -> str:
    import cv2

    from so101_sim.arm import SimArm
    from so101_sim.camera import SimCamera
    from so101_sim.render import RenderService
    from so101_sim.world import SimWorld

    poses = _poses_from_config(cfg)
    world = SimWorld(cfg, start=False)
    rs = RenderService(world)
    wrist = SimCamera(world, "arm", renderer=rs, blocking=True)
    tiles = []
    try:
        for obj, pname, opened in GRASPS:
            pose = {k: v for k, v in poses[pname].items() if k != "gripper"}
            world.reset({**pose, "gripper": opened})
            SimArm(world)
            world.stage_object(obj, pose)
            world.advance(0.1)
            _, img = wrist.read(fresh=True)
            cv2.putText(img, f"{obj} @ {pname}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)
        cv2.imwrite(out, np.hstack(tiles))
    finally:
        rs.close()
        world.close()
    print(f"wrote {out}")
    return out


def run_mirror(cfg, port: str, robot_id: str, seconds: float = 60.0) -> None:
    """Real arm (torque off) -> twin; prints the per-joint values side by side."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from so101_sim.arm import SimArm
    from so101_sim.world import SimWorld

    real = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={}))
    real.connect()
    real.bus.disable_torque()
    world = SimWorld(cfg, start=True)
    arm = SimArm(world, speed_raw=0, accel_raw=0)
    try:
        t_end = time.time() + seconds
        while time.time() < t_end:
            obs = real.get_observation()
            joints = {k[:-4]: float(v) for k, v in obs.items() if k.endswith(".pos")}
            arm.send_joints(joints)
            sim = arm.get_joints()
            print(
                "  ".join(f"{k[:8]}={joints[k]:6.1f}/{sim[k]:6.1f}" for k in joints),
                end="\r",
                flush=True,
            )
            time.sleep(0.05)
    finally:
        real.disconnect()
        world.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 twin calibration checks")
    ap.add_argument("what", choices=["poses", "grasp", "framing", "mirror"])
    ap.add_argument("--config", default=None, help="pipeline.yaml / workshop.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="my_awesome_follower_arm")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.what == "poses":
        run_poses(cfg, args.out or "so101_poses.png")
    elif args.what == "grasp":
        run_grasp(cfg)
    elif args.what == "framing":
        run_framing(cfg, args.out or "so101_framing.png")
    else:
        run_mirror(cfg, args.port, args.id)


if __name__ == "__main__":
    main()
