# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Full-speed motion test for the SO-101 arm over the ROS 2 nodes.

Drives **each joint end to end** - from its calibrated minimum to its
calibrated maximum - **at full speed**. The arm always starts and finishes at
the physical rest pose (``REST_POSE``), and returns to rest between sweeps. This
is a simple node check that every ROS 2 joint node commands its servo through
its whole range - no pass/fail assertions, just visible motion.

    client.move_to() → arm_controller → topic_based_ros2_control
        → /so101/hardware/joint_commands → So101ServerNode → Feetech bus → servos

Sweep order
-----------
1. **paired proximal move (coordinated)** - shoulder_lift and elbow_flex move
   TOGETHER away from rest (shoulder_lift to 50% of its travel, elbow_flex to
   100%) and return together. The move is one trajectory goal, so both joints
   finish at the same instant and elbow_flex is always at 2× shoulder_lift's
   progress.
2. **every remaining joint, one at a time** - shoulder_pan, wrist_flex,
   wrist_roll, gripper - each swept end to end on its own, returning to its rest
   before the next.

"End of range" is the end *away from* rest: if a joint's rest angle is negative
it sweeps to its max, otherwise to its min.

Start pose
----------
The arm starts from ``REST_POSE`` - its safe parked pose. We do NOT start at the
calibration-derived midpoint.

Range
-----
Range comes from the LeRobot calibration file (``calibration_limits_and_rest``),
so each joint sweeps exactly as far as it was calibrated to travel. The hard
``JOINT_LIMITS`` in ``arm_interface`` still clamp every command as a backstop.

Full speed
----------
Each move's duration is the servo profile planner's (``vla_pipeline.robot.servo_profile``)
for the servos' full speed and acceleration; the trajectory controller
interpolates it and reports when the joints got there.

Calibration
-----------
Before any motion the test checks for a LeRobot calibration file. If one exists
it asks whether to recalibrate (default: keep it); if none exists, calibration
is required and runs first. See ``--recalibrate`` / ``--skip-calibration``.

By default the test spins the bus-owning server node **in-process** and starts
its controllers (``launch/so101_control.launch.py``). If the server and its
controllers already run elsewhere (``ros2 launch launch/strix_vla.launch.py``),
pass ``--external-server`` to attach to them instead.

Usage:

    python -m vla_pipeline.robot.test_ros2_arm_motion
    python -m vla_pipeline.robot.test_ros2_arm_motion --yes
    python -m vla_pipeline.robot.test_ros2_arm_motion --external-server

SAFETY: this moves a real arm at full speed. Keep the workspace clear, keep a
hand near the power switch. The test always parks at REST_POSE on exit
(including on Ctrl-C).
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from vla_pipeline.robot.arm_interface import (
    GRIPPER_CLOSED_FLOOR,
    JOINT_LIMITS,
    MOTOR_NAMES,
    REST_POSE,
    _GRIPPER_NAME,
    calibrate_arm,
    calibration_exists,
    calibration_limits_and_rest,
    calibration_path,
)
from vla_pipeline.robot.robot_node import (
    Ros2ArmClient,
    So101ServerNode,
    _require_ros2,
)
from vla_pipeline.utils.config import DEFAULT_CONFIG, load_config

logger = logging.getLogger("test_ros2_arm_motion")

# Choreography:
#   1. shoulder_lift and elbow_flex move TOGETHER away from rest - shoulder_lift
#      to 50% of its travel, elbow_flex to 100% - then return to rest together,
#   2. every remaining joint is then swept end to end, one at a time.
# The two big proximal joints are exercised as a paired move first; the distal
# joints are tested individually afterward.
PAIRED_JOINTS = ("shoulder_lift", "elbow_flex")

# Fraction of the rest → far-end travel each paired joint moves to.
# PAIRED_FRACTION is the default; PAIRED_FRACTIONS overrides it per joint.
PAIRED_FRACTION = 0.5
PAIRED_FRACTIONS = {"shoulder_lift": 0.5, "elbow_flex": 1.0}

# Back the sweep extremes off the calibrated limits by this much so the arm
# stops just short of its hard stops instead of straining against them. The
# servo's own steady-state error already eats ~1-2° at a range edge; this keeps
# it from sitting there pushing (which is what heats a gravity-loaded joint).
# Units are degrees for the arm joints and the same magnitude for the gripper's
# 0-100 scale (a few % off each end).
SWEEP_MARGIN_DEG = 3.0


def _sweep_range(
    joint: str, limits: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Reachable (lo, hi) for ``joint``: calibration range ∩ hard JOINT_LIMITS,
    inset by SWEEP_MARGIN_DEG so the sweep stops short of the hard stops."""
    cal_lo, cal_hi = limits.get(joint, JOINT_LIMITS.get(joint, (-90.0, 90.0)))
    hard_lo, hard_hi = JOINT_LIMITS.get(joint, (-180.0, 180.0))
    lo, hi = max(cal_lo, hard_lo), min(cal_hi, hard_hi)
    # Inset both ends, but never past each other (collapse to the midpoint for a
    # range narrower than 2× the margin).
    mid = (lo + hi) / 2.0
    swept_lo, swept_hi = (
        min(lo + SWEEP_MARGIN_DEG, mid),
        max(hi - SWEEP_MARGIN_DEG, mid),
    )
    # Keep the gripper clear of its closed stop. Its calibrated low end sits at the
    # mechanical stop; commanding there stalls the servo (suspect for id=6 dropping
    # off the bus), so floor its low end a few % open. Body joints are untouched.
    if joint == _GRIPPER_NAME:
        swept_lo = min(max(swept_lo, GRIPPER_CLOSED_FLOOR), swept_hi)
    return swept_lo, swept_hi


def _far_end(joint: str, limits: dict[str, tuple[float, float]]) -> float:
    """The end of ``joint``'s range that is *away from* its rest position.

    Rule: if rest is negative the far end is the maximum (hi); if rest is
    positive (or zero) the far end is the minimum (lo). So a joint resting near
    one extreme sweeps out toward the opposite extreme.
    """
    lo, hi = _sweep_range(joint, limits)
    return hi if REST_POSE.get(joint, 0.0) < 0 else lo


def _partial(joint: str, frac: float, limits: dict[str, tuple[float, float]]) -> float:
    """Angle ``frac`` of the way from ``joint``'s rest pose toward its far end."""
    rest = REST_POSE.get(joint, 0.0)
    return rest + frac * (_far_end(joint, limits) - rest)


def _move(arm: Ros2ArmClient, targets: dict[str, float]) -> None:
    """Drive one or more joints to absolute targets in one trajectory goal."""
    how = arm.move_to(targets)
    if how != "arrived":
        logger.warning("  the move to %s ended %s", targets, how)


def run_motion_test(arm: Ros2ArmClient, limits: dict[str, tuple[float, float]] | None = None) -> None:
    """Paired proximal move, then each remaining joint one by one.

    The arm always starts and finishes at ``REST_POSE``.

        1. ``PAIRED_JOINTS`` (shoulder_lift, elbow_flex) move TOGETHER away from
           rest - shoulder_lift to 50% of its travel, elbow_flex to 100% - and
           then return to rest together,
        2. every remaining joint is swept end to end, one at a time, returning to
           its own rest after each.

    The paired move is one trajectory goal: the controller interpolates both
    joints over the same duration, so they finish at the same instant (elbow
    twice the travel → always at 2× the shoulder's progress).

    Full speed throughout. ``limits`` come from the calibration file and define
    how far each joint sweeps; missing entries fall back to ``JOINT_LIMITS``.
    """
    limits = limits or dict(JOINT_LIMITS)
    home = dict(REST_POSE)

    logger.info("Homing to rest pose at full speed...")
    _move(arm, home)

    # 1) Move the paired joints out, then back, in one goal each way.
    out = {
        j: _partial(j, PAIRED_FRACTIONS.get(j, PAIRED_FRACTION), limits)
        for j in PAIRED_JOINTS
    }
    back = {j: REST_POSE.get(j, 0.0) for j in PAIRED_JOINTS}
    detail = ", ".join(
        "%s → %.1f° (%.0f%%)"
        % (j, out[j], PAIRED_FRACTIONS.get(j, PAIRED_FRACTION) * 100)
        for j in PAIRED_JOINTS
    )

    logger.info("Paired move (coordinated, arrive together): %s", detail)
    _move(arm, out)
    logger.info("  returning %s to rest (coordinated)...", " + ".join(PAIRED_JOINTS))
    _move(arm, back)

    # 2) Sweep every remaining joint end to end, one at a time.
    singles = [j for j in MOTOR_NAMES if j not in PAIRED_JOINTS]
    for joint in singles:
        lo, hi = _sweep_range(joint, limits)
        rest_j = REST_POSE.get(joint, 0.0)
        logger.info("Sweeping %-13s  rest → %.1f° → %.1f° → rest", joint, lo, hi)
        _move(arm, {joint: lo})  # drive this one joint to one end
        _move(arm, {joint: hi})  # then all the way to the other end
        logger.info("  returning %s to rest before next joint...", joint)
        _move(arm, {joint: rest_j})

    _move(arm, home)  # settle whole arm at rest

    logger.info(
        "Done - %s moved together out and back, then %d joints one by one.",
        " + ".join(PAIRED_JOINTS),
        len(singles),
    )


def _start_stack(cfg: dict, config_path):
    """A real-hardware So101ServerNode on the hardware topics, on a background
    executor, and its controllers. Returns ``(node, executor, launch)``."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, start_controllers

    if not rclpy.ok():
        rclpy.init()
    robot = {**cfg["robot"], "command_topic": HARDWARE_COMMANDS, "state_topic": HARDWARE_STATES}
    node = So101ServerNode({**cfg, "robot": robot}, dry_run=False)
    # One thread: rclpy's MultiThreadedExecutor rebuilds its wait set in a
    # tight loop under the controllers' 60 Hz setpoints (see robot_node).
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    return node, executor, start_controllers(config_path, "pipeline")


def _ensure_calibration(
    cfg: dict, recalibrate: bool = False, skip: bool = False
) -> bool:
    """Calibration gate run before any motion.

    Returns False if calibration was needed but couldn't be completed.
    """
    robot_id = cfg["robot"]["robot_id"]
    have = calibration_exists(robot_id)

    if have:
        path = calibration_path(robot_id)
        if skip:
            logger.info("Using existing calibration: %s", path)
            return True
        if recalibrate:
            logger.info(
                "Recalibration forced - redoing calibration for '%s'.", robot_id
            )
            calibrate_arm(cfg, force=True)
            return True
        print(f"\n  A calibration file already exists for '{robot_id}':\n    {path}")
        ans = input("  Recalibrate before testing? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            calibrate_arm(cfg, force=True)
        else:
            logger.info("Keeping existing calibration.")
        return True

    if skip:
        logger.warning(
            "No calibration file for '%s' and --skip-calibration set - "
            "joint angles may be meaningless. Proceeding anyway.",
            robot_id,
        )
        return True
    if not sys.stdin.isatty():
        logger.error(
            "No calibration for '%s' and no interactive terminal to run "
            "it. Calibrate first in a foreground shell:\n"
            "    python -m vla_pipeline.robot.arm_interface --calibrate",
            robot_id,
        )
        return False
    print(
        f"\n  No calibration found for '{robot_id}'. Calibration is required "
        "before moving the arm."
    )
    try:
        calibrate_arm(cfg)
    except Exception as e:
        logger.error("Calibration failed: %s", e)
        return False
    return calibration_exists(robot_id)


def main() -> None:
    """CLI entry point: calibrate if needed, confirm with the user, then run the full-speed motion sweep."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Full-speed SO-101 motion test over the ROS 2 nodes: first a coordinated shoulder_lift+elbow_flex move, then every remaining joint swept end to end one at a time."
    )
    parser.add_argument("--config", default=None, help="path to pipeline config")
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="attach to a server and controllers already running elsewhere instead of starting them",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the 'this moves real hardware' confirmation prompt",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="force recalibration even if a calibration file exists "
        "(skips the interactive prompt and just recalibrates)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="never calibrate, even if no calibration file exists "
        "(not recommended - angles may be meaningless)",
    )
    args = parser.parse_args()

    _require_ros2()
    cfg = load_config(args.config)

    if not _ensure_calibration(
        cfg, recalibrate=args.recalibrate, skip=args.skip_calibration
    ):
        sys.exit(1)

    if not args.yes:
        print("\n  WARNING: This test moves the PHYSICAL SO-101 arm at FULL SPEED.")
        print("     Clear the workspace and keep a hand near the power switch.")
        if input("     Type 'go' to continue: ").strip().lower() != "go":
            print("Aborted.")
            sys.exit(1)

    import rclpy

    from so101_sim.control import stop_controllers

    server = executor = launch = None
    arm = None
    try:
        if not args.external_server:
            logger.info("Spinning So101ServerNode in-process (real hardware) and starting its controllers...")
            server, executor, launch = _start_stack(cfg, args.config or DEFAULT_CONFIG)
        else:
            logger.info("Using an external server and controllers (strix_vla.launch.py running elsewhere).")
            if not rclpy.ok():
                rclpy.init()

        arm = Ros2ArmClient(cfg)
        limits, _rest, _meta = calibration_limits_and_rest(cfg["robot"]["robot_id"])
        if limits:
            logger.info(
                "Using calibration-derived sweep range for %d joints.", len(limits)
            )
        else:
            logger.warning(
                "No calibration limits available - falling back to "
                "hard-coded JOINT_LIMITS."
            )
        run_motion_test(arm, limits=limits or None)

    except KeyboardInterrupt:
        logger.warning("Interrupted -- parking arm and shutting down.")
    finally:
        if arm is not None:
            try:
                arm.move_to(dict(REST_POSE))
            except Exception as e:
                logger.error("Rest-park via client failed: %s", e)
            try:
                arm.close()
            except Exception as e:
                logger.error("Client close failed: %s", e)

        if launch is not None:
            try:
                stop_controllers(launch)
            except Exception as e:
                logger.error("Controller shutdown failed: %s", e)
        if server is not None:
            try:
                if executor is not None:
                    executor.shutdown()
                server.destroy_node()
            except Exception as e:
                logger.error("Server shutdown failed: %s", e)

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    print("\nFull-speed motion test complete.")


if __name__ == "__main__":
    main()
