# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""SO-101 arm access layer.

Behaviors talk to an :class:`ArmClient`. The transport is :class:`LeRobotArm`
(the Feetech bus) or :class:`DryRunArm`. The digital twin's
:class:`~vla_pipeline.robot.sim_arm.SimArmClient` is that same transport with
a :class:`so101_sim.SimFollower` in place of the serial bus. ``Ros2ArmClient``
publishes joint commands to the ROS 2 node that owns the bus.

Joint convention (degrees, LeRobot SO-101 calibration space):
``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper``.

:data:`REST_POSE`, :data:`GRIPPER_CLOSED_FLOOR`, :data:`JOINT_LIMITS` and
:func:`clamp_joints` are defined in :mod:`common.motion`.

Independent test
----------------
    python -m vla_pipeline.robot.arm_interface --dry-run
    python -m vla_pipeline.robot.arm_interface --port /dev/ttyACM0 --id my_awesome_follower_arm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_common_path() -> None:
    """Put ``workshop/solution`` on ``sys.path`` so ``import common`` resolves."""
    root = Path(__file__).resolve().parents[2]
    sol = str(root / "workshop" / "solution")
    if sol not in sys.path:
        sys.path.append(sol)


ensure_common_path()

from common.motion import (  # noqa: E402
    DIRECT_MODE_ACCEL_RAW,
    DIRECT_MODE_SPEED_RAW,
    GRASP_LIMITS,
    GRIPPER_CLOSED_FLOOR,
    GRIPPER_TORQUE_LIMIT,
    JOINT_LIMITS,
    REST_POSE,
    ArmClient,
    DryRunArm,
    LeRobotArm,
    clamp_joints,
    restore_twin,
    sim_world,
)
from so101_sim import MOTOR_NAMES  # noqa: E402

__all__ = [
    "DIRECT_MODE_ACCEL_RAW",
    "DIRECT_MODE_SPEED_RAW",
    "GRASP_LIMITS",
    "GRIPPER_CLOSED_FLOOR",
    "GRIPPER_TORQUE_LIMIT",
    "JOINT_LIMITS",
    "MOTOR_NAMES",
    "REST_POSE",
    "ArmClient",
    "DryRunArm",
    "LeRobotArm",
    "clamp_joints",
    "ensure_common_path",
    "restore_twin",
    "sim_world",
]


def make_bus_arm(cfg: dict) -> ArmClient:
    """The bus-owning transport: the real Feetech bus, or the digital twin
    when ``robot.backend: sim``."""
    r = cfg["robot"]
    if r.get("backend", "real") == "sim":
        from vla_pipeline.robot.sim_arm import SimArmClient

        return SimArmClient(cfg)
    return LeRobotArm(r["motor_port"], r["robot_id"], r.get("max_relative_target", 8.0))


def make_arm(cfg: dict) -> ArmClient:
    """Build the ArmClient described by ``cfg['robot']``."""
    r = cfg["robot"]
    if not r.get("enable_motors", True):
        return DryRunArm()
    if r.get("use_ros2", False):
        from vla_pipeline.robot.robot_node import Ros2ArmClient

        return Ros2ArmClient(cfg)
    return make_bus_arm(cfg)


# =============================================================================
# Calibration
# =============================================================================
# LeRobot stores per-robot calibration at
#   ~/.cache/huggingface/lerobot/calibration/robots/<robot_type>/<robot_id>.json
# Calibration is INTERACTIVE - it prompts you to move the arm and press ENTER -
# so it can only run in a foreground terminal, never in the backgrounded server
# node. The launcher detects a missing file and runs calibrate_arm() first.


def calibration_path(robot_id: str) -> Path | None:
    """Return the existing calibration file for ``robot_id``, or None."""
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
    hits = sorted(base.glob(f"*/{robot_id}.json"))
    return hits[0] if hits else None


def calibration_exists(robot_id: str) -> bool:
    """Return True if a calibration file exists for robot_id."""
    return calibration_path(robot_id) is not None


# Feetech STS3215 encoder resolution (12-bit). LeRobot normalizes degrees as
# ``(raw - mid) * 360 / (resolution - 1)`` with ``mid = (range_min+range_max)/2``.
_STS3215_RESOLUTION = 4096
_DEG_PER_COUNT = 360.0 / (_STS3215_RESOLUTION - 1)

# The gripper uses 0..100 normalization; body joints run in DEGREES mode.
# See lerobot so_follower + motors_bus._normalize.
_GRIPPER_NAME = "gripper"


def load_calibration(robot_id: str) -> dict[str, dict] | None:
    """Load and parse the LeRobot calibration JSON for ``robot_id``.

    Returns a dict ``{motor_name: {id, drive_mode, homing_offset, range_min,
    range_max}}`` or ``None`` if no calibration file exists / it can't be read.
    """
    path = calibration_path(robot_id)
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read calibration %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Unexpected calibration format in %s", path)
        return None
    return data


def calibration_limits_and_rest(
    robot_id: str,
    full_range_eps: int = 3500,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], dict[str, dict]]:
    """Derive per-joint software limits and a rest pose from the calibration file.

    The SO-101 follower runs body joints in DEGREES mode: the calibrated zero is
    the **midpoint** of the recorded raw range, and the usable travel each way is
    ``±(range_max - range_min)/2`` converted to degrees. So:

        rest[joint]  = 0.0                         (midpoint maps to 0°)
        limits[joint] = (-half_span_deg, +half_span_deg)

    The gripper uses 0..100 normalization, so its rest is the 50 midpoint and its
    limits are (0, 100).

    A joint that was **not range-swept** during calibration (notably
    ``wrist_roll`` - LeRobot deliberately skips it) keeps the near-full encoder
    range, giving ≈±180°; we detect that (span > ``full_range_eps`` counts) and
    flag it via the returned ``meta`` so callers can warn that its rest/limits
    come from the default range, not a real sweep.

    Returns ``(limits, rest, meta)`` where ``meta[joint]`` is the raw calibration
    entry augmented with ``span_counts`` and ``swept`` (bool).
    """
    cal = load_calibration(robot_id)
    limits: dict[str, tuple[float, float]] = {}
    rest: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if not cal:
        return limits, rest, meta

    for name, entry in cal.items():
        try:
            rmin = float(entry["range_min"])
            rmax = float(entry["range_max"])
        except (KeyError, TypeError, ValueError):
            continue
        span_counts = abs(rmax - rmin)
        swept = span_counts <= full_range_eps  # full encoder range => not swept
        if name == _GRIPPER_NAME:
            # 0..100 normalization: midpoint is 50, range is the full [0,100].
            limits[name] = (0.0, 100.0)
            rest[name] = 50.0
        else:
            half_span_deg = 0.5 * span_counts * _DEG_PER_COUNT
            limits[name] = (-half_span_deg, half_span_deg)
            rest[name] = 0.0  # DEGREES mode: calibrated midpoint == 0°
        meta[name] = {**entry, "span_counts": span_counts, "swept": swept}
    return limits, rest, meta


def calibrate_arm(cfg: dict, force: bool = False) -> None:
    """Run LeRobot's interactive SO-101 calibration (foreground / TTY only).

    Connecting already triggers calibration when the file is missing or
    mismatched; ``force`` re-runs it even when a valid file exists. Either way
    this must run in a real terminal because it calls ``input()`` to step you
    through moving the arm.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Calibration needs an interactive terminal (it asks you to move the "
            "arm and press ENTER) but stdin is not a TTY. Run it in the "
            "foreground:\n    python -m vla_pipeline.robot.arm_interface --calibrate"
        )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    r = cfg["robot"]
    existing = calibration_path(r["robot_id"])
    if existing and not force:
        logger.info(
            "Calibration already exists at %s - nothing to do "
            "(use --calibrate --force to redo).",
            existing,
        )
        return

    config = SO101FollowerConfig(
        port=r["motor_port"],
        id=r["robot_id"],
        cameras={},
        max_relative_target=r.get("max_relative_target", 8.0),
    )
    robot = SO101Follower(config)
    print(f"\nCalibrating SO-101 '{r['robot_id']}' on {r['motor_port']}.")
    print("Follow the prompts: move the arm as instructed and press ENTER.\n")
    robot.connect()  # auto-calibrates when the file is missing/mismatched
    if force:
        robot.calibrate()  # explicit recalibration even if a file existed
    robot.disconnect()
    saved = calibration_path(r["robot_id"])
    print(f"\nCalibration complete: {saved}")


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: component-test the arm, show calibration, or run interactive calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 arm interface - component test / calibration"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="no hardware, just print commands"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run interactive calibration (foreground) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --calibrate: recalibrate even if a file exists",
    )
    parser.add_argument(
        "--config", default=None, help="pipeline.yaml (for --calibrate)"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="print calibration-derived limits/rest for --id and exit",
    )
    args = parser.parse_args()

    if args.show_calibration:
        limits, rest, meta = calibration_limits_and_rest(args.id)
        if not limits:
            print(f"No calibration found for '{args.id}'.")
            return
        print(f"Calibration-derived limits/rest for '{args.id}':")
        for m in MOTOR_NAMES:
            if m in limits:
                lo, hi = limits[m]
                swept = meta[m].get("swept", True)
                tag = "" if swept else "  (NOT range-swept - default range)"
                print(
                    f"  {m:<14} limits=({lo:7.1f}, {hi:7.1f})  rest={rest[m]:6.1f}{tag}"
                )
        return

    if args.calibrate:
        from vla_pipeline.utils.config import load_config

        calibrate_arm(load_config(args.config), force=args.force)
        return

    arm: ArmClient = DryRunArm() if args.dry_run else LeRobotArm(args.port, args.id)
    try:
        print("Current joints:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Wiggling gripper (open → close → open)...")
        arm.go_to({"gripper": 60.0}, duration_s=1.5)
        arm.go_to({"gripper": 5.0}, duration_s=1.5)
        arm.go_to({"gripper": 30.0}, duration_s=1.0)
        print("\nReturning to rest pose...")
        arm.go_to_rest()
        print("Arm interface test PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
