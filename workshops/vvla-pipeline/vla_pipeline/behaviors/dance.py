# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Dance behavior: beat-synchronized keyframe choreography over the ArmClient.

A small set of named moves, each a list of joint keyframes, played at the
configured BPM. Works over ROS 2 or direct serial - and with ``--dry-run``
on machines without the arm.

Independent test
----------------
    python -m vla_pipeline.behaviors.dance --dry-run --duration 8
    python -m vla_pipeline.behaviors.dance                      # real arm
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import time

from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, make_arm
from vla_pipeline.utils.config import load_config

logger = logging.getLogger(__name__)

DANCE_MOVES = {
    "sway": [
        {
            "shoulder_pan": -45.0,
            "shoulder_lift": -60.0,
            "elbow_flex": 60.0,
            "wrist_roll": -40.0,
        },
        {
            "shoulder_pan": 45.0,
            "shoulder_lift": -60.0,
            "elbow_flex": 60.0,
            "wrist_roll": 40.0,
        },
    ],
    "pump": [
        # Folded any tighter (elbow 95, wrist 60) the wrist camera mount hits
        # the shoulder and the jaws the table, and the servos trip on overload.
        {"shoulder_lift": -90.0, "elbow_flex": 80.0, "wrist_flex": 50.0},
        {"shoulder_lift": -20.0, "elbow_flex": 20.0, "wrist_flex": -20.0},
    ],
    "disco_point": [
        {
            "shoulder_pan": -60.0,
            "shoulder_lift": -30.0,
            "elbow_flex": 10.0,
            "wrist_flex": 0.0,
        },
        {
            "shoulder_pan": 60.0,
            "shoulder_lift": -85.0,
            "elbow_flex": 80.0,
            "wrist_flex": 50.0,
        },
    ],
    "wrist_spin": [
        {"shoulder_lift": -55.0, "elbow_flex": 70.0, "wrist_roll": -90.0},
        {"shoulder_lift": -55.0, "elbow_flex": 70.0, "wrist_roll": 90.0},
    ],
    "gripper_chomp": [
        {"gripper": 80.0, "wrist_flex": 30.0},
        {"gripper": 5.0, "wrist_flex": 50.0},
    ],
}



def run_dance(
    cfg: dict,
    arm: ArmClient,
    duration_s: float | None = None,
    stop_check=None,
    seed: int | None = None,
) -> None:
    """Play a randomized sequence of dance moves at the configured BPM until duration_s elapses."""
    d = cfg["behaviors"]["dance"]
    bpm = float(d["bpm"])
    duration_s = float(d["duration_s"]) if duration_s is None else duration_s
    beat = 60.0 / bpm
    rng = random.Random(seed)

    logger.info("Dancing for %.0f s at %.0f BPM", duration_s, bpm)
    order = list(DANCE_MOVES)
    rng.shuffle(order)
    move_cycle = itertools.cycle(order)

    deadline = time.time() + duration_s
    try:
        while time.time() < deadline:
            if stop_check and stop_check():
                break
            move = next(move_cycle)
            logger.info("[dance] move: %s", move)
            # Two bars of the move (4 beats each keyframe alternation).
            for keyframe in DANCE_MOVES[move] * 4:
                if time.time() >= deadline or (stop_check and stop_check()):
                    break
                arm.go_to(keyframe, duration_s=beat * 0.9, fps=30.0, stop_check=stop_check)
    except KeyboardInterrupt:
        pass
    finally:
        if stop_check and stop_check():
            logger.info("Dance stopped - holding position.")
        else:
            logger.info("Dance over - bowing out to rest pose.")
            arm.go_to_rest(stop_check=stop_check)


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run the dance behavior standalone."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Dance behavior - component test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--duration", type=float, default=None, help="seconds (default: config)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm: ArmClient = DryRunArm() if args.dry_run else make_arm(cfg)
    try:
        run_dance(cfg, arm, duration_s=args.duration)
        print("\ndance PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
