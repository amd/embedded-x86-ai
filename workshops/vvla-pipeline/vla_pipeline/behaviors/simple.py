# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Simple one-shot behaviors: wave and home.

Both are deliberately tiny - they exist mostly as the template for adding new
voice commands (see :mod:`vla_pipeline.llm.intents`).

Independent test
----------------
    python -m vla_pipeline.behaviors.simple --behavior wave --dry-run
    python -m vla_pipeline.behaviors.simple --behavior home
"""

from __future__ import annotations

import argparse
import logging

from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, ensure_common_path, make_arm
from vla_pipeline.utils.config import load_config

ensure_common_path()

from common.motion import WAVE_LEFT as _WAVE_LEFT  # noqa: E402
from common.motion import WAVE_RIGHT as _WAVE_RIGHT, WAVE_UP as _WAVE_UP

logger = logging.getLogger(__name__)


def run_wave(cfg: dict, arm: ArmClient, stop_check=None) -> None:
    """Raise the arm and wave the wrist ``behaviors.wave.cycles`` times."""
    cycles = int(cfg["behaviors"].get("wave", {}).get("cycles", 3))
    logger.info("[wave] waving %d times", cycles)
    arm.go_to(_WAVE_UP, duration_s=2.0, stop_check=stop_check)
    for _ in range(cycles):
        if stop_check and stop_check():
            break
        arm.go_to(_WAVE_LEFT, duration_s=0.4, stop_check=stop_check)
        arm.go_to(_WAVE_RIGHT, duration_s=0.4, stop_check=stop_check)
    arm.go_to_rest()


def run_home(cfg: dict, arm: ArmClient, stop_check=None) -> None:
    """Return to the rest pose. A locked grip stays locked: "release" opens
    it, and gesture-mimic hands the gripper to the pinch."""
    logger.info("[home] returning to rest pose")
    arm.go_to_rest()


def main() -> None:
    """CLI entry point: run the wave or home behavior standalone."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Simple behaviors - component test")
    parser.add_argument("--behavior", choices=["wave", "home"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm: ArmClient = DryRunArm() if args.dry_run else make_arm(cfg)
    try:
        {"wave": run_wave, "home": run_home}[args.behavior](cfg, arm)
        print(f"\n{args.behavior} PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
