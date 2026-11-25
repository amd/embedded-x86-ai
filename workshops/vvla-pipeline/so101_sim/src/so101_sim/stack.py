# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The ROS graph ``run_pipeline.sh`` starts. The notebook starts this same launch.

``python -m so101_sim.stack`` resets the ignored SIGINT a background shell job
inherits, then replaces itself with ``ros2 launch launch/strix_vla.launch.py``,
started through :mod:`so101_sim.ros_launch`.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
from pathlib import Path

from so101_sim.ros_launch import python_args

REPO = Path(__file__).resolve().parents[3]
LAUNCH = REPO / "launch" / "strix_vla.launch.py"


def command(*, arm: bool, cameras: bool, digital_twin: bool, dry_run: bool) -> list[str]:
    """The ``python -m so101_sim.stack`` command ``run_pipeline.sh`` runs."""
    return [
        sys.executable, "-m", "so101_sim.stack",
        "--arm", str(arm).lower(),
        "--cameras", str(cameras).lower(),
        "--digital-twin", str(digital_twin).lower(),
        "--dry-run", str(dry_run).lower(),
    ]


def launch_argv(
    *,
    arm: bool,
    cameras: bool,
    digital_twin: bool,
    dry_run: bool,
    config: str | None = None,
) -> list[str]:
    """The launch's command: ``ros2 launch``, through :mod:`so101_sim.ros_launch`."""
    arguments = [
        f"arm:={str(arm).lower()}",
        f"cameras:={str(cameras).lower()}",
        f"digital_twin:={str(digital_twin).lower()}",
        f"dry_run:={str(dry_run).lower()}",
    ]
    if config:
        arguments.append(f"config:={config}")
    return [sys.executable, *python_args(LAUNCH, *arguments)]


def accept_stop_signals() -> None:
    """Leave a background shell job able to stop on SIGINT.

    ``run_pipeline.sh`` starts this module with ``&``. A non-interactive bash
    does not give that job its own process group, and it inherits SIGINT
    ignored. ``signal.signal`` cannot clear an inherited ignore. The cleanup
    in the script signals the group (``kill -INT -- -$pid``), so this process
    has to lead the group and accept the signal before it becomes the launch.
    """
    if os.getpgrp() != os.getpid():
        os.setsid()
    libc = ctypes.CDLL(None)
    libc.signal(signal.SIGINT, 0)
    libc.signal(signal.SIGQUIT, 0)


def exec_launch(**kwargs) -> None:
    """Replace this process with the launch. The process dies with its parent."""
    from so101_sim.parent_death import die_with_parent

    die_with_parent()
    argv = launch_argv(**kwargs)
    accept_stop_signals()
    os.execvp(argv[0], argv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start launch/strix_vla.launch.py")
    parser.add_argument("--arm", required=True, choices=("true", "false"))
    parser.add_argument("--cameras", required=True, choices=("true", "false"))
    parser.add_argument("--digital-twin", default="false", choices=("true", "false"))
    parser.add_argument("--dry-run", default="false", choices=("true", "false"))
    parser.add_argument("--config", default=None)
    flag = {"true": True, "false": False}
    args = parser.parse_args()
    exec_launch(
        arm=flag[args.arm],
        cameras=flag[args.cameras],
        digital_twin=flag[args.digital_twin],
        dry_run=flag[args.dry_run],
        config=args.config,
    )


if __name__ == "__main__":
    main()
