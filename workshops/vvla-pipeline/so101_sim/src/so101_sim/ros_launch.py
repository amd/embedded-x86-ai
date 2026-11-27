# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``ros2 launch``, started so that SIGINT cannot make it hang.

    python -m so101_sim.ros_launch LAUNCH_FILE [name:=value ...]

Every launch this repository starts goes through here (:func:`python_args`).

ROS launch handles SIGINT in its event loop, through a wake-up socket. It
also leaves Python's own SIGINT handler in place, which raises
``KeyboardInterrupt`` in the main thread wherever that thread happens to be,
and it catches that exception and runs its loop again
(``launch/launch_service.py``: "KeyboardInterrupt is caught and ignored, as
signals are handled separately"). The main thread is inside the event loop.
When the exception lands after the loop has taken a callback off its ready
list and before it has run it, the callback is lost. A stopped launch was
seen in that state (2026-11-25): 28 events in its queue, its one task that
takes events off that queue still waiting for the wake-up that was lost, all
its nodes gone, and the process never exiting. It happened in 1 of 14 stops
that arrived while the launch's nodes were exiting; with Python's handler
replaced, as here, in 0 of 70.

Here Python's SIGINT handler does nothing. The signal still reaches ROS
launch's own handling, which is the one it acts on.
"""

from __future__ import annotations

import signal
import sys

MODULE = "so101_sim.ros_launch"


def python_args(launch_file, *arguments: str) -> list[str]:
    """The arguments after the Python interpreter that run ``launch_file`` with ``arguments``."""
    import importlib.util

    if importlib.util.find_spec("ros2cli") is None:
        raise RuntimeError("ROS 2 is not sourced: source /opt/ros/jazzy/setup.bash first")
    return ["-m", MODULE, str(launch_file), *arguments]


def main(argv: list[str] | None = None) -> int:
    from ros2cli.cli import main as ros2

    signal.signal(signal.SIGINT, lambda *_: None)  # no KeyboardInterrupt inside the event loop
    return ros2(argv=["launch", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
