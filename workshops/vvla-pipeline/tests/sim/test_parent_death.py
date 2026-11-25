# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``so101_sim.parent_death``: a command's processes die with the process that started it.

The commands here are stand-ins for a launch: a parent starts the wrapper,
the wrapper starts the command, and the parent is killed outright. One
command exits on SIGINT, as a launch does. The other ignores SIGINT and never
exits, as a ``ros2 launch`` was seen to do (2026-11-25; its cause is removed
in ``so101_sim/ros_launch.py``): whatever makes a command not stop, the
wrapper must kill it once ``STOP_S`` have passed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from so101_sim.parent_death import STOP_S
from twin_ros_rig import wait_exit

# The longest case waits out STOP_S once.
pytestmark = [pytest.mark.sim, pytest.mark.timeout(STOP_S + 20)]

# A command that says it is ready, with its pid, and then waits for ever.
STOPS_ON_SIGINT = "import os, signal; print(os.getpid(), flush=True); signal.pause()"
IGNORES_SIGINT = (
    "import os, signal; signal.signal(signal.SIGINT, signal.SIG_IGN); "
    "print(os.getpid(), flush=True)\nwhile True: signal.pause()"
)
# The parent: starts the wrapper on the command, passes the command's pid on, then waits to be killed.
PARENT = (
    "import subprocess, sys, signal; "
    "child = subprocess.Popen([sys.executable, '-m', 'so101_sim.parent_death', sys.executable, '-c', sys.argv[1]], "
    "stdout=subprocess.PIPE, text=True); "
    "print(child.pid, child.stdout.readline().strip(), flush=True); signal.pause()"
)


def _started(command: str):
    """``(parent, wrapper pid, command pid)`` once the command has said it is ready."""
    parent = subprocess.Popen([sys.executable, "-c", PARENT, command], stdout=subprocess.PIPE, text=True)
    wrapper, pid = (int(part) for part in parent.stdout.readline().split())
    return parent, wrapper, pid


@pytest.mark.parametrize("command", [STOPS_ON_SIGINT, IGNORES_SIGINT], ids=["stops-on-sigint", "ignores-sigint"])
def test_a_command_dies_with_its_killed_parent(command):
    parent, wrapper, pid = _started(command)
    try:
        killed = time.monotonic()
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait()
        # The kernel's exit events: the command first, then the wrapper that waited for it.
        assert wait_exit(pid), "the command outlived its parent"
        took = time.monotonic() - killed
        assert wait_exit(wrapper), "the wrapper outlived its parent"
    finally:
        for left in (pid, wrapper):
            try:
                os.kill(left, signal.SIGKILL)  # never leave one behind, whatever the result
            except ProcessLookupError:
                pass
    print(f"\nthe command was gone {took:.2f} s after its parent was killed")
    if command is IGNORES_SIGINT:
        assert took >= STOP_S, f"a command that had not exited was killed after {took:.2f} s, before STOP_S"
    else:
        assert took < STOP_S, f"a command that exits on SIGINT was only gone after {took:.2f} s"
