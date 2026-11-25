# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``so101_sim.ros_launch``: how this repository starts a ROS launch.

A launch started with Python's own SIGINT handler in place gets a
``KeyboardInterrupt`` inside its event loop on every stop, swallows it, and
can lose the callback the loop was about to run: such a launch was seen never
to exit (2026-11-25, 1 of 14 stops that came while its nodes were exiting; 0
of 70 once that handler was replaced). What is tested here is the state that
rules it out, and one stop at the worst moment found: inside the launch
process, Python's SIGINT handler is not the one that raises, and a launch
sent SIGINT while it is still starting stops and exits 0.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from so101_sim.parent_death import STOP_S
from so101_sim.ros_launch import MODULE, python_args

# A launch of one plain process stopped within 0.1 s (2026-11-25); STOP_S is the stop's own bound.
pytestmark = [pytest.mark.sim, pytest.mark.timeout(STOP_S + 20)]

REPORT = "python's SIGINT handler raises KeyboardInterrupt:"
# Runs inside the launch process (an OpaqueFunction), beside one process that stays until stopped.
LAUNCH_FILE = f'''
import signal, sys
from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction


def report(context):
    print("{REPORT}", signal.getsignal(signal.SIGINT) is signal.default_int_handler, flush=True)


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=[sys.executable, "-c", "import signal; signal.pause()"], name="stays", output="screen"),
        OpaqueFunction(function=report),
    ])
'''


def _handler_report(command: list[str], stop: bool) -> tuple[str, int | None, str]:
    """Run ``command`` and read its report line. With ``stop``, send it one SIGINT, at once,
    while it is still starting, and wait for it: ``(report, exit status, the rest of its output)``.
    Without, the launch is killed: ``(report, None, "")``."""
    launch = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True)
    try:
        report = ""
        for line in launch.stdout:  # the launch's own lines, as they come
            if REPORT in line:
                report = line.strip()
                break
        assert report, "the launch ended without reporting its SIGINT handler"
        if not stop:
            return report, None, ""
        os.kill(launch.pid, signal.SIGINT)
        rest, _ = launch.communicate(timeout=STOP_S)
        return report, launch.returncode, rest
    finally:
        if launch.poll() is None:
            os.killpg(launch.pid, signal.SIGKILL)
            launch.wait()


def test_a_launch_started_here_has_no_raising_sigint_handler_and_still_stops(tmp_path):
    launch_file = tmp_path / "one_process.launch.py"
    launch_file.write_text(LAUNCH_FILE)
    report, status, rest = _handler_report([sys.executable, *python_args(launch_file)], stop=True)
    assert report == f"{REPORT} False", report
    # ROS launch's own handling took the signal: it says so, stops its process, and exits.
    assert "user interrupted with ctrl-c (SIGINT)" in rest, rest
    assert status == 0, f"the launch exited {status}:\n{rest}"


def test_the_plain_ros2_launch_command_has_the_raising_handler(tmp_path):
    """The premise: started as ``ros2 launch``, the launch process does have that handler.

    Sent one SIGINT at the moment the first test sends it, such a launch did not exit in 12 of
    12 tries, and one started through ``so101_sim.ros_launch`` exited in 12 of 12 (2026-11-25).
    That is ROS's behaviour and not asserted here: this launch is not stopped, it is killed."""
    import shutil

    launch_file = tmp_path / "one_process.launch.py"
    launch_file.write_text(LAUNCH_FILE)
    command = [sys.executable, shutil.which("ros2"), "launch", str(launch_file)]
    report, _status, _rest = _handler_report(command, stop=False)
    assert report == f"{REPORT} True", report


def test_every_launch_of_the_stacks_goes_through_it():
    from so101_sim import stack

    command = stack.launch_argv(arm=True, cameras=False, digital_twin=True, dry_run=False)
    assert command[:3] == [sys.executable, "-m", MODULE], command
    for path in ("so101_sim/src/so101_sim/control.py", "workshop/solution/ros/node_supervisor.py",
                 "workshop/project/ros/node_supervisor.py"):
        from notebook_harness import PIPE_ROOT

        text = (PIPE_ROOT / path).read_text()
        assert "python_args(LAUNCH_FILE" in text, f"{path} does not start its launch through so101_sim.ros_launch"
