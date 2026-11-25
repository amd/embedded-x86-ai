# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""run_pipeline.sh starts the cameras and the arm as two ``ros2 launch`` runs.

One launch ends everything in it when one of its parts raises. A missing xacro
in the arm's part therefore took the camera nodes down with it. The arm is
made to fail here with a stand-in ``xacro`` package that raises on import; the
real ``strix_vla.launch.py`` runs unchanged.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys

import pytest

from notebook_harness import PIPE_ROOT
from so101_sim.control import BRING_UP_S
from twin_ros_rig import TEST_DOMAIN_ID

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.timeout(45)]
pexpect = pytest.importorskip("pexpect")
popen_spawn = pytest.importorskip("pexpect.popen_spawn")

# Cold start of a camera node and of a launch. The passing tests on this
# machine finished in 5.61 s, so these bound the wait for that event.
CAMERA_START_S = 15.0
LAUNCH_EXIT_S = 20.0
STOP_S = 10.0
BROKEN_XACRO = 'raise ImportError("simulated missing xacro (test_launch_split)")\n'


@pytest.fixture
def broken_xacro(tmp_path):
    """A ``xacro`` package on PYTHONPATH whose import fails, as a missing one does."""
    pkg = tmp_path / "broken_xacro" / "xacro"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(BROKEN_XACRO)
    return tmp_path / "broken_xacro"


MARKER_VAR = "LAUNCH_SPLIT_TEST_MARKER"


def _env(broken_path=None, marker=None):
    env = {**os.environ, "ROS_DOMAIN_ID": str(TEST_DOMAIN_ID), "PYTHONUNBUFFERED": "1"}
    if broken_path is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(broken_path), os.environ.get("PYTHONPATH")]))
    if marker is not None:
        env[MARKER_VAR] = marker  # every process the launch starts inherits it
    return env


def _processes_marked(marker: str) -> dict[int, str]:
    """``{pid: command line}`` of the live processes started with ``marker`` in their environment."""
    found = {}
    needle = f"{MARKER_VAR}={marker}".encode()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            with open(f"{entry.path}/environ", "rb") as f:
                if needle not in f.read().split(b"\0"):
                    continue
            with open(f"{entry.path}/cmdline", "rb") as f:
                found[int(entry.name)] = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue  # the process ended while it was read
    return found


def _launch(arm: bool, cameras: bool, log_path, broken_path=None, marker=None):
    """One ``so101_sim.stack`` launch, as ``run_pipeline.sh`` starts it."""
    cmd = [sys.executable, "-m", "so101_sim.stack",
           "--arm", str(arm).lower(), "--cameras", str(cameras).lower(), "--dry-run", "true"]
    child = popen_spawn.PopenSpawn(
        cmd, cwd=str(PIPE_ROOT), env=_env(broken_path, marker), encoding="utf-8",
        codec_errors="replace", timeout=None,
    )
    child.logfile_read = open(log_path, "w")
    return child


def _stop(child) -> int:
    """SIGINT the launch's process group, as Ctrl-C does, and return its exit status."""
    if child.proc.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
    try:
        rc = child.proc.wait(timeout=STOP_S)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.proc.wait()
        pytest.fail(f"launch did not stop within {STOP_S:.0f} s of SIGINT (log: {child.logfile_read.name})")
    # The process has exited. Read the rest of its output before the log is judged.
    if not child.flag_eof:
        child.expect(pexpect.EOF, timeout=STOP_S)
    return rc


def test_an_arm_launch_with_a_missing_xacro_exits_with_an_error(tmp_path, broken_xacro):
    log = tmp_path / "arm.log"
    arm = _launch(arm=True, cameras=False, log_path=log, broken_path=broken_xacro)
    try:
        arm.expect(pexpect.EOF, timeout=LAUNCH_EXIT_S)
    finally:
        rc = _stop(arm)
    assert rc != 0, f"the arm launch should fail, not exit {rc} (log: {log})"
    assert "simulated missing xacro" in log.read_text(), f"the failure is not in the log: {log}"


def test_a_failing_arm_launch_leaves_no_arm_server_running(tmp_path, broken_xacro):
    """The launch ends, and with it everything it started: a server left behind would
    keep the serial bus, and the next start could not connect to the arm."""
    log = tmp_path / "arm.log"
    marker = f"{os.getpid()}-{tmp_path.name}"
    arm = _launch(arm=True, cameras=False, log_path=log, broken_path=broken_xacro, marker=marker)
    left = {}
    try:
        arm.expect(pexpect.EOF, timeout=LAUNCH_EXIT_S)
        rc = _stop(arm)
        left = _processes_marked(marker)
    finally:
        for pid in _processes_marked(marker):
            os.kill(pid, signal.SIGKILL)  # never leave one behind, whatever the result
    # The launch must have failed the way this test makes it fail, not for another reason.
    assert rc != 0 and "simulated missing xacro" in log.read_text(), f"the arm launch did not fail on the missing xacro (log: {log})"
    assert not left, f"the failed arm launch left processes running: {left} (log: {log})"


def test_the_pipeline_launch_is_not_started_with_sigint_ignored():
    """bash ignores SIGINT on a background job. ``so101_sim.stack`` clears that
    before exec, which is how ``run_pipeline.sh`` starts the launch."""
    script = (PIPE_ROOT / "run_pipeline.sh").read_text()
    assert "python -m so101_sim.stack" in script
    # The reporter is the background job. It runs the same reset the launch runs,
    # then reads the signal mask the launch would inherit. wait returns its status.
    reporter = (
        "from so101_sim.stack import accept_stop_signals; "
        "accept_stop_signals(); "
        "import sys; "
        "mask = int(open('/proc/self/status').read().split('SigIgn:\\t')[1].split()[0], 16); "
        "sys.exit(0 if not mask & 2 else 3)"
    )
    proc = subprocess.run(
        ["bash", "-c", f"{sys.executable} -c {shlex.quote(reporter)} & wait $!"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_an_arm_launch_brings_up_the_controllers_on_the_arm_server(tmp_path):
    """The controllers start before the arm server. Their hardware side waits for the
    server's first joint state, so the arm controller still comes up active.

    ``so101_sim.controller_bringup`` says so in one line when every controller is in its
    state, then exits on its own. Ctrl-C comes after that exit."""
    log = tmp_path / "arm.log"
    marker = f"{os.getpid()}-{tmp_path.name}"
    arm = _launch(arm=True, cameras=False, log_path=log, marker=marker)
    try:
        # The stack's own bring-up budget: the server's first state, then the controllers.
        arm.expect(
            r"\[controller_bringup-\d+\][^\n]*controllers up:[^\n]*arm_controller active",
            timeout=BRING_UP_S,
        )
        arm.expect(r"\[controller_bringup-\d+\]: process has finished cleanly", timeout=BRING_UP_S)
        _stop(arm)
        arm.logfile_read.flush()
        text = log.read_text()
    finally:
        for pid in _processes_marked(marker):
            os.kill(pid, signal.SIGKILL)  # never leave one behind, whatever the result
    # SIGINT is a stop, not a crash: the server releases the bus and exits 0.
    assert "so101_server_dry" in text and "process has finished cleanly" in text, text
    assert "process has died" not in text, text


def test_a_failing_arm_launch_does_not_end_the_camera_launch(tmp_path, broken_xacro):
    cam_log, arm_log = tmp_path / "cameras.log", tmp_path / "arm.log"
    cam = _launch(arm=False, cameras=True, log_path=cam_log)
    try:
        # The event: the mount camera node has started inside the camera launch.
        cam.expect(r"\[camera_mount-\d+\]: process started", timeout=CAMERA_START_S)
        arm = _launch(arm=True, cameras=False, log_path=arm_log, broken_path=broken_xacro)
        arm.expect(pexpect.EOF, timeout=LAUNCH_EXIT_S)
        arm_rc = _stop(arm)
        assert arm_rc != 0, f"the arm launch should fail (log: {arm_log})"
        # The arm launch has exited. Its failure must not have ended the camera launch.
        assert cam.proc.poll() is None, f"the camera launch ended with the arm launch (log: {cam_log})"
    finally:
        _stop(cam)
