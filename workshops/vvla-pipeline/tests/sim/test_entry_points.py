# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The stacks' entry points on the digital twin, driven like a person typing.

``run_pipeline.sh --digital-twin`` and the workshop's ``ros.node_supervisor``
and ``app.py --digital-twin`` start the twin and its controllers as they
start the arm on the rig. Each typed command waits for the line its behavior
ends with: the pipeline dispatcher's ``=== <intent> ended (<how>) ===``, the
workshop app's ``=== '<text>' handled ===``.

Tier 2 (ROS 2): ``./run_sim_tests.sh``. The pipeline's run needs llama-server
and the webcam, so it runs with the docs tests: ``cd workshop && ./run_notebooks.sh --test``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from notebook_harness import PIPE_ROOT
from twin_ros_rig import assert_domain_free, join_test_domain, wait_exit

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(600)]
pexpect = pytest.importorskip("pexpect")
popen_spawn = pytest.importorskip("pexpect.popen_spawn")
pytest.importorskip("rclpy")

SOLUTION = PIPE_ROOT / "workshop" / "solution"
# Both stacks were ready 3.9-5.7 s after their start; a cold llama-server load
# adds 10-30 s (run_notebooks.sh). The longest behavior, a dance, took 22.4 s.
START_S = 120.0
BEHAVIOR_S = 60.0
PIPELINE_ENDED = r"=== (\w+) ended \((ok|stopped|failed)\) ==="
WORKSHOP_HANDLED = r"=== '([^']*)' handled ==="


@pytest.fixture
def twin_domain():
    join_test_domain()
    assert_domain_free("a twin already runs on the test domain")
    yield
    assert_domain_free("a twin outlived its entry point", wait=True)


def _spawn(command: str, cwd, log_path):
    """``command`` on pipes, not a pty (a pty fork in this multi-threaded process
    may deadlock), in a session of its own, so :func:`_close` can stop all of it."""
    child = popen_spawn.PopenSpawn(
        ["setsid", "bash", "-c", command], cwd=str(cwd), env={**os.environ, "PYTHONUNBUFFERED": "1"},
        encoding="utf-8", codec_errors="replace", timeout=None,
    )
    child.logfile_read = open(log_path, "w")
    return child


def _close(child) -> int | None:
    """Let ``child`` finish and return its exit status. One that still runs gets
    SIGINT on its process group, as Ctrl-C sends it; one that outlasts that is
    killed, and fails the test."""
    if child.proc.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
        except pexpect.TIMEOUT:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise AssertionError(f"{child.proc.args} did not exit within {BEHAVIOR_S:.0f} s of SIGINT") from None
    else:
        child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    return child.wait()


@pytest.mark.docs
def test_run_pipeline_drives_the_twin(twin_domain, tmp_path):
    """Every object picked up and dropped, the scripted behaviors, a stop, and
    a clean shutdown: the launch reports no node that died."""
    log = tmp_path / "run_pipeline.log"
    child = _spawn("./run_pipeline.sh --digital-twin --text-commands --headless", PIPE_ROOT, log)
    try:
        child.expect(r"\[say\] Ready\.", timeout=START_S)
        for text, spoken in (
            ("pick up the ball", "Got the sports ball"),
            ("drop it", "Released"),
            ("pick up the bottle", "Got the bottle"),
            ("drop it", "Released"),
            ("pick up the toothbrush", "Got the toothbrush"),
            ("drop it", "Released"),
            ("wave at me", None),
            ("hold this", "No object detected"),
            ("go home", None),
            ("dance", None),
        ):
            child.sendline(text)
            child.expect(PIPELINE_ENDED, timeout=BEHAVIOR_S)
            assert child.match.group(2) == "ok", f"{text!r}: {child.match.group(1)} ended {child.match.group(2)}"
            assert spoken is None or f"[say] {spoken}" in child.before, f"{text!r} never said {spoken!r}"
        child.sendline("dance")
        child.expect(r"=== dance  ===", timeout=BEHAVIOR_S)
        child.expect(r"move_to arrived", timeout=BEHAVIOR_S)
        child.sendline("stop")
        child.expect(PIPELINE_ENDED, timeout=BEHAVIOR_S)
        assert child.match.group(2) == "stopped", f"'stop' left the dance {child.match.group(2)}"
        child.sendeof()
        child.expect(r"Shutting down - parking arm", timeout=BEHAVIOR_S)
        child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    finally:
        status = _close(child)
        child.logfile_read.close()
    assert status == 0, f"run_pipeline.sh exited {status}"
    died = [line for line in log.read_text().splitlines() if "process has died" in line]
    assert not died, "a node of the launch died on shutdown:\n" + "\n".join(died)


def test_supervisor_starts_the_twin_and_its_controllers_for_the_client(twin_domain, tmp_path):
    """The robot notebook's "On the rig": the supervisor, then the client's round trip."""
    sup = _spawn(f"exec {sys.executable} -m ros.node_supervisor --only arm --digital-twin", SOLUTION,
                 tmp_path / "supervisor.log")
    try:
        sup.expect(r"nodes: all server nodes up", timeout=START_S)
        client = subprocess.run([sys.executable, "-m", "ros.arm_client"], cwd=SOLUTION, env=os.environ.copy(),
                                capture_output=True, text=True, timeout=BEHAVIOR_S)
        assert "ROS 2 round-trip PASSED" in client.stdout, client.stdout[-3000:] + client.stderr[-3000:]
    finally:
        status = _close(sup)
    assert status == 0, f"the supervisor exited {status}"


def test_workshop_app_drives_the_twin(twin_domain, tmp_path):
    app = _spawn(f"exec {sys.executable} app.py --digital-twin --headless --no-llm --synthetic", SOLUTION,
                 tmp_path / "app.log")
    try:
        app.expect(r"starting gesture mimic", timeout=START_S)
        app.sendline("stop")
        app.expect(r"Mimic stopped after \d+ frames", timeout=BEHAVIOR_S)
        for text in ("dance", "wave", "go home", "drop it", "hold this"):
            app.sendline(text)
            app.expect(WORKSHOP_HANDLED, timeout=BEHAVIOR_S)
            assert "Traceback" not in app.before, f"{text!r} raised:\n{app.before[-3000:]}"
        app.sendline("quit")
        app.expect(r"bye - parking the arm", timeout=BEHAVIOR_S)
        app.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    finally:
        status = _close(app)
    assert status == 0, f"app.py exited {status}"


def test_the_twin_and_controllers_die_with_the_app(twin_domain, tmp_path):
    """app.py killed outright: the server nodes its supervisor started exit too,
    with every process each of them started (each is a process group of its own)."""
    app = _spawn(f"exec {sys.executable} app.py --digital-twin --headless --no-llm --synthetic", SOLUTION,
                 tmp_path / "app.log")
    app.expect(r"starting gesture mimic", timeout=START_S)
    found = subprocess.run(["pgrep", "-P", str(app.pid)], capture_output=True, text=True).stdout.split()
    children = [int(pid) for pid in found]
    assert children, "app.py started no server node"
    os.kill(app.pid, signal.SIGKILL)
    app.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    app.wait()
    for pid in children:
        assert wait_exit(pid), f"server node {pid} outlived app.py"
        left = subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True).stdout.split()
        for orphan in map(int, left):
            assert wait_exit(orphan), f"process {orphan} of server node {pid} outlived it"
