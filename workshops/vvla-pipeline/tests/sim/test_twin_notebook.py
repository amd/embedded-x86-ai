# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``02_digital_twin.ipynb`` section 5, driven the way JupyterLab drives it.

The notebook runs top to bottom in the ``vvla-workshop`` kernel. Then the
console cell (``c0000016``) runs for real: its ``ui_events`` loop and its
dispatch thread. The test types into the console's text box and clicks its
buttons with the same widget messages a browser sends. So a typed command
takes the console's own path: ``submit`` -> ``voice.on_transcript`` -> the
``voice.commands`` queue -> ``voice.dispatch_loop`` -> ``voice.run`` ->
``handle_transcript`` (``voice`` is the notebook's ``common.twin_voice.VoiceCommands``).

What the kernel does is checked three ways: a second client on the
kernel's iopub channel (matched by content, so it does not matter which
cell the dispatch thread's output is attributed to), ROS readings of the
twin taken by this process, and the kernel namespace once the session ends.

The in-memory copy of the notebook starts with ``RIG_STANDINS``
(:mod:`notebook_harness`): a photo of people is the webcam and there is no
microphone. Whisper, Llama on the iGPU and YOLO on the NPU are real. The
file on disk is not changed.

``source scripts/ryzen_ai_env.sh && source /opt/ros/jazzy/setup.bash && pytest tests/sim -m docs``
(without ``ryzen_ai_env.sh`` YOLO cannot load the NPU and the test fails). The
notebook kernel holds both YOLO NPU contexts, so this runs apart from the
tests whose YOLO runs in the pytest process.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from notebook_harness import NOTEBOOKS, Console, Watcher, cell_index, cell_stdout, client_for, kernel_env, until, with_standins
from twin_ros_rig import BODY_JOINTS, Probe, join_test_domain, quat_close, xml_poses

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

NOTEBOOK = NOTEBOOKS / "03_integration" / "02_digital_twin.ipynb"
RESET_LINE = "Reset. Arm parked, objects back on the table."  # common.twin_voice.RESET_LINE

NAMESPACE = '''\
import json as _json
print("NS " + _json.dumps({
    "trace": voice.trace,
    "dispatcher_alive": dispatcher.is_alive(),
    "log": [line for line, _color in console._log],
    "pid": twin.pid,
    "sessions": _SESSIONS,
}))
'''


def run_console(script) -> tuple[dict, int]:
    """Run the notebook through ``c0000014``, then ``script(console)`` against the live console cell.

    Returns the kernel namespace read after "end session" and the twin node's pid.
    The cleanup cell ``c0000018`` runs last.
    """
    nb = with_standins(nbformat.read(NOTEBOOK, as_version=4))
    nb.cells.append(nbformat.v4.new_code_cell(NAMESPACE))
    upto, console_cell, cleanup = cell_index(nb, "c0000014"), cell_index(nb, "c0000016"), cell_index(nb, "c0000018")
    probe_cell = len(nb.cells) - 1
    client = client_for(nb, NOTEBOOK, timeout=900)

    async def main():
        async with client.async_setup_kernel(env=kernel_env()):
            watcher = Watcher(client.km.get_connection_info())
            try:
                for i in range(upto + 1):
                    await client.async_execute_cell(nb.cells[i], i)
                task = asyncio.create_task(client.async_execute_cell(nb.cells[console_cell], console_cell))
                console = Console(client, watcher)
                try:
                    await until(console.find_widgets, 60.0, "the console widgets")
                    await script(console)
                finally:
                    if console.widgets.get("end session"):
                        console.click("end session")
                    await asyncio.wait_for(task, timeout=120.0)
                await client.async_execute_cell(nb.cells[probe_cell], probe_cell)
                line = next(t for t in cell_stdout(nb.cells[probe_cell]).splitlines() if t.startswith("NS "))
                namespace = json.loads(line[3:])
                await client.async_execute_cell(nb.cells[cleanup], cleanup)
                return namespace
            finally:
                watcher.close()

    namespace = asyncio.run(main())
    return namespace, namespace["pid"]


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


@pytest.fixture(scope="module")
def observers(workshop_cfg):
    from common.twin_ros import TwinScene

    join_test_domain()
    probe = Probe(workshop_cfg["robot"]["state_topic"])
    scene = TwinScene()
    yield probe, scene
    scene.close()
    probe.close()


@pytest.fixture
def no_twin(observers, workshop_cfg):
    probe, _scene = observers
    state = workshop_cfg["robot"]["state_topic"]
    assert probe.wait_publishers(state, 0, timeout=15), "a twin already runs on the test domain"
    yield
    assert probe.wait_publishers(state, 0, timeout=20), "the notebook's twin node outlived its cleanup cell"


def _assert_closed(namespace: dict, pid: int) -> None:
    yolo = [(model, provider) for model, provider in namespace["sessions"] if "yolo" in model.lower()]
    assert yolo, "the notebook made no YOLO session"
    off_npu = [(Path(model).name, provider) for model, provider in yolo if provider != "VitisAIExecutionProvider"]
    assert not off_npu, f"YOLO ran off the NPU: {off_npu}"
    assert not namespace["dispatcher_alive"], "the console's dispatch thread outlived end session"
    deadline = time.time() + 15.0
    while not _gone(pid) and time.time() < deadline:
        time.sleep(0.1)
    assert _gone(pid), "the cleanup cell left the twin node running"


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_console_pickup(observers, no_twin, text, body, spoken):
    _probe, scene = observers
    seen = {}

    async def script(console):
        console.type(text)
        record = await console.record(text)
        seen["record"] = record
        seen["reading"] = scene.reading(body, after=time.time(), timeout=5.0)
        await until(lambda: console.watcher.said(f"Got the {spoken}"), 5.0, f"'Got the {spoken}'")

    namespace, pid = run_console(script)
    record, reading = seen["record"], seen["reading"]
    assert record["body"] == body
    assert record["grasped"], record
    assert record["z"] - record["z0"] > 0.03, record
    assert reading["grasped"] and reading["xyz"][2] > 0.08, reading
    assert namespace["trace"] == [record], "the namespace trace and the iopub record differ"
    _assert_closed(namespace, pid)


def test_console_reset(observers, no_twin, workshop_cfg):
    from common.motion import REST_POSE

    probe, scene = observers
    home = xml_poses(workshop_cfg)
    seen = {}

    async def script(console):
        console.type("pick up the toothbrush")
        first = await console.record("pick up the toothbrush")
        assert first["grasped"], first
        console.click("reset")
        await until(lambda: console.watcher.said(RESET_LINE), 15.0, "the reset line")
        await asyncio.sleep(1.5)  # the props settle; the servos reach the parked goal
        seen["scene"] = scene.latest(after=time.time(), timeout=5.0)
        seen["joints"] = probe.joints(after=time.time(), timeout=5.0)
        console.type("pick up the ball")
        seen["ball"] = await console.record("pick up the ball")

    namespace, pid = run_console(script)
    for name, qpos in home.items():
        got = seen["scene"]["bodies"][name]
        assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
        assert quat_close(got["quat"], qpos[3:7]), name
        assert not got["grasped"], name
    joints = seen["joints"]
    for k in BODY_JOINTS:
        assert joints[k] == pytest.approx(REST_POSE[k], abs=3.0), (k, joints)
    assert joints["gripper"] >= 70.0, joints
    ball = seen["ball"]
    assert ball["grasped"] and ball["z"] - ball["z0"] > 0.03, ball
    assert [r["text"] for r in namespace["trace"]] == ["pick up the toothbrush", "pick up the ball"]
    panel = " ".join(line.strip() for line in namespace["log"])  # the panel wraps long lines
    assert f"robot: {RESET_LINE}" in panel, namespace["log"]
    _assert_closed(namespace, pid)


def test_console_stop(observers, no_twin):
    probe, _scene = observers
    seen = {}

    async def script(console):
        start = probe.joints(timeout=10.0)
        console.type("pick up the bottle")

        def moving():
            j = probe.joints()
            return j and max(abs(j[k] - start[k]) for k in BODY_JOINTS) > 3.0

        await until(moving, 30.0, "the arm to start moving")
        console.click("stop")
        await asyncio.sleep(1.0)
        first = probe.joints(after=time.time())
        await asyncio.sleep(1.0)
        second = probe.joints(after=time.time())
        seen["drift"] = {k: round(second[k] - first[k], 2) for k in BODY_JOINTS}
        seen["record"] = await console.record("pick up the bottle", timeout=30.0)

    namespace, pid = run_console(script)
    moved = {k: d for k, d in seen["drift"].items() if abs(d) > 1.0}
    assert not moved, f"the arm kept moving after stop: {moved}"
    assert seen["record"]["grasped"] is False, seen["record"]
    assert [r["text"] for r in namespace["trace"]] == ["pick up the bottle"]
    _assert_closed(namespace, pid)
