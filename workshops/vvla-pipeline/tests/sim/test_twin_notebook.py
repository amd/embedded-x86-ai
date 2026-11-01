# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``02_digital_twin.ipynb`` section 5, driven the way JupyterLab drives it.

The module starts one ``vvla-workshop`` kernel and runs the notebook through
``c0000014`` once. Each test runs only the console cell (``c0000016``): its
``ui_events`` loop and its dispatch thread. The test types into the console's
text box and clicks its buttons with the same widget messages a browser sends.
So a typed command takes the console's own path: ``submit`` ->
``voice.on_transcript`` -> voice's waiting commands -> ``voice.dispatch_loop``
-> ``voice.run`` -> ``handle_transcript`` (``voice`` is the notebook's
``common.twin_voice.VoiceCommands``).

What the kernel does is checked three ways: a second client on the
kernel's iopub channel (matched by content, so it does not matter which
cell the dispatch thread's output is attributed to), ROS readings of the
twin taken by this process, and the kernel namespace once the session ends.

The in-memory copy of the notebook starts with ``RIG_STANDINS``
(:mod:`notebook_harness`): a generated frame is the webcam, pose comes from
``ScriptedPose``, and there is no microphone. Whisper, Llama on the iGPU
and the YOLO sessions on the NPU are real. Arm-camera detection still runs
the detector. The file on disk is not changed.

``cd workshop && ./run_notebooks.sh --test``
(without that setup YOLO cannot load the NPU and the test fails). The
notebook kernel holds both YOLO NPU contexts, so this runs apart from the
tests whose YOLO runs in the pytest process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import threading
import time
from pathlib import Path

import pytest

from notebook_harness import (
    NOTEBOOKS,
    PIPE_ROOT,
    Console,
    Watcher,
    cell_index,
    cell_stdout,
    client_for,
    kernel_env,
    until,
    with_standins,
)
from twin_ros_rig import (
    BODY_JOINTS,
    Probe,
    assert_domain_free,
    join_test_domain,
    process_gone,
    quat_close,
    twin_log_report,
    wait_exit,
    xml_poses,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

from common.twin_voice import RESET_LINE  # noqa: E402

logger = logging.getLogger(__name__)

NOTEBOOK = NOTEBOOKS / "03_integration" / "02_digital_twin.ipynb"
SESSION_ENDED = "session ended, arm parked"  # the console cell's last line

NAMESPACE = '''\
import json as _json
print("NS " + _json.dumps({
    "trace": voice.trace,
    "dispatcher_alive": dispatcher.is_alive(),
    "log": [line for line, _color in console._log],
    "pid": twin.pid,
    "twin_log": str(twin.log_path),
    "sessions": _SESSIONS,
}))
'''

# The notebook keeps INFO out of its cells (root at WARNING). This leaves the
# cells as they are and sends the workshop's INFO, and every warning, to a file.
KERNEL_LOG = '''\
import logging as _logging
_root = _logging.getLogger()
if not _root.handlers:
    _root.addHandler(_logging.lastResort)
for _h in _root.handlers:
    _h.setLevel(max(_h.level, _logging.WARNING))
_fh = _logging.FileHandler({path!r}, mode="a")
_fh.setFormatter(_logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s]: %(message)s", "%H:%M:%S"))
_root.addHandler(_fh)
for _name in ("common", "integration", "models", "ros", "so101_sim", "vla_pipeline"):
    _logging.getLogger(_name).setLevel(_logging.INFO)
import faulthandler as _faulthandler, signal as _signal
_faulthandler.register(_signal.SIGUSR1, file=_fh.stream, all_threads=True)  # the test's thread dump
print("KERNEL_LOG", getattr(globals().get("twin"), "log_path", ""))
'''

# voice.reset_scene() returns None. It clears ctx.stop_event, which the console cell's finally leaves set.
SETTLE = '''\
voice.reset_scene()
voice.trace.clear()
console._log.clear()
print("SETTLED")
'''

SESSIONS = '''\
import json as _json
print("SESSIONS " + _json.dumps(_SESSIONS))
'''

# Same line the notebook-run test rejects. Checked once on this kernel's whole transcript.
_FALLBACK = re.compile(r"fall(s|ing)? back to (the )?cpu", re.I)


def _assert_dispatcher_dead(namespace: dict) -> None:
    assert not namespace["dispatcher_alive"], "the console's dispatch thread outlived end session"


def _assert_yolo_on_npu(sessions) -> None:
    yolo = [(model, provider) for model, provider in sessions if "yolo" in model.lower()]
    assert yolo, "the notebook made no YOLO session"
    off_npu = [(Path(model).name, provider) for model, provider in yolo if provider != "VitisAIExecutionProvider"]
    assert not off_npu, f"YOLO ran off the NPU: {off_npu}"


class NotebookKernel:
    """One kernel for the module. Scenarios share it; the console cell is the part that re-runs.

    ``nbclient`` binds the client to the loop that started the kernel, so that loop
    lives on one thread and every scenario is submitted onto it.
    """

    def __init__(self):
        self.watcher: Watcher | None = None
        self.needs_new_kernel = False
        self.last_pid: int | None = None
        self.sessions = None
        self.kernel_log = PIPE_ROOT / "_artifacts" / "sim_tests" / "notebook" / "kernel.log"
        self.kernel_log.unlink(missing_ok=True)  # one log for the module; restarted kernels append
        self._log_from = 0  # kernel.log offset where the current scenario starts
        self._mark = None  # the watcher's position where the current scenario starts
        self.twin_log: str | None = None
        self._loop = asyncio.new_event_loop()
        self._open = threading.Event()
        self._closing = threading.Event()
        self._down = threading.Event()
        self._close_now = asyncio.Event()  # set on the loop: _serve shuts its kernel down
        self._opened = asyncio.Event()  # _open, for waiters on the loop
        self._serve_task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._thread_main, name="twin-notebook", daemon=False)
        self._serve_fut = None
        self._load_notebook()

    def _load_notebook(self) -> None:
        self.nb = with_standins(nbformat.read(NOTEBOOK, as_version=4))
        self.client = client_for(self.nb, NOTEBOOK)
        self.console_cell = cell_index(self.nb, "c0000016")
        self.cleanup = cell_index(self.nb, "c0000018")

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def start(self) -> None:
        self._thread.start()
        self._serve_fut = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        # _serve ending before the kernel opened wakes this wait too.
        self._serve_fut.add_done_callback(lambda _f: self._open.set())
        self._open.wait()
        if self._serve_fut.done():
            try:
                self._serve_fut.result()
            except BaseException:
                self.finish()
                raise

    def finish(self) -> None:
        try:
            # _down is set once the kernel process has exited. Coverage needs a live kernel.
            if self._open.is_set() and not self._closing.is_set() and not self._down.is_set():
                self._call(self._coverage_and_cleanup())
        finally:
            self._closing.set()
            self._loop.call_soon_threadsafe(self._close_now.set)
            self._down.wait()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()

    def run(self, script, end_with: str = "end session", after=None) -> dict:
        """Run ``script`` against a fresh console cell. ``end_with`` is "end session", "interrupt", or "rerun".

        A scenario that fails, the hang guard's timeout included, first has
        every kernel thread's stack written to kernel.log.
        """
        try:
            return self._call(self._scenario(script, end_with, after))
        except BaseException:
            self._dump_kernel_threads()
            raise

    async def _serve(self) -> None:
        self._serve_task = asyncio.current_task()
        self._down.clear()
        try:
            async with self.client.async_setup_kernel(env=kernel_env()):
                self.watcher = Watcher(self.client.km.get_connection_info())
                try:
                    await self._through("c0000014")
                    await self._log_to_file()
                    self._open.set()
                    self._opened.set()
                    await self._close_now.wait()
                finally:
                    if self.watcher is not None:
                        self.watcher.close()
                        self.watcher = None
        finally:
            self._down.set()

    async def _restart(self) -> None:
        """SIGINT during mimic wedges the NPU session, and the rerun test closes the stack.

        A new kernel process is the recovery. Re-running those cells in the same process
        starts a second twin or dies with the wedged device.
        """
        self._close_now.set()
        await asyncio.gather(self._serve_task, return_exceptions=True)
        self._close_now = asyncio.Event()
        self._opened = asyncio.Event()
        self._open.clear()
        self._load_notebook()
        serving = asyncio.create_task(self._serve())
        opened = asyncio.create_task(self._opened.wait())
        done, _ = await asyncio.wait({serving, opened}, return_when=asyncio.FIRST_COMPLETED)
        if opened not in done:
            opened.cancel()
            raise RuntimeError("the notebook kernel did not come back") from serving.exception()

    async def _through(self, cell_id: str) -> None:
        for i in range(cell_index(self.nb, cell_id) + 1):
            await self.client.async_execute_cell(self.nb.cells[i], i)

    async def _execute_new(self, source: str) -> str:
        cell = nbformat.v4.new_code_cell(source)
        self.nb.cells.append(cell)
        await self.client.async_execute_cell(cell, len(self.nb.cells) - 1)
        return cell_stdout(cell)

    async def _log_to_file(self) -> None:
        self.kernel_log.parent.mkdir(parents=True, exist_ok=True)
        text = await self._execute_new(KERNEL_LOG.format(path=str(self.kernel_log)))
        line = next((t for t in text.splitlines() if t.startswith("KERNEL_LOG")), "")
        self.twin_log = line[len("KERNEL_LOG"):].strip() or None

    async def _probe(self) -> dict:
        text = await self._execute_new(NAMESPACE)
        line = next(t for t in reversed(text.splitlines()) if t.startswith("NS "))
        namespace = json.loads(line[3:])
        self.twin_log = namespace.get("twin_log") or self.twin_log
        return namespace

    def failure_report(self, dest: Path, since: float | None) -> str:
        """What the kernel printed and logged since the scenario started, and the notebook twin's log."""
        parts = []
        if self.watcher is not None and self._mark is not None:
            out, err = self.watcher.output_since(self._mark)
            (dest / "kernel_output.txt").write_text(f"--- stdout\n{out}\n--- stderr\n{err}\n")
            lines = (out + err).splitlines()
            parts.append(
                f"notebook kernel output since the scenario started -> kernel_output.txt, "
                f"last {min(len(lines), 40)} of {len(lines)} lines:\n" + "\n".join(lines[-40:])
            )
        if self.kernel_log.exists():
            with open(self.kernel_log, "rb") as f:
                f.seek(self._log_from)
                text = f.read().decode(errors="replace")
            (dest / "kernel.log").write_text(text)
            lines = text.splitlines()
            parts.append(
                f"notebook kernel log since the scenario started -> kernel.log, "
                f"last {min(len(lines), 150)} of {len(lines)} lines:\n" + "\n".join(lines[-150:])
            )
        if self.twin_log:
            parts.append(twin_log_report(self.twin_log, dest))
        return "\n\n".join(parts) or "notebook kernel: no log"

    async def _settle(self) -> None:
        text = await self._execute_new(SETTLE)
        if "reset failed" in text or "SETTLED" not in text:
            raise RuntimeError(f"voice.reset_scene() did not park the arm:\n{text}")

    async def _end_cell(self, task, console: Console, end_with: str) -> None:
        from nbclient.exceptions import CellExecutionError

        ending = time.monotonic()
        if end_with == "interrupt":
            await self.client.km.interrupt_kernel()
        elif end_with == "rerun":
            for i in (self.console_cell, self.cleanup):
                self.watcher.execute(self.nb.cells[i].source)
        elif not console.said(SESSION_ENDED) and console.widgets.get("end session"):
            console.click("end session")
        # Not wait_for: nbclient reports a cancelled cell as "Kernel died".
        await asyncio.wait({task})
        logger.info("the console cell ended %.2f s after %r", time.monotonic() - ending, end_with)
        try:
            task.result()
        except CellExecutionError as e:
            if end_with != "interrupt" or "KeyboardInterrupt" not in str(e):
                raise

    def _dump_kernel_threads(self) -> None:
        """SIGUSR1: faulthandler (registered by ``KERNEL_LOG``) writes every thread's stack to kernel.log."""
        process = getattr(self.client.km.provisioner, "process", None)
        if process is not None and process.poll() is None:
            os.kill(process.pid, signal.SIGUSR1)

    async def _scenario(self, script, end_with: str, after) -> dict:
        if self.needs_new_kernel:
            await self._restart()
            self.needs_new_kernel = False
        assert self.watcher is not None
        self._log_from = self.kernel_log.stat().st_size if self.kernel_log.exists() else 0
        mark = self._mark = self.watcher.mark()
        task = asyncio.create_task(
            self.client.async_execute_cell(self.nb.cells[self.console_cell], self.console_cell)
        )
        console = Console(self.watcher, before=set(self.watcher.widgets()), since=mark)
        error = None
        try:
            await until(lambda: console.find_widgets() or task.done(), "the console widgets",
                        self.watcher, task=task)
            if task.done():
                task.result()  # the cell ended before its console came up: its error
            await script(console)
        except Exception as e:
            error = e
        try:
            await self._end_cell(task, console, end_with)
        except Exception as e:
            self.needs_new_kernel = True  # a cell that would not end leaves the kernel wedged
            raise e from error
        if error is None and after is not None:
            try:
                await after()
            except Exception as e:
                error = e
        try:
            namespace = await self._probe()
        except Exception as e:
            self.needs_new_kernel = True
            raise error or e
        namespace.update(
            parked=self.watcher.said(SESSION_ENDED, since=mark),
            twin_alive=not process_gone(namespace["pid"]),
        )
        self.last_pid = namespace["pid"]
        self.sessions = namespace["sessions"]
        if end_with in ("rerun", "interrupt"):
            # Interrupt wedges the NPU session. Rerun already ran the cleanup cell.
            # Either way the next scenario needs its own kernel. The probe above already
            # recorded whether this scenario's twin was still alive.
            self.needs_new_kernel = True
        else:
            # Settle a failed scenario too: the next case on this kernel checks voice.trace.
            try:
                await self._settle()
            except Exception:
                if error is None:
                    raise
        if error is not None:
            raise error
        return namespace

    async def _coverage_and_cleanup(self) -> None:
        sessions = self.sessions
        try:
            text = await self._execute_new(SESSIONS)
            line = next(t for t in reversed(text.splitlines()) if t.startswith("SESSIONS "))
            sessions = json.loads(line[len("SESSIONS "):])
        except Exception:
            if sessions is None:
                raise
        _assert_yolo_on_npu(sessions)
        assert self.watcher is not None
        fallback = [line.strip() for line in self.watcher.text().splitlines() if _FALLBACK.search(line)]
        assert not fallback, f"02_digital_twin fell back to the CPU: {fallback}"
        if self.last_pid is not None and not process_gone(self.last_pid):
            # A second hands.close() raises, so skip this when a scenario already ran c0000018.
            await self.client.async_execute_cell(self.nb.cells[self.cleanup], self.cleanup)
            assert await asyncio.to_thread(wait_exit, self.last_pid), "the cleanup cell left the twin node running"


@pytest.fixture(scope="module")
def observers(workshop_cfg):
    from common.twin_ros import TwinScene

    join_test_domain()
    probe = Probe(workshop_cfg)
    scene = TwinScene()
    yield probe, scene
    scene.close()
    probe.close()


@pytest.fixture(scope="module")
def kernel(observers):
    assert_domain_free("a twin already runs on the test domain")
    session = NotebookKernel()
    session.start()
    try:
        yield session
    finally:
        session.finish()
    assert_domain_free("the notebook's twin node outlived its cleanup cell", wait=True)


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_console_pickup(observers, kernel, text, body, spoken):
    _probe, scene = observers
    seen = {}

    async def script(console):
        console.type(text)
        record = await console.record(text)
        seen["record"] = record
        seen["reading"] = await asyncio.to_thread(scene.reading, body, after=time.time(), timeout=None)
        await until(lambda: console.said(f"Got the {spoken}"), f"'Got the {spoken}'", console.watcher)

    namespace = kernel.run(script)
    record, reading = seen["record"], seen["reading"]
    assert record["body"] == body
    assert record["grasped"], record
    assert record["z"] - record["z0"] > 0.03, record
    assert reading["grasped"] and reading["xyz"][2] > 0.08, reading
    assert namespace["trace"] == [record], "the namespace trace and the iopub record differ"
    _assert_dispatcher_dead(namespace)


def test_console_reset(observers, kernel, workshop_cfg):
    """The GUI sequence on the console's own widgets: pick up the toothbrush,
    drop it, press reset, and pick the toothbrush up again."""
    from common.motion import REST_POSE

    probe, scene = observers
    home = xml_poses(workshop_cfg)
    seen = {}

    async def script(console):
        console.type("pick up the toothbrush")
        first = await console.record("pick up the toothbrush")
        assert first["grasped"], first
        console.type("drop it")
        seen["drop"] = await console.record("drop it")
        seen["dropped"] = await asyncio.to_thread(scene.reading, "toothbrush", after=time.time(), timeout=None)
        seen["jaw"] = (await asyncio.to_thread(probe.joints, after=time.time()))["gripper"]
        console.click("reset")
        await until(lambda: console.said(RESET_LINE), "the reset line", console.watcher)
        # The reset parked the arm through the controllers, then put the props
        # back, before the console said so: the server's next report shows it parked.
        said_at = time.time()
        seen["parked"] = await asyncio.to_thread(probe.joints, after=said_at)
        seen["scene"] = await asyncio.to_thread(scene.latest, after=said_at, timeout=None, settled=True)
        again = console.watcher.mark()
        console.type("pick up the toothbrush")
        seen["again"] = await console.record("pick up the toothbrush", since=again)

    namespace = kernel.run(script)
    assert "Released." in seen["drop"]["said"], seen["drop"]
    dropped = seen["dropped"]
    assert not dropped["grasped"], f"the toothbrush is still between the jaws after 'drop it': {dropped}"
    assert seen["jaw"] > 80.0, f"'drop it' opened the jaws only to {seen['jaw']:.0f}"
    joints = seen["parked"]
    for k in BODY_JOINTS:
        assert joints[k] == pytest.approx(REST_POSE[k], abs=3.0), (k, joints)
    assert joints["gripper"] >= 70.0, joints
    assert seen["scene"] is not None, "the props did not come to rest after the reset"
    for name, qpos in home.items():
        got = seen["scene"]["bodies"][name]
        assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
        assert quat_close(got["quat"], qpos[3:7]), name
        assert not got["grasped"], name
    again = seen["again"]
    assert again["grasped"] and again["z"] - again["z0"] > 0.03, again
    texts = [r["text"] for r in namespace["trace"]]
    assert texts == ["pick up the toothbrush", "drop it", "pick up the toothbrush"], texts
    panel = " ".join(line.strip() for line in namespace["log"])  # the panel wraps long lines
    assert f"robot: {RESET_LINE}" in panel, namespace["log"]
    _assert_dispatcher_dead(namespace)


def _arm_moved(probe, start) -> bool:
    j = probe.joints()
    return bool(j) and max(abs(j[k] - start[k]) for k in BODY_JOINTS) > 3.0


def test_console_end_session_during_mimic(observers, kernel):
    """"end session" while the arm mirrors the webcam: the console cell ends
    and parks the arm. The module teardown closes the twin once."""
    probe, _scene = observers

    async def script(console):
        start = await asyncio.to_thread(probe.joints)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), "the arm to follow the webcam", probe)

    namespace = kernel.run(script)
    assert namespace["parked"], "the console cell did not park the arm"
    _assert_dispatcher_dead(namespace)


def test_console_interrupted(observers, kernel):
    """The kernel's stop button ends the console like "end session": the arm
    parks at rest, and the twin keeps running until the cleanup cell closes it."""
    from common.motion import REST_POSE

    probe, _scene = observers
    seen = {}

    async def script(console):
        start = await asyncio.to_thread(probe.joints)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), "the arm to follow the webcam", probe)

    async def after():
        seen["joints"] = await asyncio.to_thread(probe.joints, after=time.time())

    namespace = kernel.run(script, end_with="interrupt", after=after)
    assert namespace["parked"], "the interrupted console cell did not park the arm"
    assert namespace["twin_alive"], "the kernel's stop button also stopped the twin program"
    joints = seen["joints"]
    assert joints, "no joint state from the twin after the interrupt"
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, f"the arm did not reach rest after the interrupt: {off}"
    _assert_dispatcher_dead(namespace)


def test_console_run_and_advance(observers, kernel):
    """Run and advance on the running console cell, then on the cleanup cell:
    the session ends and parks the arm, the console cell does not start a
    second session, and the cleanup cell closes the twin."""
    probe, _scene = observers
    seen = {}

    async def script(console):
        seen["console"] = console
        start = await asyncio.to_thread(probe.joints)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), "the arm to follow the webcam", probe)

    async def after():
        console = seen["console"]
        await until(lambda: console.said("twin closed"), "the cleanup cell's 'twin closed'", console.watcher)
        console.watcher.execute('print("kernel free")')
        await until(lambda: console.said("kernel free"), "a cell run after the cleanup", console.watcher)
        seen["parked"] = console.count(SESSION_ENDED)

    namespace = kernel.run(script, end_with="rerun", after=after)
    assert seen["parked"] == 1, "the console cell ran a second session"
    _assert_dispatcher_dead(namespace)
    assert wait_exit(namespace["pid"]), "the cleanup cell left the twin node running"


def test_console_stop(observers, kernel):
    """The red stop button ends the console cell: the in-flight pickup is not a grasp
    and the arm parks. Saying "stop" only halts the behavior. The module teardown closes the twin."""
    from common.motion import REST_POSE

    probe, _scene = observers
    seen = {}

    async def script(console):
        start = await asyncio.to_thread(probe.joints)
        console.type("pick up the bottle")
        await until(lambda: _arm_moved(probe, start), "the arm to start moving", probe)
        console.click("stop")
        await until(lambda: console.said(SESSION_ENDED), "the console cell to end", console.watcher)
        seen["record"] = await console.record("pick up the bottle")
        seen["joints"] = await asyncio.to_thread(probe.joints, after=time.time())

    namespace = kernel.run(script)
    assert namespace["parked"], "the stop button did not end the console cell"
    assert seen["record"]["grasped"] is False, seen["record"]
    assert [r["text"] for r in namespace["trace"]] == ["pick up the bottle"]
    joints = seen["joints"]
    assert joints, "no joint state from the twin after stop"
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, f"the arm did not reach rest after stop: {off}"
    _assert_dispatcher_dead(namespace)
