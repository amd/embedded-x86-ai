# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Run a workshop notebook the way a student does in JupyterLab.

Each notebook runs top to bottom in its own ``vvla-workshop`` kernel, the
kernel ``run_notebooks.sh`` registers. A cell that loops on ``ui_events``
(a live camera, a console, the dance controls) runs while a scripted
student types into its text box and clicks its buttons with the widget
messages a browser sends, then clicks the button that ends it.

Only the in-memory copy of a notebook changes: ``RIG_STANDINS`` goes first,
and ``REPORT`` last. The rig's hardware is replaced, nothing else:

* the arm and the arm camera: the MuJoCo digital twin (``robot.backend: sim``,
  ``cameras.arm.source: sim``),
* the webcam: a generated frame, panned slowly, with pose from
  ``common.fixtures.ScriptedPose`` (the real pose session still opens),
* the microphone: none (Whisper still loads; Llama stays real).

``RIG_STANDINS`` also records every ONNX Runtime session the kernel makes,
with the provider it got, so a test can check where YOLO ran, and turns on
``common.twin_voice.REPORT_RECORDS`` so each console command's record
reaches the kernel's output, where :meth:`Watcher.record` finds it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PIPE_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = PIPE_ROOT / "workshop" / "notebooks"
KERNEL = "vvla-workshop"

RIG_STANDINS = '''\
# notebook test: the rig's hardware is replaced; the notebook is not
#   arm + arm camera -> the MuJoCo digital twin (robot.backend: sim, cameras.arm.source: sim)
#   webcam           -> a generated frame, panned; pose is ScriptedPose
#   microphone       -> none
import sys as _sys
import threading as _threading
import time as _time
from pathlib import Path as _Path

import cv2 as _cv2
import numpy as _np
import onnxruntime as _ort

_SESSIONS = []  # (model, provider) of every ONNX Runtime session this kernel makes


class _RecordedSession(_ort.InferenceSession):
    def __init__(self, path_or_bytes, *args, **kwargs):
        super().__init__(path_or_bytes, *args, **kwargs)
        model = path_or_bytes if isinstance(path_or_bytes, (str, _Path)) else "<bytes>"
        _SESSIONS.append((str(model), self.get_providers()[0]))


_ort.InferenceSession = _RecordedSession

_sys.path.insert(0, str((_Path.cwd() / ".." / ".." / "solution").resolve()))
import common.config as _config
import common.speech as _speech

_load_config = _config.load_config


def _twin_config(path=None):
    cfg = _load_config(path)
    cfg["robot"]["backend"] = "sim"
    cfg["cameras"]["arm"]["source"] = "sim"
    return cfg


_config.load_config = _twin_config

import models.vision as _vision
from common.fixtures import ScriptedPose as _ScriptedPose

_scripted_pose = _ScriptedPose()


def _scripted_infer(self, frame_bgr, conf=None):
    return _scripted_pose.infer(frame_bgr)


_vision.PoseEstimator.infer = _scripted_infer


class _PhotoCamera:
    """The webcam: a generated frame, panned. ``read()`` blocks until the
    next frame is due at the camera's 30 fps, as the device does."""

    def __init__(self, *args, **kwargs):
        width, height = 960, 480
        x = _np.linspace(0, 255, width, dtype=_np.uint8)
        row = _np.stack([x, x[::-1], _np.full(width, 40, _np.uint8)], axis=1)
        self._img = _np.broadcast_to(row, (height, width, 3)).copy()
        self._t0 = self._last = _time.monotonic()
        self._open = True

    def isOpened(self):
        return self._open

    def read(self):
        if not self._open:
            return False, None
        wait = self._last + 1 / 30 - _time.monotonic()
        if wait > 0:
            _time.sleep(wait)
        self._last = _time.monotonic()
        span = self._img.shape[1] - 640
        x = int(span * (0.5 + 0.5 * _np.sin(0.5 * (self._last - self._t0))))
        return True, self._img[:, x : x + 640].copy()

    def grab(self):
        return self._open

    def retrieve(self):
        return self.read()

    def release(self):
        self._open = False

    def set(self, *args):
        return True

    def get(self, prop):
        sizes = {_cv2.CAP_PROP_FRAME_WIDTH: 640.0, _cv2.CAP_PROP_FRAME_HEIGHT: 480.0, _cv2.CAP_PROP_FPS: 30.0}
        return sizes.get(prop, 0.0)


_cv2.VideoCapture = _PhotoCamera


class _NoMic:
    error = "no microphone in the notebook test"
    transcribing = False

    def __init__(self, *args, **kwargs):
        self.opened = _threading.Event()
        self.opened.set()  # tried, and failed: see error

    def start(self):
        return self

    def stop(self):
        pass


_speech.MicListener = _NoMic

import common.twin_voice as _twin_voice

_twin_voice.REPORT_RECORDS = True  # the test reads each command's record from the kernel's output
'''

REPORT = '''\
import json as _json
print("REPORT " + _json.dumps({"sessions": _SESSIONS}))
'''

# A cell that loops on ui_events ends when one of these is clicked, in this order of preference.
FINISH = ("end session", "done", "stop")


def kernel_env() -> dict:
    """This process's environment (the setup's), without a display: the
    notebooks' resource HUD then fails quietly instead of opening windows."""
    env = os.environ.copy()
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(name, None)
    return env


class Watcher:
    """A second client of the kernel, as a browser tab is: it reads iopub
    (stdout, stderr, JSON displays and widget state, any parent) and sends
    widget messages and code on its own shell socket.

    Widget state is kept from the same comm messages the browser gets, so a
    waiter woken by a message sees the state that message carried.
    """

    def __init__(self, connection_info: dict):
        import zmq
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.load_connection_info(connection_info)
        self._kc.start_channels(shell=True, stdin=False, hb=False, control=False)
        self._lock = threading.Lock()
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.records: list[dict] = []
        self._widgets: dict[str, dict] = {}
        self._updates = 0  # widget updates the kernel sent (a live view's frames)
        self._listeners: list = []
        wake = f"inproc://iopub-watch-{id(self)}"
        self._wake_rx = self._kc.context.socket(zmq.PAIR)
        self._wake_rx.bind(wake)
        self._wake_tx = self._kc.context.socket(zmq.PAIR)
        self._wake_tx.connect(wake)
        self._thread = threading.Thread(target=self._pump, daemon=True, name="iopub-watch")
        self._thread.start()

    def _pump(self) -> None:
        import zmq

        iopub = self._kc.iopub_channel
        poller = zmq.Poller()
        poller.register(iopub.socket, zmq.POLLIN)
        poller.register(self._wake_rx, zmq.POLLIN)
        while True:
            ready = dict(poller.poll())
            if self._wake_rx in ready:
                return
            self._take(iopub.get_msg(timeout=0))

    def _take(self, msg: dict) -> None:
        kind, content = msg["msg_type"], msg["content"]
        with self._lock:
            if kind == "stream":
                (self.stdout if content.get("name") == "stdout" else self.stderr).append(content["text"])
            elif kind in ("display_data", "execute_result"):
                record = content.get("data", {}).get("application/json")
                if isinstance(record, dict) and "text" in record:
                    self.records.append(record)
            elif kind == "comm_open":
                self._widgets[content["comm_id"]] = dict(content.get("data", {}).get("state", {}))
            elif kind == "comm_msg":
                data = content.get("data", {})
                if data.get("method") in ("update", "echo_update"):
                    self._widgets.setdefault(content["comm_id"], {}).update(data.get("state", {}))
                    self._updates += data["method"] == "update"
            for wake in self._listeners:  # under the lock: none runs after its unsubscribe returns
                wake()

    def send_comm(self, comm_id: str, data: dict) -> None:
        """A widget message from this client, as a browser sends it.

        Never on nbclient's socket: nbclient waits there for a cell's reply
        with a separate poller, and a send on that socket can swallow the
        poller's wake-up.
        """
        kc = self._kc
        kc.shell_channel.send(kc.session.msg("comm_msg", {"comm_id": comm_id, "data": data}))

    def execute(self, code: str) -> None:
        """Run ``code`` from this client, as a second browser tab would. Its reply is not read."""
        self._kc.execute(code)

    def subscribe(self, callback):
        """Call ``callback()`` after every kernel message from now on; returns the unsubscribe."""
        with self._lock:
            self._listeners.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.remove(callback)

        return unsubscribe

    def widgets(self) -> dict[str, dict]:
        """Every widget's state, by comm id, as the kernel last sent it."""
        with self._lock:
            return {cid: dict(state) for cid, state in self._widgets.items()}

    def updates_since(self, since: "Mark") -> int:
        with self._lock:
            return self._updates - since.updates

    def mark(self) -> "Mark":
        """Cursor for output that arrives after this call. Earlier scenarios stay behind it."""
        with self._lock:
            return Mark(len(self.stdout), len(self.records), self._updates, len(self.stderr))

    def output_since(self, since: "Mark") -> tuple[str, str]:
        """``(stdout, stderr)`` the kernel printed after ``since``, from any thread."""
        with self._lock:
            return "".join(self.stdout[since.stdout:]), "".join(self.stderr[since.stderr:])

    def said(self, phrase: str, since: "Mark | int" = 0) -> bool:
        with self._lock:
            return phrase in "".join(self.stdout[_stdout_at(since):])

    def count(self, phrase: str, since: "Mark | int" = 0) -> int:
        with self._lock:
            return "".join(self.stdout[_stdout_at(since):]).count(phrase)

    def text(self) -> str:
        with self._lock:
            return "".join(self.stdout) + "".join(self.stderr)

    def record(self, text: str, since: "Mark | int" = 0) -> dict | None:
        """The latest record of ``text`` after ``since``.

        The latest record in the whole list is still the previous scenario's until
        this command finishes, so a wait has to start behind the mark.
        """
        with self._lock:
            found = [r for r in self.records[_records_at(since):] if r["text"] == text]
            return found[-1] if found else None

    def close(self) -> None:
        self._wake_tx.send(b"")  # the pump returns on it
        self._thread.join()
        self._wake_tx.close()
        self._wake_rx.close()
        self._kc.stop_channels()


@dataclass(frozen=True)
class Mark:
    """How far the watcher had read when a scenario began."""

    stdout: int = 0
    records: int = 0
    updates: int = 0
    stderr: int = 0


def _stdout_at(since: Mark | int) -> int:
    return since.stdout if isinstance(since, Mark) else int(since)


def _records_at(since: Mark | int) -> int:
    return since.records if isinstance(since, Mark) else 0


async def until(predicate, what: str, *sources, task=None):
    """``predicate()`` once it holds, checked again whenever a source reports new data.

    ``sources`` have ``subscribe(callback) -> unsubscribe`` (the
    :class:`Watcher`, a ROS probe); ``task`` ending also wakes the check.
    A wait whose event never comes is the module's hang guard's to fail.
    """
    loop = asyncio.get_running_loop()
    changed = asyncio.Event()

    def wake(*_):
        loop.call_soon_threadsafe(changed.set)

    unsubscribe = [source.subscribe(wake) for source in sources]
    if task is not None:
        task.add_done_callback(wake)
    started = loop.time()
    try:
        while True:
            changed.clear()
            value = predicate()
            if value:
                logger.info("%s after %.2f s", what, loop.time() - started)
                return value
            await changed.wait()
    finally:
        for off in unsubscribe:
            off()
        if task is not None:
            task.remove_done_callback(wake)


class Console:
    """Widgets of a running cell, found by label, and the messages a browser sends them.

    ``before`` is the watcher's widget ids from before the cell ran; the
    cell's own widgets are the ones that appear after.
    """

    def __init__(self, watcher: Watcher, before: set | None = None, since: Mark | int = 0):
        self.watcher = watcher
        self._before = set(before or ())
        self.since = since
        self.widgets: dict[str, str] = {}

    def said(self, phrase: str) -> bool:
        return self.watcher.said(phrase, since=self.since)

    def count(self, phrase: str) -> int:
        return self.watcher.count(phrase, since=self.since)

    def _new(self):
        return {cid: s for cid, s in self.watcher.widgets().items() if cid not in self._before}

    def buttons(self) -> dict[str, str]:
        return {s["description"]: cid for cid, s in self._new().items() if s.get("_model_name") == "ButtonModel"}

    def text_box(self) -> str | None:
        return next((cid for cid, s in self._new().items() if s.get("_model_name") == "TextModel"), None)

    def find_widgets(self) -> bool:
        """The 02_digital_twin console's widgets: its text box and its three buttons."""
        found = {}
        for comm_id, state in self._new().items():
            model = state.get("_model_name")
            if model == "TextModel" and state.get("placeholder", "").startswith("or type a command"):
                found["typed"] = comm_id
            elif model == "ButtonModel" and state.get("description") in ("stop", "reset", "end session"):
                found[state["description"]] = comm_id
        self.widgets = found
        return {"typed", "stop", "reset", "end session"} <= found.keys()

    def type(self, text: str, comm_id: str | None = None) -> None:
        """Type ``text`` into a text box and press Enter."""
        target = comm_id or self.widgets.get("typed") or self.text_box()
        self.watcher.send_comm(target, {"method": "update", "state": {"value": text}, "buffer_paths": []})

    def click(self, label: str) -> None:
        comm_id = self.widgets.get(label) or self.buttons()[label]
        self.watcher.send_comm(comm_id, {"method": "custom", "content": {"event": "click"}})

    async def record(self, text: str, since: Mark | None = None) -> dict:
        """The record of ``text`` after ``since`` (default: since the scenario began)."""
        after = self.since if since is None else since
        return await until(lambda: self.watcher.record(text, since=after), f"the record of {text!r}", self.watcher)


def cell_index(nb, cell_id: str) -> int:
    return next(i for i, c in enumerate(nb.cells) if c.get("id") == cell_id)


def cell_stdout(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream")


def with_standins(nb):
    """``nb`` with ``RIG_STANDINS`` first and ``REPORT`` last (in memory)."""
    import nbformat

    nb.cells.insert(0, nbformat.v4.new_code_cell(RIG_STANDINS))
    nb.cells.append(nbformat.v4.new_code_cell(REPORT))
    return nb


def report(nb) -> dict:
    """What ``REPORT`` printed, once it has run."""
    line = next(t for t in cell_stdout(nb.cells[-1]).splitlines() if t.startswith("REPORT "))
    return json.loads(line[len("REPORT "):])


def client_for(nb, path: Path):
    """An nbclient client with no per-cell timeout: each cell ends on its own reply."""
    from nbclient import NotebookClient

    return NotebookClient(nb, kernel_name=KERNEL, timeout=None, resources={"metadata": {"path": str(path.parent)}})


@dataclass
class NotebookRun:
    output: str
    sessions: list = field(default_factory=list)


# A student action: ("type", text), ("click", label), ("wait_for", phrase), or
# ("updates", n): the cell sent n widget updates, a live view's frames.
# The kernel handles the clicks in the order they arrive, so the button that ends
# the cell comes after every earlier action.
DEFAULT_ACTIONS = (("updates", 3),)


async def _interact(client, watcher, nb, index: int, actions) -> None:
    """Run the looping cell at ``index``, act on its widgets, then click the button that ends it."""
    console = Console(watcher, before=set(watcher.widgets()))
    task = asyncio.create_task(client.async_execute_cell(nb.cells[index], index))
    cell = nb.cells[index].get("id")
    await until(lambda: task.done() or any(b in console.buttons() for b in FINISH),
                f"the buttons of cell {cell}", watcher, task=task)
    if task.done():
        task.result()
        return
    mark = watcher.mark()  # a wait counts output from the last type/click on
    for action in actions:
        kind = action[0]
        if kind == "type":
            # The cell creates its widgets one message at a time: wait for this one.
            await until(lambda: task.done() or console.text_box(), f"the text box of cell {cell}",
                        watcher, task=task)
            if not task.done():
                mark = watcher.mark()
                console.type(action[1])
        elif kind == "click":
            await until(lambda: task.done() or action[1] in console.buttons(),
                        f"the {action[1]!r} button of cell {cell}", watcher, task=task)
            if not task.done():
                mark = watcher.mark()
                console.click(action[1])
        elif kind == "wait_for":
            await until(lambda: task.done() or watcher.said(action[1], since=mark),
                        f"{action[1]!r} from cell {cell}", watcher, task=task)
        elif kind == "updates":
            await until(lambda: task.done() or watcher.updates_since(mark) >= action[1],
                        f"{action[1]} widget updates from cell {cell}", watcher, task=task)
        else:
            raise ValueError(f"unknown student action {action!r}")
        if task.done():
            task.result()
            return
    console.click(next(b for b in FINISH if b in console.buttons()))
    # Not wait_for: nbclient reports a cancelled cell as "Kernel died".
    await asyncio.wait({task})
    task.result()


def run_notebook(path: Path, script: dict | None = None) -> NotebookRun:
    """Every cell of ``path`` in order, in a fresh kernel; ``script`` maps a
    looping cell's id to the student's actions (default: watch it draw three frames)."""
    import nbformat
    from nbclient.exceptions import CellExecutionError

    nb = with_standins(nbformat.read(path, as_version=4))
    client = client_for(nb, path)
    script = script or {}

    async def main() -> NotebookRun:
        async with client.async_setup_kernel(env=kernel_env()):
            watcher = Watcher(client.km.get_connection_info())
            try:
                for i, cell in enumerate(nb.cells):
                    if cell.cell_type != "code":
                        continue
                    try:
                        if "ui_events(" in cell.source:
                            await _interact(client, watcher, nb, i, script.get(cell.get("id"), DEFAULT_ACTIONS))
                        else:
                            await client.async_execute_cell(cell, i)
                    except CellExecutionError as e:
                        raise AssertionError(f"{path.name}, cell {cell.get('id')}:\n{e}") from None
                return NotebookRun(output=watcher.text(), sessions=report(nb)["sessions"])
            finally:
                watcher.close()

    return asyncio.run(main())
