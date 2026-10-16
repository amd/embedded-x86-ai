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
* the webcam: Ultralytics' ``zidane.jpg`` (two people), panned slowly,
* the microphone: none (Whisper still loads; Llama stays real).

``RIG_STANDINS`` also records every ONNX Runtime session the kernel makes,
with the provider it got, so a test can check where YOLO ran, and turns on
``common.twin_voice.REPORT_RECORDS`` so each console command's record
reaches the kernel's output, where :meth:`Watcher.record` finds it.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PIPE_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = PIPE_ROOT / "workshop" / "notebooks"
KERNEL = "vvla-workshop"

RIG_STANDINS = '''\
# notebook test: the rig's hardware is replaced; the notebook is not
#   arm + arm camera -> the MuJoCo digital twin (robot.backend: sim, cameras.arm.source: sim)
#   webcam           -> Ultralytics' zidane.jpg (two people), panned slowly
#   microphone       -> none
import importlib.util as _ilu
import sys as _sys
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

_spec = _ilu.find_spec("ultralytics")
_PHOTO = _Path(_spec.submodule_search_locations[0]) / "assets" / "zidane.jpg" if _spec else None
if _PHOTO is None or not _PHOTO.is_file():
    raise RuntimeError(
        "the webcam stand-in is Ultralytics' zidane.jpg: run the README's install step "
        "`pip install ultralytics` in .venv"
    )


class _PhotoCamera:
    """The webcam: two people panning left and right, 30 frames a second."""

    def __init__(self, *args, **kwargs):
        img = _cv2.imread(str(_PHOTO))
        width = round(img.shape[1] * 480 / img.shape[0])
        self._img = _cv2.resize(img, (width, 480))
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
        pass

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
    """A second client on the kernel's iopub channel: stdout, stderr and JSON displays, any parent."""

    def __init__(self, connection_info: dict):
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.load_connection_info(connection_info)
        self._kc.start_channels(shell=False, stdin=False, hb=False, control=False)
        self._lock = threading.Lock()
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.records: list[dict] = []
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True, name="iopub-watch")
        self._thread.start()

    def _pump(self) -> None:
        while self._running:
            try:
                msg = self._kc.get_iopub_msg(timeout=0.2)
            except Exception:
                continue
            kind, content = msg["msg_type"], msg["content"]
            with self._lock:
                if kind == "stream":
                    (self.stdout if content.get("name") == "stdout" else self.stderr).append(content["text"])
                elif kind in ("display_data", "execute_result"):
                    record = content.get("data", {}).get("application/json")
                    if isinstance(record, dict) and "text" in record:
                        self.records.append(record)

    def mark(self) -> int:
        with self._lock:
            return len(self.stdout)

    def said(self, phrase: str, since: int = 0) -> bool:
        with self._lock:
            return phrase in "".join(self.stdout[since:])

    def text(self) -> str:
        with self._lock:
            return "".join(self.stdout) + "".join(self.stderr)

    def record(self, text: str) -> dict | None:
        with self._lock:
            return next((r for r in self.records if r["text"] == text), None)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._kc.stop_channels()


async def until(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


class Console:
    """Widgets of a running cell, found by label, and the messages a browser sends them."""

    def __init__(self, client, watcher: Watcher, before: set | None = None):
        self._client = client
        self.watcher = watcher
        self._before = set(before or ())
        self.widgets: dict[str, str] = {}

    def _new(self):
        return {cid: s for cid, s in self._client.widget_state.items() if cid not in self._before}

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

    def _send(self, comm_id: str, data: dict) -> None:
        kc = self._client.kc
        kc.shell_channel.send(kc.session.msg("comm_msg", {"comm_id": comm_id, "data": data}))

    def type(self, text: str, comm_id: str | None = None) -> None:
        """Type ``text`` into a text box and press Enter."""
        target = comm_id or self.widgets.get("typed") or self.text_box()
        self._send(target, {"method": "update", "state": {"value": text}, "buffer_paths": []})

    def click(self, label: str) -> None:
        comm_id = self.widgets.get(label) or self.buttons()[label]
        self._send(comm_id, {"method": "custom", "content": {"event": "click"}})

    async def record(self, text: str, timeout: float = 240.0) -> dict:
        return await until(lambda: self.watcher.record(text), timeout, f"the record of {text!r}")


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


def client_for(nb, path: Path, timeout: float = 600.0):
    from nbclient import NotebookClient

    return NotebookClient(nb, kernel_name=KERNEL, timeout=timeout, resources={"metadata": {"path": str(path.parent)}})


@dataclass
class NotebookRun:
    output: str
    sessions: list = field(default_factory=list)


# A student action: ("type", text), ("click", label), ("wait_for", phrase, timeout_s) or ("sleep", s).
DEFAULT_ACTIONS = (("sleep", 3.0),)


async def _interact(client, watcher, nb, index: int, actions) -> None:
    """Run the looping cell at ``index``, act on its widgets, then click the button that ends it."""
    console = Console(client, watcher, before=set(client.widget_state))
    task = asyncio.create_task(client.async_execute_cell(nb.cells[index], index))
    await until(lambda: task.done() or any(b in console.buttons() for b in FINISH), 300.0,
                f"the buttons of cell {nb.cells[index].get('id')}")
    if task.done():
        task.result()
        return
    mark = watcher.mark()  # a wait_for counts output from the last type/click on
    for action in actions:
        kind = action[0]
        if kind == "type":
            mark = watcher.mark()
            console.type(action[1])
        elif kind == "click":
            mark = watcher.mark()
            console.click(action[1])
        elif kind == "wait_for":
            await until(lambda: task.done() or watcher.said(action[1], since=mark), action[2],
                        f"{action[1]!r} from cell {nb.cells[index].get('id')}")
        elif kind == "sleep":
            await asyncio.sleep(action[1])
        if task.done():
            task.result()
            return
    console.click(next(b for b in FINISH if b in console.buttons()))
    await asyncio.wait_for(task, timeout=120.0)


def run_notebook(path: Path, script: dict | None = None) -> NotebookRun:
    """Every cell of ``path`` in order, in a fresh kernel; ``script`` maps a
    looping cell's id to the student's actions (default: watch for 3 s)."""
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
