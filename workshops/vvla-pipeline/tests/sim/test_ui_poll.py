# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Cells run while a ``common.ui_poll.ui_events`` cell loops, in a real kernel.

A loop cell like the twin console's (``poll(10)`` at 15 Hz until
``finished``) runs in a ``python3`` kernel. The test queues other cells the
way JupyterLab does while it loops, presses Jupyter's stop button (SIGINT
to the kernel's process group, as the server sends it), and reads each
cell's reply and output.
"""

from __future__ import annotations

import queue

import pytest

from notebook_harness import PIPE_ROOT

pytestmark = pytest.mark.sim
pytest.importorskip("jupyter_ui_poll")
pytest.importorskip("jupyter_client")

SETUP = f"""
import sys, threading, time
sys.path.insert(0, {str(PIPE_ROOT / "workshop" / "solution")!r})
from common.ui_poll import ui_events
"""

# The console's loop. It prints "loop polling" once ui_events polls, and
# "loop holds N" whenever the cells it holds back change (the wrapper's
# ``_events``, which its replay runs).
LOOP = """
finished = threading.Event()
print("loop started")
try:
    with ui_events(on_rerun=finished.set) as poll:
        wrapper, held = ui_events(), 0
        poll(10)
        print("loop polling", flush=True)
        while not finished.is_set():
            poll(10)
            if len(wrapper._events) != held:
                held = len(wrapper._events)
                print(f"loop holds {{held}}", flush=True)
                {on_hold}
            time.sleep(1 / 15)
finally:
    finished.set()
    print("loop ended")
"""
UNTIL_STOPPED = LOOP.format(on_hold="pass")
UNTIL_HOLDING = LOOP.format(on_hold="finished.set()")


class Kernel:
    """A real kernel; every wait blocks on its iopub and shell sockets."""

    def __init__(self):
        from jupyter_client.manager import KernelManager

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=None)
        self._names: dict[str, str] = {}
        self.replies: dict[str, dict] = {}
        self.started: set[str] = set()
        self.stdout: dict[str, str] = {}
        self.run("setup", SETUP)
        self.wait_reply("setup")

    def _name(self, msg) -> str:
        return self._names.get(msg["parent_header"].get("msg_id"), "?")

    def _take(self, msg) -> None:
        name, kind = self._name(msg), msg["msg_type"]
        if kind == "execute_input":
            self.started.add(name)
        elif kind == "stream":
            self.stdout[name] = self.stdout.get(name, "") + msg["content"]["text"]
        elif kind == "execute_reply":
            self.replies[name] = msg["content"]

    def _pump(self) -> None:
        """Block until the kernel sends on iopub or shell, then take every message waiting."""
        import zmq

        poller = zmq.Poller()
        for channel in (self.kc.iopub_channel, self.kc.shell_channel):
            poller.register(channel.socket, zmq.POLLIN)
        poller.poll()
        for get in (self.kc.get_iopub_msg, self.kc.get_shell_msg):
            while True:
                try:
                    self._take(get(timeout=0))
                except queue.Empty:
                    break

    def run(self, name: str, code: str) -> None:
        self._names[self.kc.execute(code)] = name

    def until(self, predicate) -> None:
        while not predicate():
            self._pump()

    def printed(self, name: str, text: str) -> None:
        self.until(lambda: text in self.stdout.get(name, ""))

    def wait_reply(self, name: str) -> dict:
        self.until(lambda: name in self.replies)
        return self.replies[name]

    def status(self, name: str) -> str:
        return self.wait_reply(name)["status"]

    def stop_button(self) -> None:
        self.km.interrupt_kernel()

    def close(self) -> None:
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)


@pytest.fixture
def kernel():
    k = Kernel()
    yield k
    k.close()


def _start_loop(kernel, code):
    kernel.run("loop", code)
    kernel.printed("loop", "loop polling")


def test_held_back_cell_runs_after_the_loop_ends(kernel):
    _start_loop(kernel, UNTIL_HOLDING)
    kernel.run("after", 'print("after ran")')
    assert kernel.status("loop") == "ok"
    assert kernel.status("after") == "ok"
    assert kernel.stdout.get("after") == "after ran\n"


def test_stop_button_cancels_held_back_cells(kernel):
    _start_loop(kernel, UNTIL_STOPPED)
    kernel.run("after", 'print("after ran")')
    kernel.printed("loop", "loop holds 1")
    kernel.stop_button()
    loop = kernel.wait_reply("loop")
    assert loop["status"] == "error" and loop["ename"] == "KeyboardInterrupt"
    assert "loop ended" in kernel.stdout["loop"]
    assert kernel.status("after") == "aborted"
    kernel.run("next", 'print("next ran")')
    assert kernel.status("next") == "ok"


def test_stop_button_interrupts_a_held_back_cell(kernel):
    _start_loop(kernel, UNTIL_HOLDING)
    kernel.run("slow", 'print("slow sleeping", flush=True)\ntime.sleep(60)')
    kernel.printed("slow", "slow sleeping")
    kernel.stop_button()
    slow = kernel.wait_reply("slow")
    assert slow["status"] == "error" and slow["ename"] == "KeyboardInterrupt", slow
    kernel.run("next", 'print("next ran")')
    assert kernel.status("next") == "ok"


@pytest.mark.parametrize("order", [("rerun", "after"), ("after", "rerun")], ids=["rerun_first", "after_first"])
def test_running_the_loop_cell_again_ends_it(kernel, order):
    _start_loop(kernel, UNTIL_STOPPED)
    for name in order:
        kernel.run(name, UNTIL_STOPPED if name == "rerun" else 'print("after ran")')
    assert kernel.status("loop") == "ok"
    assert kernel.status("rerun") == "aborted"
    assert kernel.status("after") == "ok"
    assert "rerun" not in kernel.started
