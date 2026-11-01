# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``ui_events`` for a cell that keeps running while you use its widgets (PROVIDED).

A drop-in for ``jupyter_ui_poll.ui_events``. While such a cell loops,
jupyter_ui_poll holds back the other cells you run, and runs them once the
loop cell has ended. Here they run the way Jupyter runs queued cells:

* Jupyter's stop button interrupts them. jupyter_ui_poll runs them outside
  ipykernel's own dispatch, where the kernel ignores the stop button.
* If the loop cell ended with an error (the stop button raises
  ``KeyboardInterrupt``), they are cancelled, as after any failed cell.

The loop cell itself, run again while it loops (JupyterLab's "run and
advance" on it, which also clears its widgets), calls ``on_rerun`` and is
cancelled, instead of starting a second loop once this one ends.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from inspect import isawaitable

from IPython import get_ipython
from jupyter_ui_poll._poll import ZMQ_POLLOUT, KernelWrapper


def _same_cell(a: dict, b: dict) -> bool:
    """Two execute requests from the same notebook cell."""
    ida, idb = a.get("metadata", {}).get("cellId"), b.get("metadata", {}).get("cellId")
    if ida and idb:
        return ida == idb
    return a["content"].get("code") == b["content"].get("code")


class _Events(KernelWrapper):
    def __init__(self, shell, loop, on_rerun=None) -> None:
        super().__init__(shell, loop)
        self.on_rerun = on_rerun
        self._cancel = False

    def _execute_request(self, stream, ident, parent):
        super()._execute_request(stream, ident, parent)
        if self.on_rerun is not None and _same_cell(parent, self._original_parent[1]):
            self.on_rerun()

    def _post_run_cell(self, result=None, *args, **kw):
        self._cancel = result is None or not result.success
        super()._post_run_cell(result, *args, **kw)

    async def replay(self):
        kernel = self._kernel
        self.restore()
        sys.stdout.flush()
        sys.stderr.flush()
        shell_stream = getattr(kernel, "shell_stream", None)
        for stream, ident, parent in self._events:
            kernel.set_parent(ident, parent)
            if self._cancel or kernel._aborting or _same_cell(parent, self._original_parent[1]):
                kernel._send_abort_reply(stream, parent, ident)
            else:
                # ipykernel ignores SIGINT between the cells it dispatches; raise it here, as its dispatch does
                previous = signal.signal(signal.SIGINT, signal.default_int_handler)
                try:
                    rr = kernel.execute_request(stream, ident, parent)
                    if isawaitable(rr):
                        await rr
                except KeyboardInterrupt:
                    self._cancel = True
                finally:
                    signal.signal(signal.SIGINT, previous)
                if not self._shell.last_execution_succeeded:
                    self._cancel = True
            sys.stdout.flush()
            sys.stderr.flush()
            if shell_stream is not None:
                kernel._publish_status("idle", "shell")
                shell_stream.flush(ZMQ_POLLOUT)
            else:
                kernel._publish_status("idle")


def ui_events(on_rerun=None):
    """``with ui_events() as poll:`` then ``poll(n)`` in the loop, as with jupyter_ui_poll.

    ``on_rerun`` is called (from the polling thread) when this cell is run
    again while it loops; it should end the loop.
    """
    current = KernelWrapper._current
    if current is None:
        current = KernelWrapper._current = _Events(get_ipython(), asyncio.get_event_loop(), on_rerun)
    elif isinstance(current, _Events) and on_rerun is not None:
        current.on_rerun = on_rerun
    return current
