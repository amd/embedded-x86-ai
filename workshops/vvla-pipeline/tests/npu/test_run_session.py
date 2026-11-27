# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A notebook stop during an ONNX run lands after that run. The pipeline does not defer it."""

from __future__ import annotations

import signal
import threading

import pytest


def test_a_stop_during_a_session_run_lands_after_the_run():
    from vla_pipeline.utils.npu_session import run_session

    calls = []

    class Session:
        def run(self, names, feeds):
            calls.append("entered")
            signal.pthread_kill(threading.get_ident(), signal.SIGINT)
            calls.append("finished")
            return ["ok"]

    with pytest.raises(KeyboardInterrupt):
        run_session(Session(), None, {})
    assert calls == ["entered", "finished"]


def test_the_pipeline_does_not_defer_a_stop():
    """Ctrl-C during a pipeline NPU run is delivered in the run, so the process can die."""
    from vla_pipeline.utils import npu_session

    npu_session.defer_sigint = False
    calls = []

    class Session:
        def run(self, names, feeds):
            calls.append("entered")
            signal.pthread_kill(threading.get_ident(), signal.SIGINT)
            calls.append("finished")
            return ["ok"]

    try:
        with pytest.raises(KeyboardInterrupt):
            npu_session.run_session(Session(), None, {})
        assert calls == ["entered"]
    finally:
        npu_session.defer_sigint = True
