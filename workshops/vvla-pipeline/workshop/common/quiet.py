# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Benign-noise hushing (PROVIDED) - every notebook calls hush_benign_warnings().

The workshop stack is chatty in ways that LOOK like problems but aren't:
MediaPipe's C++ glog/absl init lines, Mesa's libEGL DRI3 probe warnings,
onnxruntime's "some nodes were not assigned to the preferred execution
provider" (expected when the VitisAI EP partitions a graph), and a
dual-matplotlib Axes3D import quirk. Real errors still come through - only
the known-benign chatter is filtered.
"""

from __future__ import annotations

import contextlib
import os
import warnings


def hush_benign_warnings() -> None:
    """Call once, first thing in a notebook (before the heavy imports)."""
    # C++ glog chatter (MediaPipe calculators) - must be set before the
    # native library loads.
    os.environ.setdefault("GLOG_minloglevel", "2")
    # Mesa's libEGL "DRI3 error / Ensure your X server supports DRI3".
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    # Dual matplotlib installs (system + pip) trip this harmless warning.
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    # webrtcvad still imports pkg_resources; setuptools warns on that import.
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    # onnxruntime: ERROR+ only - "some nodes on CPU" is expected when the
    # VitisAI EP partitions a graph.
    try:
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
    except ImportError:
        pass


@contextlib.contextmanager
def quiet_stderr():
    """Silence C++-level stderr chatter (absl / EGL / TFLite init lines)
    that Python warning filters can't reach - OS-level, so use it only
    around known-noisy calls."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class QuietHands:
    """Delegates to a hand tracker, silencing the MediaPipe C++ log lines
    that fire at graph init and on first detections."""

    def __init__(self, inner):
        self._inner = inner

    def process(self, frame):
        """Process one frame with C++ stderr chatter silenced."""
        with quiet_stderr():
            return self._inner.process(frame)

    def draw(self, frame, obs, results):
        """Delegate the overlay draw to the wrapped tracker."""
        return self._inner.draw(frame, obs, results)

    def close(self):
        """Delegate close to the wrapped tracker."""
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)
