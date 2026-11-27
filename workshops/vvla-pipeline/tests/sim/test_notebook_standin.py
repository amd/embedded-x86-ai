# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The notebook test's webcam is the generated frame, read in this process.

Replacing ``cv2.VideoCapture`` does nothing while the mount camera still
subscribes to ``/cameras/mount/image_raw``. That topic stays empty when
``/dev/video0`` is missing, and mimic then waits out the 900 second hang guard
without a frame.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from notebook_harness import RIG_STANDINS, apply_notebook_standins
from vla_pipeline.main import apply_sim_overrides
from vla_pipeline.vision.camera import DirectCamera, Ros2Camera, make_camera

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]


class _Frame:
    def __init__(self, device):
        self.device = device

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        return None


def _workshop_cameras() -> dict:
    return {
        "robot": {"backend": "real"},
        "cameras": {
            "mount": {
                "device": "/dev/video0",
                "width": 640,
                "height": 480,
                "fps": 30,
                "source": "device",
                "use_ros2": True,
                "topic": "/cameras/mount/image_raw",
            },
            "arm": {
                "source": "device",
                "use_ros2": True,
                "topic": "/cameras/arm/image_raw",
            },
        },
    }


def test_standin_cell_carries_the_one_standin_function_as_it_is():
    """The kernel cannot import the harness (site-packages has its own ``tests``)."""
    compile(RIG_STANDINS, "RIG_STANDINS", "exec")
    assert inspect.getsource(apply_notebook_standins) in RIG_STANDINS
    assert "@@" not in RIG_STANDINS
    assert "import tests" not in RIG_STANDINS and "from tests" not in RIG_STANDINS
    assert 'cfg["cameras"]["mount"]' not in RIG_STANDINS


def test_mount_reads_the_generated_frame_and_the_arm_stays_on_ros(monkeypatch):
    cfg = apply_notebook_standins(_workshop_cameras())
    apply_sim_overrides(cfg)

    assert cfg["robot"]["backend"] == "sim"
    assert cfg["cameras"]["mount"]["use_ros2"] is False
    assert cfg["cameras"]["mount"]["source"] == "device"
    assert cfg["cameras"]["arm"]["use_ros2"] is True
    assert cfg["cameras"]["arm"]["source"] == "sim"

    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", _Frame)
    cam = make_camera(cfg, "mount")
    try:
        assert isinstance(cam, DirectCamera)
        assert not isinstance(cam, Ros2Camera)
        ok, frame = cam.read()
    finally:
        cam.close()
    assert ok
    assert frame.shape == (480, 640, 3)
