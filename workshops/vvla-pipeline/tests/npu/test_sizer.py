# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Which model sizes the NPU. No NPU and no compile: only the YAML device is read."""

from __future__ import annotations


def _cfg(tmp_path, pose_device: str, detect_device: str):
    pose = tmp_path / "yolo26s-pose.onnx"
    detect = tmp_path / "yolo26s-detect.onnx"
    pose.write_bytes(b"onnx")
    detect.write_bytes(b"onnx")
    cfg = {
        "yolo_pose": {"device": pose_device, "onnx": str(pose), "cache_key": "yolo26s_pose_fp32"},
        "yolo_detect": {"device": detect_device, "onnx": str(detect), "cache_key": "yolo26s_detect_fp32"},
    }
    return cfg, pose, detect


def test_sizer_is_yolo_pose_when_both_models_are_on_the_npu(tmp_path):
    """YOLO-pose is first in the list, so it opens first and holds the NPU buffers."""
    from vla_pipeline.utils.npu_session import sizer_onnx

    cfg, pose, _ = _cfg(tmp_path, "npu", "npu")
    assert sizer_onnx(cfg) == pose


def test_sizer_is_yolo_detect_when_pose_is_on_the_cpu(tmp_path):
    """A model whose own device is not npu is skipped; the next one sizes the NPU."""
    from vla_pipeline.utils.npu_session import sizer_onnx

    cfg, _, detect = _cfg(tmp_path, "cpu", "npu")
    assert sizer_onnx(cfg) == detect
