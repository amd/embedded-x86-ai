# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Which model sizes the NPU. No NPU and no compile: only the YAML device is read."""

from __future__ import annotations


def test_sizer_skips_whisper_when_it_is_on_cpu(tmp_path):
    """Whisper is first in the list. device cpu leaves the sizer on YOLO-pose."""
    from vla_pipeline.utils.npu_session import sizer_onnx

    whisper = tmp_path / "encoder.onnx"
    pose = tmp_path / "yolo26s-pose.onnx"
    whisper.write_bytes(b"onnx")
    pose.write_bytes(b"onnx")
    cfg = {
        "whisper": {
            "device": "cpu",
            "encoder_onnx": str(whisper),
            "cache_key_encoder": "whisper_base_encoder",
        },
        "yolo_pose": {
            "device": "npu",
            "onnx": str(pose),
            "cache_key": "yolo26s_pose_fp32",
        },
    }
    assert sizer_onnx(cfg) == pose
