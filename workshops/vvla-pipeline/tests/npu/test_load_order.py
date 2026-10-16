# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLO-pose and YOLO-detect land on the NPU when the detector is built first.

    source scripts/ryzen_ai_env.sh && pytest tests/npu

The NPU buffers a process's models share live as long as it keeps an NPU
session open, so each case runs in a fresh process. Needs the Ryzen AI runtime and
the compiled YOLO models in ``cache/``; skipped without them, and failed without
them when ``VVLA_TEST_STRICT=1`` (``run_notebooks.sh --test``). The Whisper case
also needs Whisper's NPU cache, which the README's install does not build.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKSHOP = ROOT / "workshop" / "solution"
STRICT = os.environ.get("VVLA_TEST_STRICT") == "1"
YOLO_CACHES = ("yolo26s_pose_fp32", "yolo26s_detect_fp32")
WHISPER_CACHES = ("whisper_base_encoder", "whisper_base_decoder")


def _need(caches, strict: bool = STRICT) -> None:
    """Skip (or, when ``strict``, fail) unless the VitisAI EP and ``caches`` are here."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except ImportError:
        providers = []
    # The provider is listed even without the runtime; without the firmware
    # path ryzen_ai_env.sh exports, its sessions run on the CPU.
    if "VitisAIExecutionProvider" not in providers or not os.environ.get("XLNX_VART_FIRMWARE"):
        reason = "no NPU runtime: source scripts/ryzen_ai_env.sh"
    else:
        absent = [key for key in caches if not (ROOT / "cache" / key).is_dir()]
        reason = f"no compiled NPU model in cache/ for {', '.join(absent)}" if absent else None
    if reason:
        (pytest.fail if strict else pytest.skip)(reason)


def _providers(code: str, cwd: Path, path: Path) -> dict:
    """Run ``code`` in a fresh interpreter; it prints a JSON dict of providers."""
    out = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(path)!r})\n{code}"],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_pipeline_detector_before_pose():
    _need(YOLO_CACHES)
    got = _providers(
        "import json\n"
        "from vla_pipeline.utils.config import load_config\n"
        "from vla_pipeline.vision.yolo_detect_npu import ObjectDetector\n"
        "from vla_pipeline.vision.yolo_pose_npu import PoseEstimator\n"
        "cfg = load_config()\n"
        "det = ObjectDetector(cfg, device='npu')\n"
        "pose = PoseEstimator(cfg, device='npu')\n"
        "print(json.dumps({'detect': det.session.get_providers()[0], 'pose': pose.session.get_providers()[0]}))",
        ROOT, ROOT,
    )
    assert got == {"detect": "VitisAIExecutionProvider", "pose": "VitisAIExecutionProvider"}


def test_pipeline_whisper_after_both_yolos():
    _need(YOLO_CACHES)
    _need(WHISPER_CACHES, strict=False)
    got = _providers(
        "import json\n"
        "from vla_pipeline.utils.config import load_config\n"
        "from vla_pipeline.vision.yolo_pose_npu import PoseEstimator\n"
        "from vla_pipeline.vision.yolo_detect_npu import ObjectDetector\n"
        "from vla_pipeline.audio.whisper_npu import WhisperNPU\n"
        "cfg = load_config()\n"
        "cfg['whisper']['device'] = 'npu'\n"
        "pose, det = PoseEstimator(cfg), ObjectDetector(cfg)\n"  # main.py's order
        "w = WhisperNPU(cfg)\n"
        "print(json.dumps({'pose': pose.session.get_providers()[0], 'detect': det.session.get_providers()[0],\n"
        "                  'encoder': w.model.encoder.get_providers()[0], 'decoder': w.model.decoder.get_providers()[0]}))",
        ROOT, ROOT,
    )
    assert set(got.values()) == {"VitisAIExecutionProvider"}, got


def test_workshop_detector_before_pose():
    _need(YOLO_CACHES)
    got = _providers(
        "import json\n"
        "from common.config import load_config\n"
        "from models.vision import ObjectDetector, PoseEstimator\n"
        "cfg = load_config()\n"
        "det = ObjectDetector(cfg, device='npu')\n"
        "pose = PoseEstimator(cfg, device='npu')\n"
        "print(json.dumps({'detect': det.provider, 'pose': pose.provider}))",
        WORKSHOP, WORKSHOP,
    )
    assert got == {"detect": "VitisAIExecutionProvider", "pose": "VitisAIExecutionProvider"}
