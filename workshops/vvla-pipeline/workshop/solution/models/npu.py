# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The NPU session factory - the centerpiece of the workshop.

Every ONNX model this pipeline runs on the XDNA2 NPU goes through ONE
function: :func:`build_npu_session`. Vision at camera rate lives or dies on
getting this right, and the two classic Ryzen AI failure modes both start
here:

1. **Silent CPU fallback.** Requesting the NPU is not the same as getting
   it. Always verify which execution provider a session *actually* got
   (:func:`active_provider`) - a model that quietly landed on CPU still
   works, just 10-30× slower, and you won't know unless you look.
2. **The NPU context budget.** The XDNA2 partition holds a limited number of
   concurrent hardware contexts. Both YOLO models fit together; adding a
   third/fourth model (e.g. Whisper encoder+decoder) can fail with
   ``HW context creation unsuccessful / sub buffer size and offset``. That is
   NOT a stale cache - move a model to CPU (``device: cpu`` in the config)
   and read the startup logs to see what actually fit.

Compiled NPU artifacts are keyed by ``cache_dir + cache_key`` and are
portable: the first compile can take tens of minutes (done once, from the
Ryzen AI SDK venv), then every load is ~1 s from the cache. Each model MUST
use a unique ``cache_key`` or the entries collide.

Notebook: ``notebooks/01_ai_models/01_npu_yolo.ipynb``
Self-test: ``python -m models.selftest``
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from common.config import resolve

logger = logging.getLogger(__name__)


def npu_available() -> bool:
    """True if this onnxruntime build exposes the VitisAI execution provider.

    The stock ``pip install onnxruntime`` does NOT include it - only the
    Ryzen AI build (installed from the Ryzen AI wheel directory) does.
    """
    # >>> TODO 1.1: npu_available - notebooks/01_ai_models/01_npu_yolo.ipynb
    return "VitisAIExecutionProvider" in ort.get_available_providers()
    # <<< TODO 1.1


def build_npu_session(
    onnx_path,
    device: str = "npu",
    *,
    vitisai_config="config/vaiep_config.json",
    cache_dir="../../cache",
    cache_key: str = "model",
) -> ort.InferenceSession:
    """Create an ``InferenceSession`` on the NPU (VitisAI EP) or the CPU.

    Args:
        onnx_path: Path to the FP32 ONNX model (resolved against the project
            root).
        device: ``"npu"`` or ``"cpu"``. ``"npu"`` must fall back to CPU (with
            a warning) when the VitisAI EP is not present, so every component
            still runs on machines without the Ryzen AI stack.
        vitisai_config: VitisAI compiler pass configuration (JSON) - selects
            the VAIML partitioning pass and BF16 conversion.
        cache_dir: NPU compile cache directory (shared with the real
            pipeline so nothing recompiles).
        cache_key: Unique cache key for THIS model. Two models sharing a key
            corrupt each other's compiled artifacts.

    Returns:
        A ready ``onnxruntime.InferenceSession``.
    """
    onnx_path = resolve(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found - export and compile the YOLO models (see "
            "workshops/vvla-pipeline/README.md) or point the config at your model files."
        )

    # >>> TODO 1.2: build_npu_session - notebooks/01_ai_models/01_npu_yolo.ipynb
    if device == "npu" and not npu_available():
        logger.warning(
            "VitisAIExecutionProvider not available in this onnxruntime build "
            "- falling back to CPU. Install the Ryzen AI onnxruntime wheel "
            "for NPU execution."
        )
        device = "cpu"

    if device == "cpu":
        return ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])

    cache = resolve(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building NPU session for %s (cache hit ~1s; a cold compile takes "
        "30-60 min and belongs in the SDK venv)", Path(onnx_path).name,
    )
    return ort.InferenceSession(
        str(onnx_path),
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "config_file": str(resolve(vitisai_config)),
            "cache_dir": str(cache),
            "cache_key": cache_key,
            "target": "VAIML",
        }],
    )
    # <<< TODO 1.2


def active_provider(session: ort.InferenceSession) -> str:
    """The execution provider this session ACTUALLY got (first = active).

    Requesting the NPU is a wish; this is the receipt. Check it after every
    session build - a silent CPU fallback looks identical until you profile.
    """
    # >>> TODO 1.3: active_provider - notebooks/01_ai_models/01_npu_yolo.ipynb
    return session.get_providers()[0]
    # <<< TODO 1.3


def report_placement(session: ort.InferenceSession, requested: str,
                     name: str = "model") -> str:
    """PROVIDED: log where a model landed vs. where you asked it to go."""
    got = active_provider(session)
    on_npu = got == "VitisAIExecutionProvider"
    if requested == "npu" and not on_npu:
        logger.warning("%s requested NPU but is running on %s - check the "
                       "Ryzen AI wheel install and the NPU context budget.",
                       name, got)
    else:
        logger.info("%s running on %s", name, got)
    return got
