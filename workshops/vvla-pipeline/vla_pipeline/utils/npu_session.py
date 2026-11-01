# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ONNX Runtime session factory for the Ryzen AI NPU (VitisAIExecutionProvider).

YOLO-pose and YOLO-detect build their NPU sessions through
:func:`build_session` so the VitisAI compiler config, cache directory, and
provider verification live in exactly one place.

Notes for Ryzen AI SW 1.7.1 on Linux (Strix Halo / XDNA2):
- ``onnxruntime`` must be the Ryzen AI build that ships
  ``VitisAIExecutionProvider`` (installed by ``bootstrap.sh`` from the
  Ryzen AI wheel directory, see ``RYZEN_AI_WHEELS``).
- The first compile of a model for the NPU can take tens of minutes; the
  result is cached under ``cache_dir/cache_key`` and reloads in ~1 s.
- The two YOLO sessions share buffers within a process, created by its
  first NPU session and freed when the last one closes. YOLO-pose opened
  while YOLO-detect is open fails with ``HW context creation
  unsuccessful`` (``sub buffer size and offset``). :func:`size_npu` opens
  YOLO-pose first and keeps that session.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)

# First entry whose device is npu is opened and held. (config section, ONNX
# key, VitisAI config key, cache key, VitisAI target.)
_BY_SIZE = (
    ("whisper", "encoder_onnx", "encoder_vitisai_config", "cache_key_encoder", None),
    ("whisper", "decoder_onnx", "decoder_vitisai_config", "cache_key_decoder", None),
    ("yolo_pose", "onnx", "vitisai_config", "cache_key", "VAIML"),
    ("yolo_detect", "onnx", "vitisai_config", "cache_key", "VAIML"),
)
_npu_sized = False
_sizer: ort.InferenceSession | None = None  # holds the NPU buffers for the life of the process


def npu_available() -> bool:
    """True if this onnxruntime build exposes the VitisAI EP."""
    return "VitisAIExecutionProvider" in ort.get_available_providers()


def active_provider(session: ort.InferenceSession) -> str:
    """Return the highest-priority provider active on a session."""
    providers = session.get_providers()
    return providers[0] if providers else "none"


def sizer_onnx(cfg: dict) -> Path | None:
    """ONNX file of the first model whose own YAML ``device`` is ``npu``.

    Whisper is first in the list and the shipped config leaves it on the CPU,
    so this is the YOLO-pose file.
    """
    found = _sizer_choice(cfg)
    return None if found is None else found[0]


def _sizer_choice(cfg: dict):
    """``(onnx, config file, cache dir, cache key, target)`` for :func:`sizer_onnx`, or None."""
    for section, onnx_key, config_key, cache_key, target in _BY_SIZE:
        s = cfg.get(section) or {}
        if s.get("device") != "npu" or not s.get(onnx_key) or not s.get(cache_key):
            continue
        onnx_path = resolve(s[onnx_key])
        if not onnx_path.exists():
            continue
        cache_dir = str(resolve(s.get("cache_dir") or cfg.get("system", {}).get("cache_dir", "cache")))
        return onnx_path, str(resolve(s.get(config_key, "config/vaiep_config.json"))), cache_dir, s[cache_key], target
    return None


def size_npu(cfg: dict) -> None:
    """Open :func:`sizer_onnx` and keep that session, once per process.

    Call it before any other NPU session. The buffers a process's models share
    live as long as this session does.
    """
    global _npu_sized, _sizer
    if _npu_sized or not npu_available():
        return
    choice = _sizer_choice(cfg)
    if choice is None:
        return
    onnx_path, config_file, cache_dir, cache_key, target = choice
    _npu_sized = True
    options = {"config_file": config_file, "cache_dir": cache_dir, "cache_key": cache_key}
    if target:
        options["target"] = target
    try:
        _sizer = ort.InferenceSession(str(onnx_path), providers=["VitisAIExecutionProvider"],
                                      provider_options=[options])
        logger.info("NPU buffers sized by %s", onnx_path.name)
    except Exception as e:
        logger.warning("Could not size the NPU buffers with %s (%s)", onnx_path.name, e)


def build_session(
    onnx_path: str | Path,
    device: str = "npu",
    *,
    vitisai_config: str | Path = "config/vaiep_config.json",
    cache_dir: str | Path = "cache",
    cache_key: str = "model",
) -> ort.InferenceSession:
    """Create an InferenceSession on the NPU (VitisAI EP) or CPU.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        device: ``"npu"`` or ``"cpu"``. An NPU request falls back to CPU with
            a warning when VitisAI is unavailable or fails to load.
        vitisai_config: VitisAI compiler pass configuration (JSON).
        cache_dir: NPU compile cache directory.
        cache_key: Unique cache key per model.
    """
    onnx_path = resolve(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found - run ./bootstrap.sh (model export step) first."
        )

    if device == "npu" and not npu_available():
        logger.warning(
            "VitisAIExecutionProvider not available in this onnxruntime build - "
            "running %s on CPU. Install the Ryzen AI onnxruntime-vitisai wheel "
            "(see bootstrap.sh / RYZEN_AI_WHEELS) to use the NPU.",
            onnx_path.name,
        )
        device = "cpu"

    if device == "cpu":
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    cache = resolve(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building NPU session for %s (cache hit ~1s, first compile can take 30-60 min)",
        onnx_path.name,
    )
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=[
                {
                    "config_file": str(resolve(vitisai_config)),
                    "cache_dir": str(cache),
                    "cache_key": cache_key,
                    "target": "VAIML",
                }
            ],
        )
    except Exception as e:
        logger.warning(
            "VitisAI session for %s failed (%s) - running on CPU. "
            "Run scripts/verify_npu_stack.py --preflight.",
            onnx_path.name,
            e,
        )
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if "VitisAIExecutionProvider" not in session.get_providers():
        logger.warning(
            "VitisAI session for %s fell back to %s - running on CPU. "
            "Run scripts/verify_npu_stack.py --preflight.",
            onnx_path.name,
            session.get_providers(),
        )
    logger.info("%s active provider: %s", onnx_path.name, active_provider(session))
    return session
