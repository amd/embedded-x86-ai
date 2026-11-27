# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Compile the NPU models (YOLO pose and detect) for the Ryzen AI NPU.

This MUST run under the FULL Ryzen AI SDK environment (the venv created by
``install_ryzen_ai.sh``), which has the AIE/vaiml compiler. The deployment-only
``voe`` runtime in the pipeline's own ``.venv`` can RUN compiled models but
cannot compile them ("Model compilation is not supported in a deployment only
installation").

It creates a ``VitisAIExecutionProvider`` session for each model with the same
``config_file`` / ``cache_dir`` / ``cache_key`` the pipeline uses, then runs one
inference to force compilation. The resulting artifacts land in ``cache/`` and
are portable: afterwards the pipeline's ``.venv`` loads them with the deployment
runtime - no recompile, no SDK needed at runtime.

Models compiled (each keyed separately in cache/):
  - yolo26s-pose                 (config: yolo_pose.*)
  - yolo26s-detect               (config: yolo_detect.*) - voice pick-and-place

Usage (from the repo root):

    export RYZEN_AI_VENV=$HOME/ryzen_ai-1.7.1/venv
    scripts/compile_npu_models.sh
    scripts/compile_npu_models.sh --only yolo_detect

The cache_dir/keys/configs are read straight from config/pipeline.yaml so they
always match what the pipeline expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path) -> Path:
    """Resolve p to an absolute path, relative to the repo root if not already absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _compile_one(
    name: str, onnx_path: str, config_file: str, cache_dir: str, cache_key: str
) -> None:
    """Compile one ONNX model for the NPU via the VitisAI EP and verify the cache artifact landed."""
    print(f"\n=== Compiling {name} ===")
    print(f"  onnx:       {onnx_path}")
    print(f"  config:     {config_file}")
    print(f"  cache_dir:  {cache_dir}")
    print(f"  cache_key:  {cache_key}")
    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise SystemExit(
            "VitisAIExecutionProvider not available - are you in the SDK venv?"
        )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    providers = [
        (
            "VitisAIExecutionProvider",
            {
                "config_file": config_file,
                "cache_dir": cache_dir,
                "cache_key": cache_key,
            },
        )
    ]
    print("  building session (this triggers NPU compilation - can take 15-45 min)...")
    sess = ort.InferenceSession(onnx_path, providers=providers)

    # CRITICAL: if the VitisAI EP failed to load, ORT silently falls back to
    # CPU and NO NPU COMPILATION HAPPENS - but the session still "works". Detect
    # that and fail loudly, otherwise we cache nothing and the deployment runtime
    # later reports "deployment only" because there's no compiled artifact.
    active = sess.get_providers()
    if "VitisAIExecutionProvider" not in active:
        raise SystemExit(
            f"\nERROR: VitisAIExecutionProvider did NOT load for {name} - "
            f"active providers: {active}.\n"
            "The compile fell back to CPU and produced NO NPU artifacts.\n"
            "Re-run scripts/compile_npu_models.sh from the Ryzen AI 1.7.1 venv\n"
            "(export RYZEN_AI_WHEELS=$HOME/ryzen_ai-1.7.1)."
        )

    # One inference to make sure the compiled graph is exercised/finalized.
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }.get(inp.type, np.float32)
        feeds[inp.name] = np.zeros(shape, dtype=dtype)
    try:
        sess.run(None, feeds)
        print(f"  [ok] {name} compiled and ran a probe inference.")
    except Exception as e:
        # Compilation may still have succeeded even if the dummy probe shapes
        # aren't ideal; the cache is what matters.
        print(f"  [warn] probe inference failed ({e}); cache may still be valid.")

    # Confirm an artifact actually landed in the cache.
    art_dir = Path(cache_dir) / cache_key
    if not (art_dir.exists() and any(art_dir.iterdir())):
        raise SystemExit(
            f"\nERROR: no compiled artifact in {art_dir} after {name} - "
            "the NPU compile did not produce a cache. Check the log above."
        )
    print(f"  cache: {art_dir}")


def _compile_yolo_model(cfg: dict, section: str, name: str, export_hint: str) -> None:
    """Compile a YOLO ONNX model from the given config section, skipping if absent."""
    y = cfg.get(section)
    if not y:
        print(
            f"\n[skip] config section '{section}' not present - nothing to compile for {name}."
        )
        return
    onnx_path = _resolve(y["onnx"])
    if not onnx_path.exists():
        print(
            f"\n[skip] {name} ONNX not found at {onnx_path} - "
            f"run {export_hint} first."
        )
        return
    cache_dir = str(_resolve(cfg.get("system", {}).get("cache_dir", "cache")))
    vai_cfg = str(_resolve(y.get("vitisai_config", "config/vaiep_config.json")))
    _compile_one(name, str(onnx_path), vai_cfg, cache_dir, y["cache_key"])


def _compile_yolo_pose(cfg: dict) -> None:
    """Compile the YOLOv26s-pose model."""
    _compile_yolo_model(
        cfg, "yolo_pose", "yolo26s-pose", "scripts/export_yolo26s_pose.py"
    )


def _compile_yolo_detect(cfg: dict) -> None:
    """Compile the YOLOv26s-detect model."""
    _compile_yolo_model(
        cfg, "yolo_detect", "yolo26s-detect", "scripts/export_yolo26s_detect.py"
    )


def main() -> None:
    """Parse CLI args and compile the selected NPU model families."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "pipeline.yaml"))
    ap.add_argument(
        "--only",
        choices=["yolo", "yolo_pose", "yolo_detect"],
        default=None,
        help="compile just one model family (default: both YOLO models). "
        "'yolo' = both pose and detect.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))

    if args.only in (None, "yolo", "yolo_pose"):
        _compile_yolo_pose(cfg)
    if args.only in (None, "yolo", "yolo_detect"):
        _compile_yolo_detect(cfg)

    cache_dir = str(_resolve(cfg.get("system", {}).get("cache_dir", "cache")))
    print(f"\nDone. Compiled artifacts are in: {cache_dir}")
    print("You can now run the pipeline from the deployment .venv with device: npu.")


if __name__ == "__main__":
    main()
