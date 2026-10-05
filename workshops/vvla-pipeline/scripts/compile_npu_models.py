# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Compile the workshop's NPU models (YOLO pose/detect) for the Ryzen AI NPU.

This MUST run under the FULL Ryzen AI SDK environment (the venv created by
``install_ryzen_ai.sh``), which has the AIE/vaiml compiler. The deployment-only
``voe`` runtime in the pipeline's own ``.venv`` can RUN compiled models but
cannot compile them ("Model compilation is not supported in a deployment only
installation").

It creates a ``VitisAIExecutionProvider`` session for each model with the same
``config_file`` / ``cache_dir`` / ``cache_key`` the workshop uses, then runs one
inference to force compilation. The resulting artifacts land in ``cache/`` and
are portable: afterwards the workshop's ``.venv`` loads them with the deployment
runtime - no recompile, no SDK needed at runtime.

Models compiled (each keyed separately in cache/):
  - yolo26s-pose                 (config: yolo_pose.*)
  - yolo26s-detect               (config: yolo_detect.*)

Usage (from the repo root, with the SDK venv activated):

    source ./ryzenai-compile/bin/activate           # the FULL SDK venv
    python scripts/compile_npu_models.py             # both; reads workshop.yaml
    python scripts/compile_npu_models.py --only yolo_detect

The cache_dir/keys/configs are read straight from
workshop/solution/config/workshop.yaml so they always match what the workshop
expects. Relative paths in it resolve against workshop/solution/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "workshop" / "solution" / "config" / "workshop.yaml"
# Relative paths in the config resolve against the project root (the directory
# that contains config/). Set in main() once the config path is known.
PROJECT_ROOT = DEFAULT_CONFIG.parents[1]


def _setup_npu_env() -> None:
    """Set XLNX_VART_FIRMWARE and LD_LIBRARY_PATH from THIS interpreter's venv.

    The compile runs under the full SDK venv; the EP needs its native libs on
    the loader path and a specific .xclbin FILE (not a directory). We configure
    both from the running interpreter so the script is self-contained no matter
    how it's launched. Must run before onnxruntime touches the EP.
    """
    import os
    import sys
    import glob

    sp = (
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )

    # Native runtime libs.
    lib_dirs = [
        sp / "voe" / "lib",
        sp / "flexmlrt" / "lib",
        sp / "onnxruntime" / "capi",
    ]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    before = cur
    for d in lib_dirs:
        if d.is_dir() and str(d) not in cur:
            cur = f"{d}{':' + cur if cur else ''}"
    os.environ["LD_LIBRARY_PATH"] = cur

    # NPU firmware: a specific xclbin file. Strix Halo -> 2x4x4.
    if not os.environ.get("XLNX_VART_FIRMWARE"):
        override = os.environ.get("RAI_XCLBIN")
        xclbin_dir = sp / "flexml" / "flexml_extras" / "data" / "ryzen-ai" / "stx"
        chosen = None
        if override and Path(override).is_file():
            chosen = override
        else:
            for name in ("unified-2x4x4.xclbin", "unified-4x4.xclbin"):
                cand = xclbin_dir / name
                if cand.is_file():
                    chosen = str(cand)
                    break
            if not chosen:
                hits = sorted(glob.glob(str(xclbin_dir / "*.xclbin")))
                chosen = hits[0] if hits else None
        if chosen:
            os.environ["XLNX_VART_FIRMWARE"] = chosen
            print(f"  XLNX_VART_FIRMWARE = {chosen}")

    # If we extended LD_LIBRARY_PATH, re-exec once so the dynamic linker uses it
    # (it's read at process start; mutating os.environ afterward isn't enough for
    # libs loaded via DT_NEEDED). Guard with a sentinel to avoid a loop.
    if cur != before and os.environ.get("_RAI_REEXEC") != "1":
        os.environ["_RAI_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_setup_npu_env()


def _ensure_native_libs_on_path() -> None:
    """Put this venv's VitisAI native libs on LD_LIBRARY_PATH, then re-exec once.

    The VitisAI EP needs voe/lib, flexmlrt/lib, etc. on the loader path. They
    live in the active (SDK) venv's site-packages but aren't auto-added. We set
    them and re-exec the interpreter once (guarded by an env flag) so the new
    LD_LIBRARY_PATH is in effect before onnxruntime is imported.
    """
    if os.environ.get("_COMPILE_NPU_REEXEC") == "1":
        return
    import sysconfig

    sp = sysconfig.get_paths()["purelib"]
    libs = [
        os.path.join(sp, "voe", "lib"),
        os.path.join(sp, "flexmlrt", "lib"),
        os.path.join(sp, "onnxruntime", "capi"),
        os.path.join(sp, "flexml", "lib"),
    ]
    libs = [d for d in libs if os.path.isdir(d)]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    new = ":".join(libs + ([cur] if cur else []))
    # XLNX_VART_FIRMWARE must be a specific .xclbin FILE, not the directory.
    # Strix (STX) ships unified-*.xclbin; prefer the larger 4x4 partition.
    xclbin_dir = os.path.join(sp, "flexml", "flexml_extras", "data", "ryzen-ai", "stx")
    xclbin_file = ""
    if os.path.isdir(xclbin_dir):
        prefer = ["unified-4x4.xclbin", "unified-2x4x4.xclbin"]
        for name in prefer:
            cand = os.path.join(xclbin_dir, name)
            if os.path.isfile(cand):
                xclbin_file = cand
                break
        if not xclbin_file:  # fall back to any .xclbin present
            import glob

            hits = sorted(glob.glob(os.path.join(xclbin_dir, "*.xclbin")))
            xclbin_file = hits[0] if hits else ""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = new
    env["_COMPILE_NPU_REEXEC"] = "1"
    if xclbin_file:
        env["XLNX_VART_FIRMWARE"] = xclbin_file
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_ensure_native_libs_on_path()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path) -> Path:
    """Resolve p to an absolute path, relative to the project root if not already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


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
            "The compile venv's native libs (voe/lib, flexmlrt/lib) must be on\n"
            "LD_LIBRARY_PATH. Re-run the compile with the env sourced for the\n"
            "SDK venv (see scripts/compile_npu_models.py header)."
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
    """Parse CLI args and compile the selected NPU models."""
    global PROJECT_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument(
        "--only",
        choices=["yolo", "yolo_pose", "yolo_detect"],
        default=None,
        help="compile just one model (default: both). 'yolo' = both pose and detect.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    PROJECT_ROOT = config_path.parents[1]
    cfg = _load_yaml(config_path)

    if args.only in (None, "yolo", "yolo_pose"):
        _compile_yolo_pose(cfg)
    if args.only in (None, "yolo", "yolo_detect"):
        _compile_yolo_detect(cfg)

    cache_dir = str(_resolve(cfg.get("system", {}).get("cache_dir", "cache")))
    print(f"\nDone. Compiled artifacts are in: {cache_dir}")
    print("You can now run the workshop from the deployment .venv with device: npu.")


if __name__ == "__main__":
    main()
