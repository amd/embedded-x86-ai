# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Export YOLOv26s (detection) to FP32 ONNX for the Ryzen AI NPU.

Mirror of ``scripts/export_yolo26s_pose.py`` for the 80-class detection
checkpoint used by the workshop's object detection (``models/vision.py``).
Safe to re-run (skips if output exists).

    python scripts/export_yolo26s_detect.py [--out models/yolo26s]

Downloads ``yolo26s.pt`` via ultralytics if needed, exports a static batch-1
640×640 ONNX (opset 17, simplified), and validates the graph. YOLO26's
end-to-end head exports as [1, 300, 6] (xyxy, score, class) - no NMS pass at
inference time. The VitisAI EP converts FP32→BF16 at NPU compile time (see
config/vaiep_config.json); first compile is slow, cached under
``cache/yolo26s_detect_fp32`` (config ``yolo_detect.cache_key``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prevent auto-install of onnxruntime triggered by ultralytics.
os.environ["YOLO_AUTOINSTALL"] = "false"

import onnx

PT_MODEL = "yolo26s.pt"
IMG_SIZE = 640


def export_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    """Export the ultralytics checkpoint at pt_path to a static FP32 ONNX at onnx_path."""
    from ultralytics import YOLO

    print(f"Exporting {pt_path} to ONNX FP32...")
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=17,
        simplify=True,
        dynamic=False,
        batch=1,
    )
    default_name = Path(str(pt_path).replace(".pt", ".onnx"))
    if default_name.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        default_name.replace(onnx_path)
    # ultralytics writes yolo26s.pt/.onnx into the CWD; tuck the weights
    # next to the ONNX and remove any stray copy left in the repo root.
    for stray in (Path(PT_MODEL), Path(str(PT_MODEL).replace(".pt", ".onnx"))):
        if stray.exists() and stray.resolve() != onnx_path.resolve():
            dest = onnx_path.parent / stray.name
            try:
                stray.replace(dest) if stray.suffix == ".pt" else stray.unlink()
            except OSError:
                pass
    print("Saved:", onnx_path)


def validate(onnx_path: Path) -> None:
    """Load the ONNX model, run the checker, and print its input/output shapes."""
    print("\nValidating ONNX model...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}  {shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}  {shape}")
    print("ONNX model is VALID")


def main() -> None:
    """Export the YOLOv26s detection checkpoint to ONNX and validate it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/yolo26s")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    onnx_out = out / "yolo26s.onnx"
    if onnx_out.exists() and not args.force:
        print(f"{onnx_out} already present - skipping export (use --force to redo).")
        return

    out.mkdir(parents=True, exist_ok=True)
    pt = out / PT_MODEL
    export_to_onnx(pt if pt.exists() else Path(PT_MODEL), onnx_out)
    validate(onnx_out)
    print("\nDone - next: compile it for the NPU (see the README, 'Compile for the NPU').")


if __name__ == "__main__":
    main()
