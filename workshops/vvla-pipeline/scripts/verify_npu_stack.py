#!/usr/bin/env python3
"""Verify that the Ryzen AI deployment stack can load and use VitisAI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, errors: list[str]) -> None:
    print(f"  [FAIL] {message}")
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="return nonzero on any required check")
    args = parser.parse_args()
    errors: list[str] = []

    if Path("/dev/accel/accel0").exists():
        print("  [ok] XDNA device: /dev/accel/accel0")
    else:
        fail("XDNA device /dev/accel/accel0 is missing", errors)

    xrt = os.environ.get("XILINX_XRT")
    if xrt and Path(xrt, "lib").is_dir():
        print(f"  [ok] XILINX_XRT={xrt}")
    else:
        fail("XILINX_XRT is not configured", errors)

    firmware = os.environ.get("XLNX_VART_FIRMWARE", "")
    if firmware and Path(firmware).is_file():
        print(f"  [ok] firmware: {firmware}")
    else:
        shown = firmware or "(unset)"
        fail(f"XLNX_VART_FIRMWARE does not name an existing xclbin ({shown})", errors)

    try:
        import onnxruntime as ort
    except Exception as exc:
        fail(f"onnxruntime import failed: {exc}", errors)
        return 1

    available = ort.get_available_providers()
    print(f"  providers: {available}")
    if "VitisAIExecutionProvider" not in available:
        fail("VitisAIExecutionProvider is not registered", errors)
    else:
        print("  [ok] VitisAIExecutionProvider is registered")

    if errors:
        print(f"NPU preflight failed ({len(errors)} problem(s)).")
        return 1 if args.preflight else 0
    print("NPU preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
