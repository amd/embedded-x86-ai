# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""How many NPU AI Engines a running test has reserved, and how many are working.

The driver query ``DRM_AMDXDNA_QUERY_AIE_METADATA`` is the count of each tile
kind. On this machine that is 4 AI-engine rows, 1 memory-tile row and 1 shim
row, across 8 columns: 32 AI Engines and 8 memory tiles. ``xrt-smi`` prints
that same grid as topology ``6x8``; the 6 is every tile row, not 6 engines.

Which columns a context holds still comes from ``xrt-smi examine -r
aie-partitions``. An engine counts as busy when that context's
``command_completions`` counter moved since the previous report. The counter
is cumulative, so the first report has no busy count.
"""

from __future__ import annotations

import ctypes
import fcntl
import glob
import json
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# xrt-smi examine is one report per invocation. This is the time between
# reports. The stage ending closes stdin, and that close is the stop event.
SAMPLE_S = 1.0

_TOPOLOGY = re.compile(r"^(\d+)x(\d+)$")


def find_xrt_smi() -> str | None:
    """The xrt-smi binary, including the path XRT's setup.sh adds."""
    for cand in ("xrt-smi", "/opt/xilinx/xrt/bin/xrt-smi", "/usr/bin/xrt-smi"):
        found = shutil.which(cand)
        if found:
            return found
    return None


def _as_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def topology_from_text(text: str) -> tuple[int, int] | None:
    """``(rows, columns)`` from the device table, or None.

    The table ends in the topology cell, for example ``|aie2p|6x8|``. The
    second number is the column count ``Total Columns`` also reports.
    """
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        match = _TOPOLOGY.fullmatch(cells[-1])
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def total_columns(platform: dict | None) -> int | None:
    """``total_columns`` from an ``examine -r platform`` JSON report."""
    if not platform:
        return None
    for dev in platform.get("devices") or []:
        for item in dev.get("platforms") or []:
            region = item.get("static_region") or {}
            count = _as_int(region.get("total_columns"))
            if count is not None:
                return count
    return None


def partitions_from_json(data: dict | None) -> list[dict]:
    """Partitions in an ``examine -r aie-partitions`` JSON report.

    Each item is ``start_col``, ``num_cols``, and ``contexts`` of
    ``pid`` / ``slot`` / ``completions``. An empty report uses ``""`` for
    ``partitions``; that is no partition, not an error.
    """
    found = []
    if not data:
        return found
    for dev in data.get("devices") or []:
        raw = (dev.get("aie_partitions") or {}).get("partitions")
        if not isinstance(raw, list):
            continue
        for part in raw:
            contexts = []
            hw = part.get("hw_contexts") or []
            if isinstance(hw, dict):
                hw = [hw]
            if not isinstance(hw, list):
                hw = []
            for ctx in hw:
                contexts.append({
                    "pid": _as_int(ctx.get("pid", ctx.get("process_id"))),
                    "slot": _as_int(ctx.get("slot_index", ctx.get("context_id", ctx.get("id")))),
                    "completions": _as_int(ctx.get("command_completions")),
                })
            found.append({
                "start_col": _as_int(part.get("start_col")),
                "num_cols": _as_int(part.get("num_cols")),
                "contexts": contexts,
            })
    return found


def columns_of(part: dict) -> set[int]:
    """Column indexes a partition owns, from its start and count."""
    start, count = part.get("start_col"), part.get("num_cols")
    if start is None or count is None or count < 1:
        return set()
    return set(range(start, start + count))


def _context_key(part: dict, ctx: dict) -> tuple:
    return (part.get("start_col"), part.get("num_cols"), ctx.get("pid"), ctx.get("slot"))


def decode_aie_metadata(raw: bytes) -> dict | None:
    """Core, memory and shim rows from ``amdxdna_drm_query_aie_metadata``.

    Layout is that struct in ``drm/amdxdna_accel.h``: column size, column
    count, total rows, version, then core, mem and shim tile metadata.
    """
    if len(raw) < 64:
        return None
    columns, total_rows = struct.unpack_from("<HH", raw, 4)
    core_rows, core_start = struct.unpack_from("<HH", raw, 16)
    mem_rows, mem_start = struct.unpack_from("<HH", raw, 32)
    shim_rows, shim_start = struct.unpack_from("<HH", raw, 48)
    if columns < 1 or core_rows < 1:
        return None
    return {
        "columns": columns,
        "total_rows": total_rows,
        "core_rows": core_rows,
        "core_row_start": core_start,
        "mem_rows": mem_rows,
        "mem_row_start": mem_start,
        "shim_rows": shim_rows,
        "shim_row_start": shim_start,
    }


def array_geometry(path: str | None = None) -> dict | None:
    """One ``DRM_AMDXDNA_QUERY_AIE_METADATA`` ioctl. None if the device is absent.

    ``DRM_AMDXDNA_GET_INFO`` is ioctl id 7 and ``QUERY_AIE_METADATA`` is param 1
    in the kernel uapi. The request is ``_IOWR('d', 0x40 + 7, get_info)``.
    """
    nodes = [path] if path else sorted(glob.glob("/dev/accel/accel*"))
    if not nodes or nodes[0] is None:
        return None
    try:
        fd = os.open(nodes[0], os.O_RDWR)
    except OSError:
        return None
    try:
        meta = ctypes.create_string_buffer(64)
        info = ctypes.create_string_buffer(struct.pack("<IIQ", 1, 64, ctypes.addressof(meta)))
        request = ((2 | 1) << 30) | (16 << 16) | (ord("d") << 8) | (0x40 + 7)
        try:
            fcntl.ioctl(fd, request, info, True)
        except OSError:
            return None
        return decode_aie_metadata(bytes(meta))
    finally:
        os.close(fd)


def _count(per_column: int | None, columns: set[int]) -> int | None:
    if per_column is None:
        return None
    return per_column * len(columns)


def sample_from(geometry: dict | None, parts: list[dict], previous: dict | None) -> dict:
    """One report: engines reserved by live contexts, and engines whose work advanced."""
    reserved: set[int] = set()
    busy: set[int] = set()
    busy_known = previous is not None
    prev_completions = {}
    if previous is not None:
        for part in previous["partitions"]:
            for ctx in part["contexts"]:
                if ctx["completions"] is not None:
                    prev_completions[_context_key(part, ctx)] = ctx["completions"]
    described = []
    for part in parts:
        if not part["contexts"]:
            continue
        cols = columns_of(part)
        reserved |= cols
        moved = False
        for ctx in part["contexts"]:
            before = prev_completions.get(_context_key(part, ctx))
            if ctx["completions"] is not None and before is not None and ctx["completions"] > before:
                moved = True
        if moved:
            busy |= cols
        described.append({
            "start_col": part["start_col"],
            "num_cols": part["num_cols"],
            "columns": sorted(cols),
            "contexts": part["contexts"],
        })
    core = None if geometry is None else geometry.get("core_rows")
    mem = None if geometry is None else geometry.get("mem_rows")
    return {
        "columns": None if geometry is None else geometry.get("columns"),
        "total_rows": None if geometry is None else geometry.get("total_rows"),
        "core_rows": core,
        "core_row_start": None if geometry is None else geometry.get("core_row_start"),
        "mem_rows": mem,
        "shim_rows": None if geometry is None else geometry.get("shim_rows"),
        "partitions": described,
        "engines_reserved": _count(core, reserved),
        "memory_tiles_reserved": _count(mem, reserved),
        "columns_reserved": sorted(reserved),
        "engines_busy": _count(core, busy) if busy_known else None,
        "memory_tiles_busy": _count(mem, busy) if busy_known else None,
        "columns_busy": sorted(busy) if busy_known else None,
    }


def _span(columns: list[int]) -> str:
    if not columns:
        return "none"
    ranges = []
    start = prev = columns[0]
    for col in columns[1:]:
        if col == prev + 1:
            prev = col
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = col
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _tile_phrase(per_column: int | None, n_columns: int, kind: str) -> str:
    if per_column is None:
        return ""
    return f", {per_column * n_columns} {kind}"


def summarize(samples: list[dict]) -> list[str]:
    """Lines a test log can print. Empty when nothing was sampled."""
    if not samples:
        return ["no xrt-smi samples"]
    geo = next((s for s in samples if s.get("core_rows") and s.get("columns")), None)
    lines = []
    if geo:
        lines.append(
            f"array {geo['total_rows']}x{geo['columns']} tiles: "
            f"{geo['shim_rows']} shim row, {geo['mem_rows']} memory-tile row, "
            f"{geo['core_rows']} AI-engine rows starting at row {geo['core_row_start']}, "
            f"{geo['columns']} columns"
        )
        lines.append(
            f"{geo['core_rows'] * geo['columns']} AI engines and "
            f"{geo['mem_rows'] * geo['columns']} memory tiles on the NPU"
        )
    else:
        lines.append("AI-engine rows unknown (the driver metadata query did not answer)")
    live = [s for s in samples if s["partitions"]]
    if not live:
        lines.append(f"{len(samples)} samples, no hardware context (no AI engines reserved)")
        return lines
    peak_ctx = 0
    peak_parts = 0
    peak_reserved = None
    peak_busy = None
    core_rows = geo["core_rows"] if geo else None
    mem_rows = geo["mem_rows"] if geo else None
    reserved_sets: set[tuple[int, ...]] = set()
    part_shapes: set[tuple[tuple[int, int], ...]] = set()
    for s in live:
        contexts = sum(len(p["contexts"]) for p in s["partitions"])
        peak_ctx = max(peak_ctx, contexts)
        peak_parts = max(peak_parts, len(s["partitions"]))
        if s["engines_reserved"] is not None:
            peak_reserved = s["engines_reserved"] if peak_reserved is None else max(peak_reserved, s["engines_reserved"])
        if s["engines_busy"] is not None:
            peak_busy = s["engines_busy"] if peak_busy is None else max(peak_busy, s["engines_busy"])
        reserved_sets.add(tuple(s["columns_reserved"]))
        part_shapes.add(tuple((p["start_col"], p["num_cols"]) for p in s["partitions"]))
    lines.append(
        f"{len(samples)} samples, {len(live)} with a hardware context, "
        f"peak {peak_ctx} context(s) in {peak_parts} partition(s)"
    )
    if peak_reserved is None:
        lines.append("columns reserved, but the driver did not give AI-engine rows")
    else:
        lines.append(f"peak AI engines reserved: {peak_reserved}")
    if peak_busy is None:
        lines.append("no second sample, so busy engines (completions moving) were not measured")
    else:
        lines.append(f"peak AI engines busy: {peak_busy}")
    for columns in sorted(reserved_sets):
        lines.append(
            "columns reserved: "
            + _span(list(columns))
            + _tile_phrase(core_rows, len(columns), "AI engines")
            + _tile_phrase(mem_rows, len(columns), "memory tiles")
        )
    for shape in sorted(part_shapes, key=lambda item: (len(item), item)):
        bits = []
        for start, count in shape:
            if start is None or count is None:
                bits.append("unknown span")
            else:
                bits.append(
                    f"columns {start}-{start + count - 1}"
                    + _tile_phrase(core_rows, count, "AI engines")
                    + _tile_phrase(mem_rows, count, "memory tiles")
                )
        lines.append("partition " + " | ".join(bits))
    return lines


def _run_json(binary: str, report: str) -> dict | None:
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="aie_")
    os.close(fd)
    path = Path(tmp)
    try:
        proc = subprocess.run(
            [binary, "examine", "-r", report, "-f", "JSON", "-o", str(path), "--force"],
            check=False, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0 or not path.is_file():
            return None
        return json.loads(path.read_text())
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        path.unlink(missing_ok=True)


def watch(stop_fd: int, out_path: Path, summary_path: Path, sample_s: float = SAMPLE_S) -> int:
    """Sample until ``stop_fd`` closes. Write JSON lines and a text summary.

    Returns 0 even when xrt-smi is missing: a missing tool is reported in the
    summary and does not fail the test stage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    binary = find_xrt_smi()
    if binary is None:
        summary_path.write_text("xrt-smi not found, AI engines not measured\n")
        return 0
    geometry = array_geometry()
    poller = select.poll()
    poller.register(stop_fd, select.POLLHUP | select.POLLIN | select.POLLERR | select.POLLNVAL)
    previous = None
    samples = []
    with out_path.open("w") as handle:
        while True:
            parts = partitions_from_json(_run_json(binary, "aie-partitions"))
            current = sample_from(geometry, parts, previous)
            current["t"] = time.time()
            handle.write(json.dumps(current) + "\n")
            handle.flush()
            samples.append(current)
            previous = {"partitions": current["partitions"]}
            if poller.poll(int(sample_s * 1000)):
                break
    summary_path.write_text("\n".join(summarize(samples)) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``--stop-on-stdin --out report.jsonl --summary report.txt``."""
    import argparse
    parser = argparse.ArgumentParser(description="Record NPU AI Engine use until stdin closes")
    parser.add_argument("--stop-on-stdin", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.stop_on_stdin:
        parser.error("--stop-on-stdin is required")
    try:
        return watch(sys.stdin.fileno(), args.out, args.summary)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
