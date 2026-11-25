# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
resource_hud.py -- always-on-top consolidated resource HUD
==========================================================
A tiny window that floats above every other window (browser, terminals,
slides) and shows three numbers only:

    CPU   %              busy % from `top`             (burgundy)
    GPU   %              busy % from amdgpu sysfs       (orange)
    NPU   N inf/s | Idle inferences/sec from `xrt-smi` (cyan)

Open and close it whenever you like:

    * From a shell:     ./launch_monitor.sh        (or  python3 resource_hud.py)
    * From a notebook:  open_monitor()  /  close_monitor()   (see monitor.py)

Controls once it is open:
    * Drag anywhere to move it.
    * Esc, right-click, or the x closes it.
    * Press 't' to toggle always-on-top.

Every data source is optional -- a missing tool just shows N/A or Idle, so
this runs fine on a plain dev machine with no NPU/GPU tooling installed.
Sources:
    top -bn1                     -> CPU busy %  (100 - idle)
    amdgpu gpu_busy_percent      -> GPU busy % (compute + graphics; radeontop fallback)
    xrt-smi aie-partitions JSON  -> NPU inferences/sec (delta command_completions / dt)
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

# --------------------------------------------------------------------------
# Consolidated palette (as requested):
#   CPU = burgundy, GPU = orange, NPU = cyan
# --------------------------------------------------------------------------
BG = "#16161f"  # window frame / outer accent
CARD = "#1e1e2a"  # inner card
FG = "#f2f2f8"  # primary text
DIM = "#8a8aa3"  # secondary text
TRACK = "#101018"  # empty bar track

CPU_COL = "#8c1d40"  # burgundy
GPU_COL = "#ff8c42"  # orange
NPU_COL = "#22d3ee"  # cyan

CPU_INTERVAL = 1.0  # seconds between top polls
GPU_INTERVAL = 1.0  # seconds between radeontop polls
NPU_INTERVAL = 3.0  # xrt-smi is slower; poll it less often


# ==================================================================== collectors


def _run(cmd, timeout):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


def read_cpu():
    """CPU busy percent = 100 - idle, parsed from `top -bn1`. None if unknown."""
    out = _run(["top", "-bn1", "-w", "200"], 6)
    m = re.search(r"%Cpu\(s\):.*?([\d.]+)\s*id", out)
    if m:
        return round(100.0 - float(m.group(1)), 1)
    return None


def _find_gpu_busy():
    """Locate the amdgpu overall-busy sysfs node (the driver's true engine gauge)."""
    for p in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        if os.access(p, os.R_OK):
            return p
    return None


_GPU_BUSY = _find_gpu_busy()


def read_gpu_busy():
    """Overall iGPU busy % from amdgpu ``gpu_busy_percent`` sysfs, or None.

    This counts compute (ROCm/HIP) as well as graphics, so it reflects the
    load of Gemma 4 on the iGPU that the workshop actually drives. Clamped to 100.
    """
    if not _GPU_BUSY:
        return None
    try:
        with open(_GPU_BUSY) as fh:
            return min(float(fh.read().strip()), 100.0)
    except (OSError, ValueError):
        return None


def _read_radeontop_gpu():
    """GPU busy % from one `radeontop` dump line (graphics pipe only). None if unavailable."""
    out = _run(["radeontop", "-d", "-", "-l", "1"], 10)
    line = next((l for l in out.splitlines() if "gpu" in l and "%" in l), "")
    m = re.search(r"\bgpu\s+([\d.]+)%", line)
    if m:
        return float(m.group(1))
    return None


def read_gpu():
    """GPU busy percent, or None if no source is available.

    Prefers the amdgpu ``gpu_busy_percent`` sysfs node, the driver's real
    engine-busy gauge, which includes ROCm/HIP *compute* load. radeontop's
    ``gpu`` counter only watches the graphics/render pipe, so it reads near-idle
    while the iGPU is busy running Gemma - that mismatch is why the HUD used to
    show a wrong (too-low) GPU %. radeontop is kept as a fallback for machines
    without the sysfs node.
    """
    busy = read_gpu_busy()
    if busy is not None:
        return busy
    return _read_radeontop_gpu()


def _find_xrt_smi():
    for cand in ("xrt-smi", "/opt/xilinx/xrt/bin/xrt-smi", "/usr/bin/xrt-smi"):
        p = shutil.which(cand)
        if p:
            return p
    return None


_XRT_SMI = _find_xrt_smi()


def _parse_hw_contexts(out):
    """Parse the multi-line `xrt-smi examine` HW Contexts table.

    Each context is a group of pipe-delimited lines between |---- separators:
        |690273   |1       |0 |0 |0 |Normal |   row0: pid|ctx|...|priority
        |N/A      |Idle    |0 |0 |  |26     |   row1: pname|status|...|gops
        |N/A      |2528 KB |  |  |  |1      |   row2: mem|instr_bo|...|fps
        |         |        |  |  |  |2000   |   row3: ...|latency
    """
    lines = out.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "HW Contexts" in ln:
            start = i
            break
    if start is None:
        return []

    groups, current = [], []
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if not s.startswith("|"):
            if current:
                groups.append(current)
                current = []
            if s and not s.startswith("|"):
                break  # left the table
            continue
        if re.match(r"\|[=\-]", s):  # |====| or |----| separator
            if current:
                groups.append(current)
                current = []
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        current.append(cells)
    if current:
        groups.append(current)

    ctxs = []
    for g in groups:

        def cell(r, c):
            try:
                return g[r][c]
            except IndexError:
                return ""

        if not re.match(r"\d+$", cell(0, 0)):
            continue  # header group or junk
        ctxs.append(
            {
                "pid": cell(0, 0),
                "status": cell(1, 1),
                "gops": cell(1, 5),
                "fps": cell(2, 5),
            }
        )
    return ctxs


def npu_inferences_per_sec(out):
    """FALLBACK: sum the ``fps`` column across active HW contexts in the xrt-smi
    text table. Pure function (testable).

    Used only when the structured JSON completions counter is unavailable. The
    text table's fps/latency columns are unreliable - their meaning and position
    shift between xrt-smi versions and NPU generations - so prefer the
    delta-completions/dt path in :func:`read_npu`. Returns a float when work is
    seen, else None.
    """
    if not out or "No hardware contexts" in out:
        return None
    total, got = 0.0, False
    for c in _parse_hw_contexts(out):
        if c["status"].lower() in ("idle", ""):
            continue
        try:
            total += float(c["fps"])
            got = True
        except (ValueError, TypeError):
            pass
    return total if (got and total > 0) else None


def _read_npu_json():
    """Structured ``aie-partitions`` JSON report from xrt-smi, parsed to a dict.

    The JSON report is stable across xrt-smi versions (unlike the free-text
    table) and carries the ``command_completions`` counter we sample. Returns
    None if xrt-smi is missing or the call fails.
    """
    if not _XRT_SMI:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="npu_")
    os.close(fd)
    try:
        _run(
            [
                _XRT_SMI,
                "examine",
                "-r",
                "aie-partitions",
                "-f",
                "JSON",
                "-o",
                tmp,
                "--force",
            ],
            20,
        )
        with open(tmp) as f:
            return json.load(f)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def npu_completions_total(data):
    """Sum ``command_completions`` across every HW context in the JSON report.

    Pure function (testable). Returns an int total, or None when the counter is
    absent (e.g. an older xrt-smi that does not emit it).
    """
    if not data:
        return None
    total, found = 0, False
    for dev in data.get("devices", []):
        ap = dev.get("aie_partitions") or {}
        for part in ap.get("partitions", []):
            for ctx in part.get("hw_contexts", []):
                comp = ctx.get("command_completions")
                if isinstance(comp, bool):
                    continue
                if isinstance(comp, (int, float)):
                    total += int(comp)
                    found = True
                elif isinstance(comp, str) and comp.strip().isdigit():
                    total += int(comp)
                    found = True
    return total if found else None


class _NpuRate:
    """Turn xrt-smi's cumulative ``command_completions`` into inferences/sec.

    The NPU exposes no live throughput gauge, but that counter rises by one per
    completed inference, so sampling it over wall-clock time gives the real
    rate: inferences/s = delta(completions) / delta(t).
    """

    def __init__(self):
        self._last_total = None
        self._last_t = None

    def update(self, total):
        """Feed the latest completions total; return inferences/sec, or None."""
        now = time.monotonic()
        if total is None:
            self._last_total = self._last_t = None
            return None
        ips = None
        if self._last_total is not None and self._last_t is not None:
            dt = now - self._last_t
            dcount = total - self._last_total
            if dt > 0 and dcount >= 0:
                ips = dcount / dt
        self._last_total = total
        self._last_t = now
        return ips


_NPU_RATE = _NpuRate()


def read_npu():
    """NPU inferences/sec, or None (Idle / tool not present).

    Primary path: sample xrt-smi's cumulative ``command_completions`` counter
    (aie-partitions JSON) and report delta-count / delta-time. This is the
    reliable throughput signal; the old code summed the text table's ``fps``
    column, which reads unreliably on current Ryzen AI / XDNA2 hardware - the
    reason the NPU number was wrong. Falls back to that text parse only when the
    JSON counter is unavailable (older xrt-smi). The first sample returns None
    (no delta yet), so the readout shows Idle until the next poll.
    """
    if not _XRT_SMI:
        return None
    total = npu_completions_total(_read_npu_json())
    if total is not None:
        ips = _NPU_RATE.update(total)
        return ips if (ips and ips > 0) else None
    out = _run([_XRT_SMI, "examine", "--report", "all"], 20) or _run(
        [_XRT_SMI, "examine"], 15
    )
    return npu_inferences_per_sec(out)


# ========================================================================= GUI
import tkinter as tk  # noqa: E402  (kept here so collectors import without a display)


class Stat(tk.Frame):
    """One row: coloured tag | optional bar | value."""

    def __init__(self, parent, label, color, show_bar=True):
        super().__init__(parent, bg=CARD)
        self.color = color
        tk.Label(
            self,
            text=label,
            bg=CARD,
            fg=color,
            width=4,
            anchor="w",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side="left")
        if show_bar:
            self.canvas = tk.Canvas(
                self, width=88, height=10, bg=TRACK, highlightthickness=0
            )
            self.canvas.pack(side="left", padx=(2, 8))
        else:
            self.canvas = None
        self.val = tk.Label(
            self,
            text="--",
            bg=CARD,
            fg=FG,
            width=9,
            anchor="e",
            font=("Monospace", 12, "bold"),
        )
        self.val.pack(side="right")

    def set_pct(self, pct):
        if self.canvas is not None:
            self.canvas.delete("all")
            if pct is not None:
                w = min(max(pct, 0.0), 100.0) / 100.0 * 88
                self.canvas.create_rectangle(0, 0, w, 10, fill=self.color, outline="")
        self.val.config(text="N/A" if pct is None else f"{pct:.0f}%", fg=FG)

    def set_text(self, text, dim=False):
        self.val.config(text=text, fg=DIM if dim else self.color)


class HUD(tk.Tk):
    """The frameless always-on-top window: CPU/GPU bars + NPU inf/s readout."""

    def __init__(self, topmost=True):
        super().__init__()
        self.title("Resource HUD")
        self.overrideredirect(True)  # frameless, sleek
        self._topmost = topmost
        try:
            self.attributes("-topmost", topmost)
        except tk.TclError:
            pass
        self.configure(bg=CPU_COL)  # 2px accent border around the card

        card = tk.Frame(self, bg=CARD, padx=12, pady=9)
        card.pack(padx=2, pady=2)

        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x")
        tk.Label(
            head, text="RESOURCES", bg=CARD, fg=DIM, font=("TkDefaultFont", 8, "bold")
        ).pack(side="left")
        close = tk.Label(
            head,
            text="✕",
            bg=CARD,
            fg=DIM,
            cursor="hand2",
            font=("TkDefaultFont", 10, "bold"),
        )
        close.pack(side="right")
        close.bind("<Button-1>", self._close)

        self.cpu = Stat(card, "CPU", CPU_COL)
        self.cpu.pack(fill="x", pady=(7, 1))
        self.gpu = Stat(card, "GPU", GPU_COL)
        self.gpu.pack(fill="x", pady=1)
        self.npu = Stat(card, "NPU", NPU_COL, show_bar=False)
        self.npu.pack(fill="x", pady=(1, 0))

        # drag-to-move from anywhere on the card
        for w in (self, card, head, self.cpu, self.gpu, self.npu):
            w.bind("<ButtonPress-1>", self._start_move)
            w.bind("<B1-Motion>", self._on_move)
        self.bind("<Escape>", self._close)
        self.bind("<Button-3>", self._close)  # right-click closes
        self.bind("<Key-t>", lambda e: self._toggle_top())

        self.update_idletasks()
        self._place_top_right()

        self._cpu = self._gpu = self._npu = None
        self._stop = threading.Event()
        threading.Thread(target=self._poll_fast, daemon=True).start()
        threading.Thread(target=self._poll_npu, daemon=True).start()
        self.after(250, self._refresh)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # -------------------------------------------------------------- placement
    def _place_top_right(self):
        w, h = self.winfo_width(), self.winfo_height()
        sw = self.winfo_screenwidth()
        self.geometry(f"+{max(sw - w - 24, 0)}+40")

    def _start_move(self, e):
        self._ox = e.x_root - self.winfo_x()
        self._oy = e.y_root - self.winfo_y()

    def _on_move(self, e):
        self.geometry(f"+{e.x_root - self._ox}+{e.y_root - self._oy}")

    # -------------------------------------------------------------- polling
    def _poll_fast(self):
        while not self._stop.is_set():
            self._cpu = read_cpu()
            self._gpu = read_gpu()
            time.sleep(min(CPU_INTERVAL, GPU_INTERVAL))

    def _poll_npu(self):
        while not self._stop.is_set():
            self._npu = read_npu()
            time.sleep(NPU_INTERVAL)

    # -------------------------------------------------------------- UI tick
    def _refresh(self):
        if self._topmost:  # re-assert so WMs don't drop us
            try:
                self.lift()
                self.attributes("-topmost", True)
            except tk.TclError:
                pass
        self.cpu.set_pct(self._cpu)
        self.gpu.set_pct(self._gpu)
        if self._npu is None:
            self.npu.set_text("Idle", dim=True)
        else:
            self.npu.set_text(f"{self._npu:.0f} inf/s")
        self.after(500, self._refresh)

    # -------------------------------------------------------------- controls
    def _toggle_top(self):
        self._topmost = not self._topmost
        try:
            self.attributes("-topmost", self._topmost)
        except tk.TclError:
            pass

    def _close(self, *_):
        self._stop.set()
        self.destroy()
        return "break"


def main():
    """Parse ``--no-top`` and run the HUD event loop."""
    ap = argparse.ArgumentParser(
        description="Always-on-top consolidated CPU/GPU/NPU HUD."
    )
    ap.add_argument(
        "--no-top", action="store_true", help="do not force the window to stay on top"
    )
    args = ap.parse_args()
    HUD(topmost=not args.no_top).mainloop()


if __name__ == "__main__":
    main()
