# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Watchable recordings of digital-twin runs.

``SimRecorder`` writes a 3-pane MP4 - what the user camera saw (scripted or
webcam) | the wrist camera | the 3-D view - with a HUD strip (sim time,
active behavior, joint goals vs. actual, gripper current bar, latest
events), plus a ``.jsonl`` timeline of every event. ``write_index`` builds
an ``index.html`` that plays all recordings of a test session side by side
with their pass/fail status.

The panes are extra low-resolution views on the shared renderer (one GPU
pass for everything); frames are encoded by an ``ffmpeg`` subprocess on a
writer thread so recording never stalls rendering.
"""

from __future__ import annotations

import html
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from so101_sim import MOTOR_NAMES

logger = logging.getLogger(__name__)

_HUD_H = 60
_GRIPPER = MOTOR_NAMES.index("gripper")
_CLOSE_LIMIT = 0.5  # the Torque_Limit (fraction) load_threshold is set for; a hold runs lower
_SHORT = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wflex",
    "wrist_roll": "roll",
    "gripper": "grip",
}


class SimRecorder:
    """Record a twin run to ``path`` (.mp4) + ``path.with_suffix('.jsonl')``."""

    _ids = 0

    def __init__(
        self,
        world,
        path,
        *,
        renderer=None,
        pane: tuple[int, int] = (320, 240),
        fps: float = 15.0,
        user_source=None,
        user_label: str = "user camera",
        title: str = "",
        load_threshold: float | None = 120.0,
        sample_s: float = 0.2,
    ):
        self.world = world
        self.path = Path(path)
        self.timeline_path = self.path.with_suffix(".jsonl")
        self.pane = (int(pane[0]), int(pane[1]))
        self.fps = float(fps)
        self.user_source = user_source
        self.user_label = user_label
        self.title = title
        self.load_threshold = load_threshold
        self.sample_s = float(sample_s)
        self._last_sample = -1e9
        self._renderer = renderer
        self._label = ""
        self._events: list[str] = []
        self._q: queue.Queue = queue.Queue(maxsize=90)
        self._proc = None
        self._writer = None
        self._cv_writer = None
        self._thread = None
        self._t0 = None
        self._frames = 0
        self._timeline = None
        self._running = False
        SimRecorder._ids += 1
        self._tag = f"rec{SimRecorder._ids}"
        self._views = []

    # ---------------------------------------------------------------- API
    def start(self) -> "SimRecorder":
        from so101_sim.render import shared_renderer

        self.path.parent.mkdir(parents=True, exist_ok=True)
        rs = self._renderer or shared_renderer(self.world)
        self._renderer = rs
        cams = self.world.cfg["cameras"]
        arm_cam = (cams.get("arm") or {}).get("camera", "wrist_cam")
        view = dict(cams.get("view") or {})
        view_cam = view.pop("camera", "free")
        if view_cam == "free":
            view_cam = {k: view[k] for k in ("lookat", "distance", "azimuth", "elevation") if k in view}
        w, h = self.pane
        self._arm_view = rs.add_view(f"{self._tag}:arm", arm_cam, w, h, self.fps)
        self._3d_view = rs.add_view(f"{self._tag}:view", view_cam, w, h, self.fps)
        self._views = [self._arm_view.name, self._3d_view.name]
        self._timeline = open(self.timeline_path, "w", encoding="utf-8")
        self._open_encoder()
        self._running = True
        self._thread = threading.Thread(target=self._write_loop, name=f"{self._tag}-writer", daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        rs.add_hook(self._on_pass)
        self.event("record_start", title=self.title)
        return self

    def label(self, text: str) -> None:
        """Set the HUD's 'what is happening' line (e.g. the active behavior)."""
        self._label = str(text)

    def event(self, kind: str, **data) -> None:
        """Timestamp an event into the timeline (and the HUD ticker)."""
        rec = {"t_sim": round(self.world.time, 3), "t_wall": round(time.time(), 3), "event": kind, **data}
        if self._timeline is not None:
            try:
                self._timeline.write(json.dumps(rec, default=str) + "\n")
                self._timeline.flush()
            except ValueError:
                pass
        detail = " ".join(f"{k}={v}" for k, v in data.items() if k not in ("title",))
        self._events.append(f"{rec['t_sim']:6.2f}s {kind} {detail}".strip())
        self._events = self._events[-3:]

    def stamp(self, status: str, message: str = "", seconds: float = 1.5) -> None:
        """Append ``seconds`` of the final frame with a PASS/FAIL banner."""
        if not self._running:
            return
        self._renderer.remove_hook(self._on_pass)
        frame = self.compose(self.world.time)
        W = frame.shape[1]
        color = {"passed": (40, 140, 40), "failed": (30, 30, 190)}.get(status, (90, 90, 90))
        cv2.rectangle(frame, (0, 0), (W, 46), color, -1)
        cv2.putText(frame, status.upper(), (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for i, line in enumerate(str(message).splitlines()[:2]):
            cv2.putText(frame, line[:110], (140, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        self.event("result", status=status, message=str(message)[:500])
        for _ in range(max(1, int(seconds * self.fps))):
            self._q.put(frame)

    def stop(self, keep: bool = True) -> Path | None:
        """Finish the file; ``keep=False`` deletes it (passing test, --sim-video failed)."""
        if not self._running:
            return self.path if keep else None
        self.event("record_stop")
        rs = self._renderer
        rs.remove_hook(self._on_pass)
        self._running = False
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=20.0)
            except Exception:
                self._proc.kill()
        if self._cv_writer is not None:
            self._cv_writer.release()
        for name in self._views:
            rs.release_view(name)
        if self._timeline is not None:
            self._timeline.close()
            self._timeline = None
        if not keep:
            for p in (self.path, self.timeline_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return None
        return self.path

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ----------------------------------------------------------- encoding
    @property
    def frame_size(self) -> tuple[int, int]:
        w, h = self.pane
        return 3 * w, h + _HUD_H

    def _open_encoder(self) -> None:
        W, H = self.frame_size
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{self.fps:g}",
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            logger.warning("ffmpeg not found - writing mp4v with OpenCV (may not play in browsers)")
            self._cv_writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, H)
            )

    def _write_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                if self._proc is not None:
                    self._proc.stdin.write(item.tobytes())
                elif self._cv_writer is not None:
                    self._cv_writer.write(item)
            except (BrokenPipeError, ValueError, OSError) as e:
                logger.warning("recorder: encoder stopped (%s)", e)
                return

    # ------------------------------------------------------------ compose
    def _on_pass(self, service, due, t_sim) -> None:
        if not self._running or self._3d_view not in due:
            return
        if t_sim - self._last_sample >= self.sample_s:
            self._last_sample = t_sim
            self._sample()
        frame = self.compose(t_sim)
        # keep wall-clock timing honest: repeat frames if the renderer lagged
        target = int((time.perf_counter() - self._t0) * self.fps) + 1
        n = max(1, min(target - self._frames, 4))
        for _ in range(n):
            try:
                self._q.put_nowait(frame)
                self._frames += 1
            except queue.Full:
                break

    def _sample(self) -> None:
        snap = self.world.snapshot()
        rec = {
            "t_sim": round(snap["t"], 3),
            "event": "state",
            "joints": {k: round(v, 1) for k, v in snap["joints"].items()},
            "goal": {k: round(v, 1) for k, v in snap["goal"].items()},
            "gripper_current": round(
                snap["current_a"]["gripper"] * 1000.0 / self.world.current.ma_per_unit
            ),
        }
        if self._timeline is not None:
            try:
                self._timeline.write(json.dumps(rec) + "\n")
            except ValueError:
                pass

    def _pane(self, img, text: str) -> np.ndarray:
        w, h = self.pane
        if img is None:
            out = np.full((h, w, 3), 40, np.uint8)
            cv2.putText(out, "no signal", (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        else:
            out = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            out = out.copy()
        cv2.rectangle(out, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return out

    def compose(self, t_sim: float) -> np.ndarray:
        user = None
        if self.user_source is not None:
            try:
                user = self.user_source()
            except Exception:
                user = None
        arm = self._arm_view.latest()[1]
        view = self._3d_view.latest()[1]
        top = np.hstack(
            [
                self._pane(user, self.user_label),
                self._pane(arm, "wrist camera (sim)"),
                self._pane(view, "digital twin"),
            ]
        )
        return np.vstack([top, self._hud(t_sim, top.shape[1])])

    def _hud(self, t_sim: float, width: int) -> np.ndarray:
        hud = np.full((_HUD_H, width, 3), 24, np.uint8)
        snap = self.world.snapshot()
        head = f"t={t_sim:6.2f}s  {self.title}  {self._label}".strip()
        cv2.putText(hud, head[:110], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        joints = "  ".join(
            f"{_SHORT[m]} {snap['joints'][m]:6.1f}/{snap['goal'][m]:6.1f}" for m in MOTOR_NAMES
        )
        cv2.putText(hud, joints, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 220, 255), 1)
        if self._events:
            cv2.putText(hud, self._events[-1][:120], (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 255, 170), 1)
        # gripper current bar (raw Present_Current counts, threshold tick)
        counts = snap["current_a"]["gripper"] * 1000.0 / self.world.current.ma_per_unit
        x0, x1 = width - 210, width - 10
        cv2.rectangle(hud, (x0, 40), (x1, 52), (70, 70, 70), 1)
        frac = min(counts / 450.0, 1.0)
        level = self.load_threshold
        if level:
            level *= float(self.world.protection.limit[_GRIPPER]) / _CLOSE_LIMIT
        color = (0, 200, 255) if (level and counts >= level) else (0, 200, 0)
        cv2.rectangle(hud, (x0 + 1, 41), (x0 + 1 + int(frac * (x1 - x0 - 2)), 51), color, -1)
        if level:
            tx = x0 + int(min(level / 450.0, 1.0) * (x1 - x0))
            cv2.line(hud, (tx, 37), (tx, 55), (255, 255, 255), 1)
        cv2.putText(hud, f"gripper current {counts:4.0f}", (x0, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        return hud


def write_index(out_dir, results: dict | None = None, title: str = "SO-101 digital-twin test videos") -> Path:
    """``index.html`` playing every ``*.mp4`` in ``out_dir`` with its status + timeline."""
    out_dir = Path(out_dir)
    results = results or {}
    cards = []
    for mp4 in sorted(out_dir.glob("*.mp4")):
        name = mp4.stem
        status = results.get(name, "")
        color = {"passed": "#2e7d32", "failed": "#c62828"}.get(status, "#555")
        tl = mp4.with_suffix(".jsonl")
        events = []
        if tl.exists():
            for line in tl.read_text(encoding="utf-8").splitlines()[:400]:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t_sim", "t_wall", "event"))
                events.append(f"<li><code>{e.get('t_sim', 0):7.2f}s</code> <b>{html.escape(str(e.get('event')))}</b> {html.escape(extra)}</li>")
        cards.append(
            f"""<section><h2>{html.escape(name)} <span style="background:{color}">{html.escape(status or 'recorded')}</span></h2>
<video src="{html.escape(mp4.name)}" controls preload="metadata" width="960"></video>
<details><summary>timeline ({len(events)} events)</summary><ul>{''.join(events)}</ul></details></section>"""
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:24px}}section{{margin-bottom:32px}}
h2 span{{font-size:13px;padding:2px 8px;border-radius:4px;color:#fff;margin-left:8px}}
video{{background:#000;border-radius:6px}}code{{color:#9cf}}li{{font-size:13px}}</style></head>
<body><h1>{html.escape(title)}</h1><p>{len(cards)} recording(s) - generated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
{''.join(cards)}</body></html>"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
