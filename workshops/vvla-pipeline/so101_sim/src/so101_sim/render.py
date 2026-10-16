# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Offscreen rendering of every twin camera on ONE GL context and thread.

Rules that keep the iGPU cheap to share with llama-server:

* one EGL context, one render thread for all views (the wrist camera, the
  operator camera, the free 3-D view, segmentation passes);
* the world lock is held only for ``mjv_updateScene`` (a CPU copy of the
  scene), never while the GPU draws;
* all due views are drawn as tiles into one offscreen framebuffer and read
  back with a single ``mjr_readPixels`` - one GPU sync per pass;
* consumers get the newest frame (``View.latest`` / ``View.wait``); nothing
  queues, so a slow consumer never makes the renderer fall behind;
* each frame carries the wall-clock time its scene was captured
  (``View.latest_stamped`` / ``View.wait_stamped``), so a consumer can tell
  whether the pixels show the arm before or after a given moment.

Segmentation tiles use ``mjRND_SEGMENT`` + ``mjRND_IDCOLOR`` and are decoded
to per-pixel body ids - ground truth for the scripted detector in tests.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from so101_sim import gl

logger = logging.getLogger(__name__)

_MAX_BUFFER = 4096


@dataclass
class _Tile:
    view: "View"
    kind: str  # "rgb" | "seg"
    x: int = 0
    y: int = 0


class View:
    """One camera stream: newest BGR frame (+ optional body-id segmentation)."""

    def __init__(self, name, camera, width, height, fps, segmentation, model, mujoco):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.segmentation = bool(segmentation)
        self._mj = mujoco
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=4000)
        self._set_camera(model, camera)
        self.seq = 0
        self._frame: np.ndarray | None = None
        self._stamp = 0.0  # time.time() when the newest frame's scene was captured
        self._segs: collections.deque = collections.deque(maxlen=8)
        self._cond = threading.Condition()
        self.next_due = 0.0
        self.refs = 0

    def _set_camera(self, model, camera) -> None:
        mj = self._mj
        if isinstance(camera, dict) or camera in (None, "free"):
            p = camera if isinstance(camera, dict) else {}
            self.camera.type = mj.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = p.get("lookat", (0.17, 0.0, 0.18))
            self.camera.distance = float(p.get("distance", 0.9))
            self.camera.azimuth = float(p.get("azimuth", 150.0))
            self.camera.elevation = float(p.get("elevation", -24.0))
            self.camera_name = "free"
        else:
            cid = camera if isinstance(camera, int) else model.camera(camera).id
            self.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = int(cid)
            self.camera_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, int(cid))

    # -- consumer side ------------------------------------------------------
    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._cond:
            return self.seq, self._frame

    def wait(self, after_seq: int, timeout: float) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than ``after_seq`` (or timeout); newest wins."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq > after_seq, timeout=timeout)
            return self.seq, self._frame

    def latest_stamped(self) -> tuple[int, np.ndarray | None, float]:
        """``latest()`` plus the wall-clock time the frame's scene was captured."""
        with self._cond:
            return self.seq, self._frame, self._stamp

    def wait_stamped(
        self, after_seq: int, timeout: float
    ) -> tuple[int, np.ndarray | None, float]:
        """``wait()`` plus the wall-clock time the frame's scene was captured."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq > after_seq, timeout=timeout)
            return self.seq, self._frame, self._stamp

    def segmentation_for(self, seq: int | None = None) -> np.ndarray | None:
        """Body-id map (int32, -1 = background) for frame ``seq`` (newest if None)."""
        with self._cond:
            if not self._segs:
                return None
            if seq is None:
                return self._segs[-1][1]
            for s, seg in reversed(self._segs):
                if s <= seq:
                    return seg
            return self._segs[0][1]

    # -- producer side ------------------------------------------------------
    def _publish(self, frame, seg, stamp: float = 0.0) -> None:
        with self._cond:
            self.seq += 1
            self._frame = frame
            self._stamp = float(stamp)
            if seg is not None:
                self._segs.append((self.seq, seg))
            self._cond.notify_all()


class RenderService:
    """Owns the GL context and renders all registered views on a thread."""

    def __init__(self, world, require_hw: bool | None = None, start: bool = True):
        self.world = world
        self._mj = None
        self._views: dict[str, View] = {}
        self._hooks: list = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.renderer = ""
        self.require_hw = (
            bool(world.cfg.get("require_hw_gl", False)) if require_hw is None else require_hw
        )
        self.pass_ms = 0.0  # last render+readback pass
        self.passes = 0
        gl.configure(world.cfg.get("gl", "auto"), world.cfg.get("egl_device", 0))
        import mujoco

        self._mj = mujoco
        if start:
            self.start()

    # -- views -------------------------------------------------------------
    def add_view(
        self,
        name: str,
        camera="free",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        segmentation: bool = False,
    ) -> View:
        with self._lock:
            v = self._views.get(name)
            if v is None:
                v = View(
                    name, camera, width, height, fps, segmentation, self.world.model, self._mj
                )
                self._views[name] = v
            elif segmentation and not v.segmentation:
                v.segmentation = True
            v.refs += 1
            return v

    def release_view(self, name: str) -> None:
        with self._lock:
            v = self._views.get(name)
            if v is None:
                return
            v.refs -= 1
            if v.refs <= 0:
                del self._views[name]

    def add_hook(self, fn) -> None:
        """``fn(service, rendered_views, t_sim)`` after every pass (render thread)."""
        self._hooks.append(fn)

    def remove_hook(self, fn) -> None:
        try:
            self._hooks.remove(fn)
        except ValueError:
            pass

    # -- thread ------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="so101-render", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=20.0):
            raise RuntimeError("render thread did not come up")
        if self._error is not None:
            raise RuntimeError(f"GL context failed: {self._error}") from self._error

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _layout(self, tiles: list[_Tile]) -> tuple[int, int]:
        """Shelf-pack tiles left-to-right; returns the used (width, height)."""
        x = y = shelf_h = used_w = 0
        for t in tiles:
            w, h = t.view.width, t.view.height
            if x + w > _MAX_BUFFER:
                x = 0
                y += shelf_h
                shelf_h = 0
            t.x, t.y = x, y
            x += w
            shelf_h = max(shelf_h, h)
            used_w = max(used_w, x)
        return used_w, y + shelf_h

    def _run(self) -> None:
        mj = self._mj
        model = self.world.model
        ctx = con = None
        buf_w = buf_h = 0
        try:
            ctx = mj.GLContext(_MAX_BUFFER, _MAX_BUFFER)
            ctx.make_current()
            self.renderer = gl.check_renderer(self.require_hw)
        except BaseException as e:  # noqa: BLE001 - reported to start()
            self._error = e
            self._running = False
            self._ready.set()
            return
        self._ready.set()
        try:
            while self._running:
                now = time.perf_counter()
                with self._lock:
                    views = list(self._views.values())
                due = [v for v in views if now >= v.next_due]
                if not due:
                    nxt = min((v.next_due for v in views), default=now + 0.01)
                    time.sleep(min(max(nxt - now, 0.0005), 0.01))
                    continue
                for v in due:
                    v.next_due = max(v.next_due + 1.0 / v.fps, now)
                tiles = [_Tile(v, "rgb") for v in due]
                tiles += [_Tile(v, "seg") for v in due if v.segmentation]
                used_w, used_h = self._layout(tiles)
                if con is None or used_w > buf_w or used_h > buf_h:
                    buf_w = max(used_w, buf_w, 1280)
                    buf_h = max(used_h, buf_h, 960)
                    model.vis.global_.offwidth = buf_w
                    model.vis.global_.offheight = buf_h
                    if con is not None:
                        con.free()
                    con = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
                    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, con)
                    logger.debug("offscreen buffer %dx%d", buf_w, buf_h)
                t0 = time.perf_counter()
                with self.world.lock:
                    for v in due:
                        mj.mjv_updateScene(
                            model, self.world.data, v.option, None, v.camera,
                            mj.mjtCatBit.mjCAT_ALL, v.scene,
                        )
                    t_sim = float(self.world.data.time)
                    t_wall = time.time()
                seg_flag = mj.mjtRndFlag.mjRND_SEGMENT
                id_flag = mj.mjtRndFlag.mjRND_IDCOLOR
                for t in tiles:
                    v = t.view
                    rect = mj.MjrRect(t.x, t.y, v.width, v.height)
                    if t.kind == "seg":
                        v.scene.flags[seg_flag] = 1
                        v.scene.flags[id_flag] = 1
                    mj.mjr_render(rect, v.scene, con)
                    if t.kind == "seg":
                        v.scene.flags[seg_flag] = 0
                        v.scene.flags[id_flag] = 0
                rgb = np.empty((used_h, used_w, 3), dtype=np.uint8)
                mj.mjr_readPixels(rgb, None, mj.MjrRect(0, 0, used_w, used_h), con)
                frames = {}
                segs = {}
                for t in tiles:
                    v = t.view
                    # GL origin is bottom-left: tile rows count from the bottom
                    tile = rgb[t.y : t.y + v.height, t.x : t.x + v.width][::-1]
                    if t.kind == "rgb":
                        frames[v.name] = np.ascontiguousarray(tile[:, :, ::-1])
                    else:
                        segs[v.name] = self._decode_seg(tile, v.scene, model)
                self.pass_ms = 1000.0 * (time.perf_counter() - t0)
                self.passes += 1
                for v in due:
                    v._publish(frames[v.name], segs.get(v.name), t_wall)
                for fn in list(self._hooks):
                    try:
                        fn(self, due, t_sim)
                    except Exception as e:
                        logger.warning("render hook failed: %s", e)
        finally:
            try:
                if con is not None:
                    con.free()
                if ctx is not None:
                    ctx.free()
            except Exception as e:  # pragma: no cover
                logger.debug("GL teardown: %s", e)

    def _decode_seg(self, tile: np.ndarray, scene, model) -> np.ndarray:
        ids = (
            tile[:, :, 0].astype(np.int32)
            + (tile[:, :, 1].astype(np.int32) << 8)
            + (tile[:, :, 2].astype(np.int32) << 16)
        )
        lut = np.full(scene.ngeom + 2, -1, dtype=np.int32)
        geom_t = self._mj.mjtObj.mjOBJ_GEOM
        for i in range(scene.ngeom):
            g = scene.geoms[i]
            if g.segid >= 0 and g.objtype == geom_t:
                lut[g.segid + 1] = model.geom_bodyid[g.objid]
        ids[(ids < 0) | (ids >= lut.size)] = 0
        return lut[ids]


_shared: RenderService | None = None
_shared_lock = threading.Lock()


def shared_renderer(world=None) -> RenderService:
    """The process-wide renderer bound to the shared world."""
    global _shared
    with _shared_lock:
        if _shared is None:
            if world is None:
                from so101_sim.world import shared_world

                world = shared_world()
            _shared = RenderService(world)
        return _shared


def close_shared_renderer() -> None:
    global _shared
    with _shared_lock:
        if _shared is not None:
            _shared.close()
            _shared = None
