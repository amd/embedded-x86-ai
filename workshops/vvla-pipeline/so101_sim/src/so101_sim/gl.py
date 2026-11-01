# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""OpenGL backend selection for offscreen rendering.

MuJoCo reads ``MUJOCO_GL`` once, when ``mujoco`` is first imported, so
:func:`configure` must run before that import - every so101_sim module that
imports mujoco calls it first.

On the Strix Halo box the right choice is EGL on the Radeon 8060S
(radeonsi): render + readback of a 640x480 view takes ~2 ms. Without a GPU
Mesa silently falls back to llvmpipe (software), which works but is slow;
:func:`check_renderer` reports which one you got so tests/benchmarks can
insist on hardware.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_configured: str | None = None


def configure(mode: str | None = "auto", device_id: int | None = 0) -> str:
    """Pick the GL backend (``egl`` | ``osmesa`` | ``glfw``) before mujoco loads.

    ``auto`` keeps an explicit ``MUJOCO_GL`` from the environment, else uses
    EGL (headless, GPU-accelerated through Mesa). ``device_id`` pins the EGL
    device so a multi-GPU box renders on the iGPU. Returns the backend in use.
    """
    global _configured
    import sys

    if _configured is not None:
        if mode not in (None, "auto", _configured):
            logger.warning(
                "GL backend already %s; ignoring request for %s", _configured, mode
            )
        return _configured
    if mode in (None, "auto"):
        backend = os.environ.get("MUJOCO_GL", "egl")
    else:
        backend = mode
    if "mujoco" in sys.modules and os.environ.get("MUJOCO_GL") != backend:
        logger.warning(
            "mujoco was imported before so101_sim.gl.configure(); "
            "MUJOCO_GL=%s may not take effect",
            backend,
        )
    os.environ["MUJOCO_GL"] = backend
    if backend == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if device_id is not None:
            os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(int(device_id)))
    elif backend == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    _configured = backend
    return backend


def renderer_string() -> str:
    """``GL_RENDERER`` of the current context (call with a context current)."""
    try:
        from OpenGL import GL

        raw = GL.glGetString(GL.GL_RENDERER)
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception as e:  # pragma: no cover - depends on the GL stack
        return f"unknown ({e})"


def is_software(renderer: str) -> bool:
    """True for the CPU rasterizers Mesa falls back to."""
    r = renderer.lower()
    return any(s in r for s in ("llvmpipe", "softpipe", "swrast", "lavapipe"))


def check_renderer(require_hw: bool = False) -> str:
    """Log (and optionally enforce) hardware rendering for the current context."""
    r = renderer_string()
    if is_software(r):
        msg = (
            f"MuJoCo is rendering in SOFTWARE ({r}). Expect ~10x slower frames. "
            "Check MUJOCO_EGL_DEVICE_ID and that /dev/dri/renderD* is accessible."
        )
        if require_hw:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info("MuJoCo GL renderer: %s", r)
    return r
