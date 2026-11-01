# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Call :mod:`vla_pipeline.behaviors.manipulation` from a pipeline ``(cfg, arm, ...)`` signature.

Pipeline configs nest behavior settings under ``behaviors``. The handlers in
:mod:`~vla_pipeline.behaviors.manipulation` read them at the top level of
``ctx.cfg``, as the workshop's copy does.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_BEHAVIOR_KEYS = ("pick_place", "grip", "fetch_block", "dance", "wave")


def flatten_cfg(cfg: dict) -> dict:
    """Copy ``behaviors.<name>`` onto the top level when that key is absent."""
    out = dict(cfg)
    behaviors = cfg.get("behaviors") or {}
    for key in _BEHAVIOR_KEYS:
        if key not in out and key in behaviors:
            out[key] = behaviors[key]
    if "mimic" not in out and "gesture_mimic" in behaviors:
        out["mimic"] = behaviors["gesture_mimic"]
    return out


class Ctx:
    """The ``ctx`` the :mod:`~vla_pipeline.behaviors.manipulation` handlers take."""

    def __init__(
        self,
        cfg,
        arm=None,
        *,
        say=None,
        stop_check=None,
        stop_event=None,
        shared_state=None,
        detector=None,
        arm_camera=None,
        camera=None,
        pose=None,
    ):
        self.cfg = flatten_cfg(cfg) if isinstance(cfg, dict) else cfg
        self.arm = arm
        self.detector = detector
        self.arm_camera = arm_camera
        self.camera = camera
        self.pose = pose
        self.shared_state = {} if shared_state is None else shared_state
        self._say = say
        # A dwell waits on the event, so a stop ends it at once. With only
        # stop_check (main's API), dwells run their full length.
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._stop = stop_check or (lambda: False)

    def say(self, message: str) -> None:
        if self._say is not None:
            self._say(message)
        else:
            logger.info("%s", message)

    def stop_check(self) -> bool:
        return self.stop_event.is_set() or bool(self._stop())


def command(object_name: str = ""):
    return SimpleNamespace(object_name=object_name or "")
