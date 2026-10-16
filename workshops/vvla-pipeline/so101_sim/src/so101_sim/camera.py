# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``cv2.VideoCapture``-shaped access to a twin camera.

``SimCamera(world, "arm")`` behaves like the USB wrist camera: ``read()``
returns ``(True, bgr_frame)`` at the configured size, ``isOpened()`` /
``release()`` exist, and in blocking mode ``read()`` waits for the next
frame the way a V4L2 capture does - so loops written for the real camera
(no sleep, one inference per frame) are paced at 30 fps instead of spinning.
Non-blocking mode returns the newest frame immediately (ROS timer callers).
"""

from __future__ import annotations

import collections

from so101_sim.render import shared_renderer


class SimCamera:
    """A twin camera role (``arm`` | ``operator`` | ``view``) or MJCF camera name."""

    def __init__(
        self,
        world=None,
        role: str = "arm",
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        blocking: bool = True,
        segmentation: bool = False,
        renderer=None,
    ):
        if world is None:
            from so101_sim.world import shared_world

            world = shared_world()
        self.world = world
        self.role = role
        spec = dict(world.cfg["cameras"].get(role) or {"camera": role})
        camera = spec.pop("camera", role)
        if camera == "free":
            camera = {k: spec[k] for k in ("lookat", "distance", "azimuth", "elevation") if k in spec}
        self.width = int(width or spec.get("width", 640))
        self.height = int(height or spec.get("height", 480))
        self.fps = float(fps or spec.get("fps", 30))
        self.blocking = blocking
        self._service = renderer or shared_renderer(world)
        self._name = f"{role}:{self.width}x{self.height}"
        self.view = self._service.add_view(
            self._name, camera, self.width, self.height, self.fps, segmentation
        )
        self._last_seq = 0
        # id(frame) -> (render sequence, wall-clock scene capture time)
        self._recent: collections.OrderedDict[int, tuple[int, float]] = collections.OrderedDict()
        self.last_stamp: float | None = None  # capture time of the last frame read()
        self._open = True

    def read_after(self, settled_at: float):
        """Frame rendered after this call.

        ``settled_at`` is when the arm finished moving. The render sequence,
        not that clock, says whether the pixels show the settled pose.
        """
        return self.read(fresh=True)

    def read(self, fresh: bool = False):
        """Next (blocking) or newest frame as ``(ok, bgr)``; never raises.

        ``fresh=True`` waits for a frame whose scene was captured after this
        call - use it after teleporting state (tests, offline renders).
        """
        if not self._open:
            return False, None
        if fresh:
            start, _ = self.view.latest()
            seq, frame, stamp = self.view.wait_stamped(start + 1, timeout=max(3.0 / self.fps, 0.5))
            if seq <= start + 1:  # timed out: the newest frame may predate this call
                frame = None
        elif self.blocking:
            seq, frame, stamp = self.view.wait_stamped(
                self._last_seq, timeout=max(3.0 / self.fps, 0.25)
            )
        else:
            seq, frame, stamp = self.view.latest_stamped()
        if frame is None:
            return False, None
        self._last_seq = seq
        out = frame.copy()
        self._recent[id(out)] = (seq, stamp)
        while len(self._recent) > 16:
            self._recent.popitem(last=False)
        self.last_stamp = stamp
        return True, out

    def seq_of(self, frame) -> int | None:
        """Render sequence number of a frame this camera returned (if recent)."""
        entry = self._recent.get(id(frame))
        return None if entry is None else entry[0]

    def stamp_of(self, frame) -> float | None:
        """``time.time()`` when the scene in ``frame`` was captured (if recent)."""
        entry = self._recent.get(id(frame))
        return None if entry is None else entry[1]

    def segmentation(self, frame=None):
        """Body-id map matching ``frame`` (or the newest frame)."""
        seq = self.seq_of(frame) if frame is not None else None
        return self.view.segmentation_for(seq)

    def ok(self) -> bool:
        return self._open and self.view.latest()[1] is not None

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return self._open

    def set(self, *_args) -> bool:  # cv2 API: properties are fixed by config
        return False

    def release(self) -> None:
        if self._open:
            self._open = False
            self._service.release_view(self._name)

    close = release
