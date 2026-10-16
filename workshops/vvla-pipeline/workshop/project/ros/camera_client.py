# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Camera clients - the frame source API every behavior consumes.

Behaviors never own a ``cv2.VideoCapture``; they call
``camera.read() -> (ok, frame)`` and don't care whether frames come from a
ROS 2 topic (:class:`Ros2Camera`) or straight from a device
(:class:`DirectCamera`, provided). Your TODO is the subscriber wiring:
matching QoS (best-effort/keep-last-1 - a reliable subscriber will simply
never match the camera node's best-effort publisher), a background executor
so frames arrive while the caller does other work, and a latest-frame cache.

Notebook: ``notebooks/02_ros2/01_cameras.ipynb``
Run it for real (with a camera node publishing in another terminal):
    python -m ros.camera_client --role operator
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from common.ros_helpers import from_image_msg

logger = logging.getLogger(__name__)


class DirectCamera:
    """PROVIDED: plain OpenCV capture with the same read() API (no ROS)."""

    def __init__(self, role, device, width, height, fps):
        import cv2

        self.role = role
        self._fps = float(fps)
        self._cap = cv2.VideoCapture(device)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        """Read one frame; ``(False, None)`` when the device never opened."""
        if not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        return (ok, frame if ok else None)

    def read_after(self, settled_at):
        """Discard buffered frames until a read blocks for a new exposure."""
        period = 1.0 / max(self._fps, 1.0)
        ok, frame = False, None
        for _ in range(8):
            started = time.perf_counter()
            ok, frame = self.read()
            if not ok or frame is None:
                return False, None
            if time.perf_counter() - started >= 0.5 * period:
                return True, frame
        return bool(ok), frame

    def close(self):
        """Release the OpenCV capture."""
        self._cap.release()


class Ros2Camera:
    """Latest-frame subscriber on a camera node's image topic.

    Keeps only the newest frame and reports ``(False, None)`` when the
    stream goes stale (``stale_after_s``) - so a dead publisher looks like a
    dead camera, not like a frozen image.
    """

    def __init__(self, role: str, topic: str, stale_after_s: float = 1.0):
        self.role = role
        self._stale_after = float(stale_after_s)
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._executor = None
        self._node = None
        self._setup_ros(topic)
        logger.info("[camera:%s] subscribed to %s", role, topic)

    def _setup_ros(self, topic: str) -> None:
        """Create the client node, the subscription, and a spin thread.

        Init rclpy if needed; make a node named ``camera_client_<role>``;
        subscribe to ``topic`` (``sensor_msgs/Image``) with best-effort /
        keep-last / depth-1 QoS routing into ``self._on_image``; then spin a
        ``SingleThreadedExecutor`` on a daemon thread so messages keep
        arriving while the caller's thread does vision work.
        """
        # >>> TODO 2.3: Ros2Camera._setup_ros - notebooks/02_ros2/01_cameras.ipynb
        raise NotImplementedError(
            "TODO 2.3 (Ros2Camera._setup_ros): implement me - see notebooks/02_ros2/01_cameras.ipynb"
        )
        # <<< TODO 2.3

    def _on_image(self, msg) -> None:
        """PROVIDED: decode and cache the newest frame."""
        frame = from_image_msg(msg)
        if frame is None:
            return
        with self._lock:
            self._frame = frame
            self._stamp = time.time()

    def read(self):
        """Return the newest cached frame, or ``(False, None)`` if stale/missing."""
        with self._lock:
            if self._frame is None or (time.time() - self._stamp) > self._stale_after:
                return False, None
            return True, self._frame

    def read_after(self, settled_at):
        """Newest cached frame whose stamp is later than ``settled_at``."""
        deadline = time.time() + max(self._stale_after, 0.5)
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp > settled_at:
                    return True, self._frame
            time.sleep(0.02)
        return False, None

    def close(self) -> None:
        """Stop the executor and destroy the client node."""
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()


def make_camera(cfg: dict, role: str):
    """PROVIDED: build the configured camera client for a role."""
    c = (cfg.get("cameras") or {}).get(role) or {}
    if c.get("use_ros2", False):
        return Ros2Camera(role, c["topic"], stale_after_s=c.get("stale_after_s", 1.0))
    return DirectCamera(
        role,
        c.get("device", 0),
        c.get("width", 640),
        c.get("height", 480),
        c.get("fps", 30),
    )


def main() -> None:
    """CLI: live-preview a camera role until 'q' is pressed."""
    import cv2

    from common.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Camera client - live preview")
    parser.add_argument("--role", choices=["operator", "arm"], default="operator")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cam = make_camera(load_config(args.config), args.role)
    print(f"Previewing camera '{args.role}' - press 'q' to quit.")
    misses = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                misses += 1
                if misses % 30 == 1:
                    print("  (no frames - is the camera node running?)")
                time.sleep(0.1)
                continue
            misses = 0
            cv2.imshow(f"camera:{args.role}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
