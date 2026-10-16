# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Camera access layer: one ``CameraClient`` API for the two pipeline cameras.

Roles
-----
- ``operator`` - faces the person. Feeds gesture-mimic pose
  tracking and person finding.
- ``arm``   - mounted on the SO-101 end effector. Feeds object detection in
  the jaws/workspace and pick-and-place visual servoing.

Like :class:`vla_pipeline.robot.arm_interface.ArmClient`, behaviors never own
a ``cv2.VideoCapture`` - they take a :class:`CameraClient` so the same code
runs against direct V4L2 devices (``cameras.<role>.use_ros2: false``) or the
ROS 2 camera nodes (:mod:`vla_pipeline.vision.camera_node`).

Graceful failure: ``read()`` returns ``(False, None)`` instead of raising, so
behaviors can degrade (fetch/pick already handle a missing camera) while the
client retries the device in the background.

Independent test
----------------
    python -m vla_pipeline.vision.camera --role operator          # direct preview
    python -m vla_pipeline.vision.camera --role arm --ros2     # via topic
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class CameraClient(ABC):
    """Minimal frame source API used by every behavior."""

    role: str = "?"

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Latest BGR frame. ``(False, None)`` when unavailable (never raises)."""

    def ok(self) -> bool:
        """Return True if a frame is currently available."""
        return self.read()[0]

    def close(self) -> None:  # noqa: B027
        """No-op; override in clients that own a resource to release."""


class DirectCamera(CameraClient):
    """OpenCV capture with automatic reopen on failure (no ROS required)."""

    def __init__(
        self,
        role: str,
        device: int | str,
        width: int,
        height: int,
        fps: float,
        reopen_interval_s: float = 3.0,
    ):
        import cv2  # local: keep module importable without cv2 for unit tests

        self._cv2 = cv2
        self.role = role
        self._device = device
        self._size = (int(width), int(height))
        self._fps = float(fps)
        self._reopen_interval = float(reopen_interval_s)
        self._cap = None
        self._next_retry = 0.0
        self._open()

    def _open(self) -> None:
        """Open (or reopen) the V4L2 device."""
        cap = self._cv2.VideoCapture(self._device)
        if cap.isOpened():
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            cap.set(self._cv2.CAP_PROP_FPS, self._fps)
            self._cap = cap
            logger.info("[camera:%s] opened device %s", self.role, self._device)
        else:
            cap.release()
            self._cap = None
            self._next_retry = time.time() + self._reopen_interval
            logger.warning(
                "[camera:%s] device %s unavailable - will retry every %.0fs",
                self.role,
                self._device,
                self._reopen_interval,
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame, reopening the device on failure or after the retry interval."""
        if self._cap is None:
            if time.time() >= self._next_retry:
                self._open()
            if self._cap is None:
                return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            self._next_retry = time.time() + self._reopen_interval
            return False, None
        return True, frame

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """Discard buffered frames until a read blocks for a new exposure.

        ``settled_at`` is when the arm finished moving. A buffered frame
        returns immediately; the frame that blocks for the next exposure was
        captured after the move.
        """
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

    def close(self) -> None:
        """Release the OpenCV capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class Ros2Camera(CameraClient):
    """Latest-frame subscriber on a camera node's image topic.

    Keeps only the newest frame (best-effort QoS, depth 1) so vision can never
    back up behind the publisher, and decodes ``bgr8``/``rgb8`` messages without
    cv_bridge to avoid ABI coupling.
    """

    def __init__(self, role: str, topic: str, stale_after_s: float = 1.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self.role = role
        self._stale_after = float(stale_after_s)
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(f"camera_client_{role}")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(Image, topic, self._on_image, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        logger.info("[camera:%s] subscribed to %s", role, topic)

    def _on_image(self, msg) -> None:
        """Decode an incoming Image message into the cached latest frame."""
        if msg.encoding not in ("bgr8", "rgb8"):
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        if msg.encoding == "rgb8":
            frame = frame[..., ::-1]
        with self._lock:
            self._frame = frame.copy()
            self._stamp = time.time()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest frame if it isn't stale, else (False, None)."""
        with self._lock:
            if self._frame is None or (time.time() - self._stamp) > self._stale_after:
                return False, None
            return True, self._frame

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """Newest cached frame whose stamp is later than ``settled_at``."""
        deadline = time.time() + max(self._stale_after, 0.5)
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp > settled_at:
                    return True, self._frame
            time.sleep(0.02)
        return False, None

    def close(self) -> None:
        """Shut down the executor and destroy the subscriber node."""
        self._executor.shutdown()
        self._node.destroy_node()


class SimCameraClient(CameraClient):
    """A camera rendered by the in-process MuJoCo digital twin (``source: sim``).

    Shares the process-wide twin with ``SimArmClient`` so the wrist camera
    moves with the simulated arm. Over ROS 2 (``use_ros2: true``) the twin's
    node publishes the same frames on the normal topic instead.
    """

    def __init__(self, cfg: dict, role: str):
        from so101_sim import SimCamera

        from vla_pipeline.robot.sim_arm import sim_world

        self.role = role
        self._cam = SimCamera(sim_world(cfg), role, blocking=True)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Next rendered frame (paced at the camera fps, like a V4L2 read)."""
        return self._cam.read()

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """Frame rendered after the arm finished moving at ``settled_at``."""
        return self._cam.read_after(settled_at)

    def ok(self) -> bool:
        """Return True once the renderer has produced a frame."""
        return self._cam.ok()

    def close(self) -> None:
        """Release this view of the twin's renderer."""
        self._cam.release()


def make_camera(cfg: dict, role: str) -> CameraClient:
    """Build the CameraClient for ``cameras.<role>`` from the pipeline config."""
    cams = cfg.get("cameras") or {}
    c = cams.get(role)
    if c is None:  # backward compat: fall back to the legacy single `camera:`
        legacy = cfg.get("camera", {})
        c = {
            "device": legacy.get("index", 0),
            "width": legacy.get("width", 640),
            "height": legacy.get("height", 480),
            "fps": legacy.get("fps", 30),
            "use_ros2": False,
        }
    if c.get("use_ros2", False):
        return Ros2Camera(role, c["topic"], stale_after_s=c.get("stale_after_s", 1.0))
    if c.get("source", "device") == "sim":
        return SimCameraClient(cfg, role)
    return DirectCamera(
        role,
        c.get("device", 0),
        c.get("width", 640),
        c.get("height", 480),
        c.get("fps", 30),
    )


# =============================================================================
# Standalone component test (live preview)
# =============================================================================


def main() -> None:
    """CLI entry point: preview one camera role in a live OpenCV window."""
    import cv2

    from vla_pipeline.utils.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Camera client - component test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--role", choices=["operator", "arm"], default="operator")
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="force the ROS 2 transport regardless of config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ros2:
        cfg.setdefault("cameras", {}).setdefault(args.role, {})["use_ros2"] = True
    cam = make_camera(cfg, args.role)
    print(f"Previewing camera '{args.role}' - press 'q' to quit.")
    misses = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                misses += 1
                if misses % 30 == 1:
                    print("  (no frames - camera offline, retrying...)")
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
