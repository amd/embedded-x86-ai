# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 camera node - one process per camera (operator / arm).

The server-node half of the camera transport: this node OWNS the V4L2
device and publishes ``sensor_msgs/Image`` (bgr8) plus a ``std_msgs/Bool``
health flag. Everything downstream (behaviors, previews, the harness)
subscribes; nobody else touches the device.

The QoS choice is the lesson here: **best-effort, keep-last, depth 1**.
Vision must never back up behind a slow consumer - the latest frame always
wins, and a dropped frame costs nothing because another is 33 ms behind it.

Capture plumbing (device open/rotate/retry) is provided; the ROS wiring -
publishers, the frame timer, message publishing - is yours.

Notebook: ``notebooks/02_ros2/01_cameras.ipynb``
Run it for real:
    python -m ros.camera_node --role operator              # real device
    python -m ros.camera_node --role operator --synthetic  # no camera needed
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from common.config import load_config
from common.ros_helpers import to_image_msg

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class _CvCapture:
    """PROVIDED: cv2.VideoCapture with the pipeline's settings applied.

    Graceful failure is part of the node contract: when the device drops
    (unplugged, busy) ``read`` returns ``(False, None)`` - so the node keeps
    running and publishes ``healthy: false`` - and the capture retries the
    device every ``REOPEN_EVERY_S`` until it comes back. Behaviors degrade
    instead of dying.
    """

    REOPEN_EVERY_S = 3.0

    def __init__(self, device, width, height, fps):
        self._device = device
        self._size = (int(width), int(height))
        self._fps = float(fps)
        self._cap = None
        self._next_reopen = 0.0
        self._open()

    def _open(self):
        """(Re)open the configured device, falling back to webcam 0."""
        # The configured device first, then the default webcam. A configured
        # path that does not exist is skipped outright: probing it makes
        # OpenCV warn noisily, and V4L2 paths are opened with the V4L2
        # backend directly for the same reason.
        candidates = [self._device, 0]
        if isinstance(self._device, str) and not Path(self._device).exists():
            candidates = [0]
        for candidate in candidates:
            cap = (
                cv2.VideoCapture(candidate, cv2.CAP_V4L2)
                if isinstance(candidate, str)
                else cv2.VideoCapture(candidate)
            )
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
                cap.set(cv2.CAP_PROP_FPS, self._fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always-fresh frames
                self._cap = cap
                if candidate != self._device:
                    logger.info(
                        "camera device %s absent - using %s instead",
                        self._device,
                        candidate,
                    )
                return
            cap.release()
        self._cap = None
        logger.warning(
            "camera device %s unavailable - node stays up, " "retrying", self._device
        )

    def read(self):
        """Read one frame, retrying the device on failure every ``REOPEN_EVERY_S``."""
        if self._cap is None:
            now = time.monotonic()
            if now >= self._next_reopen:  # periodic reopen attempt
                self._next_reopen = now + self.REOPEN_EVERY_S
                self._open()
            if self._cap is None:
                return False, None
        ok, frame = self._cap.read()
        if not ok:  # device dropped mid-stream
            self._cap.release()
            self._cap = None
            self._next_reopen = time.monotonic() + self.REOPEN_EVERY_S
            logger.warning(
                "camera read failed - node stays up, healthy=False, "
                "retrying the device"
            )
            return False, None
        return ok, frame

    def close(self):
        """Release the capture device, if open."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraNode(Node):
    """Owns one camera and publishes frames + health.

    ``frame_source`` is injectable (any object with ``read() -> (ok, frame)``)
    so the self-test can run this node on synthetic frames - the ROS wiring
    under test is identical either way.
    """

    def __init__(self, role: str, defaults: dict, frame_source=None):
        super().__init__(f"camera_{role}_node")
        self.role = role
        self.topic = str(defaults.get("topic", f"/cameras/{role}/image_raw"))
        self.health_topic = f"/cameras/{role}/healthy"
        self.fps = float(defaults.get("fps", 30.0))
        self._rotate = int(defaults.get("rotate", 0))
        self._source = frame_source or _CvCapture(
            defaults.get("device", 0),
            defaults.get("width", 640),
            defaults.get("height", 480),
            self.fps,
        )
        self._pub = None
        self._health_pub = None
        self._setup_ros()
        self.get_logger().info(f"camera '{role}' → {self.topic} @ {self.fps:.0f} fps")

    def _setup_ros(self) -> None:
        """Create the image + health publishers and the frame timer.

        Both publishers use sensor-stream QoS: best-effort reliability,
        keep-last history, depth 1. The timer fires ``self._tick`` at the
        configured frame rate.
        """
        # >>> TODO 2.1: CameraNode._setup_ros - notebooks/02_ros2/01_cameras.ipynb
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Image, self.topic, qos)
        self._health_pub = self.create_publisher(Bool, self.health_topic, qos)
        self.create_timer(1.0 / self.fps, self._tick)
        # <<< TODO 2.1

    def _tick(self) -> None:
        """PROVIDED: capture + rotate, then hand off to your publisher."""
        ok, frame = self._source.read()
        if ok and self._rotate in _ROTATIONS:
            frame = cv2.rotate(frame, _ROTATIONS[self._rotate])
        self._publish(frame if ok else None)

    def _publish(self, frame) -> None:
        """Publish one frame (or the unhealthy flag when ``frame is None``).

        Pack the ndarray with :func:`common.ros_helpers.to_image_msg`
        (pass ``node=self`` so the header gets a real timestamp), publish it,
        and always publish the health flag - subscribers use it to tell
        "camera offline" from "no news".
        """
        # >>> TODO 2.2: CameraNode._publish - notebooks/02_ros2/01_cameras.ipynb
        healthy = frame is not None
        if healthy:
            self._pub.publish(
                to_image_msg(frame, node=self, frame_id=f"camera_{self.role}")
            )
        self._health_pub.publish(Bool(data=healthy))
        # <<< TODO 2.2

    def destroy_node(self) -> None:
        """Close the frame source, then destroy the ROS node."""
        try:
            self._source.close()
        except Exception:
            pass
        super().destroy_node()


def main() -> None:
    """CLI: bring up a camera node (real or synthetic) and spin until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not ROS2_AVAILABLE:
        raise SystemExit(
            "rclpy not importable - source ROS 2 before running "
            "(offline checks: python -m ros.selftest)"
        )
    parser = argparse.ArgumentParser(description="Camera publisher node")
    parser.add_argument("--role", choices=["operator", "arm"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="publish a synthetic test pattern (no device)",
    )
    args, ros_args = parser.parse_known_args()

    defaults = (load_config(args.config).get("cameras") or {}).get(args.role, {})
    source = None
    if args.synthetic:
        from common.fixtures import SyntheticCamera

        source = SyntheticCamera(
            int(defaults.get("width", 640)), int(defaults.get("height", 480))
        )
    rclpy.init(args=ros_args)
    node = CameraNode(args.role, defaults, frame_source=source)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
