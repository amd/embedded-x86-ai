# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 camera node - one process per camera (operator / arm).

Publishes ``sensor_msgs/Image`` (bgr8) on a configurable topic at a
configurable rate, plus a ``std_msgs/Bool`` health flag so the orchestrator
and ``ros2 topic echo`` can see at a glance whether the device is alive.

All hardware specifics are **ROS parameters** (declared with defaults from
``config/pipeline.yaml``), so the launch file - not the code - decides device
IDs and resolutions:

    ros2 run ...  or  python -m vla_pipeline.vision.camera_node --role arm

Parameters
----------
``device`` (int or str), ``width``, ``height``, ``fps``, ``topic``,
``rotate`` (0/90/180/270 - arm-mounted cameras are often rotated),
``reopen_interval_s``.

Graceful failure: if the device can't be opened (unplugged, busy), the node
stays up, publishes ``healthy=False``, and retries on a timer instead of
crashing the launch.

Independent test
----------------
    # Terminal 1
    python -m vla_pipeline.vision.camera_node --role operator
    # Terminal 2
    ros2 topic hz /cameras/operator/image_raw
"""

from __future__ import annotations

import argparse
import logging

import cv2

from vla_pipeline.utils.config import load_config

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


class CameraNode(Node):
    """Owns one V4L2 device and publishes frames + health.

    ``frame_source`` (any object with ``read() -> (ok, frame)`` and
    ``release()``, e.g. ``so101_sim.SimCamera``) replaces the V4L2 device -
    that is how the digital twin publishes its wrist camera on the same topic.
    """

    def __init__(self, role: str, defaults: dict, frame_source=None):
        super().__init__(f"camera_{role}_node")
        self.role = role
        self._source = frame_source
        self.declare_parameter("device", defaults.get("device", 0))
        self.declare_parameter("width", int(defaults.get("width", 640)))
        self.declare_parameter("height", int(defaults.get("height", 480)))
        self.declare_parameter("fps", float(defaults.get("fps", 30.0)))
        self.declare_parameter(
            "topic", defaults.get("topic", f"/cameras/{role}/image_raw")
        )
        self.declare_parameter("rotate", int(defaults.get("rotate", 0)))
        self.declare_parameter(
            "reopen_interval_s", float(defaults.get("reopen_interval_s", 3.0))
        )

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._device = p("device")
        self._size = (int(p("width")), int(p("height")))
        self._fps = float(p("fps"))
        self._rotate = int(p("rotate"))
        self._reopen_interval = float(p("reopen_interval_s"))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Image, str(p("topic")), qos)
        self._health_pub = self.create_publisher(Bool, f"/cameras/{role}/healthy", qos)

        self._cap: cv2.VideoCapture | None = None
        self._open()
        self._frame_timer = self.create_timer(1.0 / self._fps, self._tick)
        self._retry_timer = self.create_timer(self._reopen_interval, self._retry)
        self.get_logger().info(
            f"camera '{role}' device={'sim' if frame_source is not None else self._device} {self._size[0]}x{self._size[1]}"
            f"@{self._fps:.0f} → {p('topic')}"
        )

    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the V4L2 device and configure its resolution/fps/buffer size."""
        if self._source is not None:
            self._cap = self._source
            return
        cap = cv2.VideoCapture(self._device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always-fresh frames
            self._cap = cap
        else:
            cap.release()
            self._cap = None
            self.get_logger().warning(
                f"camera '{self.role}' device {self._device} unavailable - retrying"
            )

    def _retry(self) -> None:
        """Timer callback: attempt to reopen the device if it isn't open."""
        if self._cap is None:
            self._open()

    def _tick(self) -> None:
        """Timer callback: publish one frame (and the health flag) if the device is open."""
        # During shutdown (SIGINT from the launcher) a timer can fire once more
        # after the context is torn down; skip rather than crash with RCLError.
        if not rclpy.ok():
            return
        healthy = False
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok:
                if self._rotate in _ROTATIONS:
                    frame = cv2.rotate(frame, _ROTATIONS[self._rotate])
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"camera_{self.role}"
                msg.height, msg.width = frame.shape[:2]
                msg.encoding = "bgr8"
                msg.step = msg.width * 3
                msg.data = frame.tobytes()
                self._pub.publish(msg)
                healthy = True
            elif self._source is None:
                self._cap.release()
                self._cap = None
                self.get_logger().warning(
                    f"camera '{self.role}' read failed - reopening"
                )
        try:
            self._health_pub.publish(Bool(data=healthy))
        except Exception:
            pass  # context torn down mid-shutdown

    def destroy_node(self) -> None:
        """Release the capture device before the base class teardown."""
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main() -> None:
    """CLI entry point: spin a single-camera ROS 2 publisher node."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not ROS2_AVAILABLE:
        raise SystemExit("rclpy not importable - source ROS 2 Jazzy before .venv.")
    parser = argparse.ArgumentParser(description="Camera publisher node")
    parser.add_argument("--role", choices=["operator", "arm"], required=True)
    parser.add_argument("--config", default=None)
    args, ros_args = parser.parse_known_args()

    defaults = (load_config(args.config).get("cameras") or {}).get(args.role, {})
    rclpy.init(args=ros_args)
    node = CameraNode(args.role, defaults)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as e:
        # RCLError on a publish during teardown is expected when the launcher
        # SIGINTs us mid-tick; anything else is worth surfacing.
        if "context is invalid" not in str(e) and "shutdown" not in str(e).lower():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
