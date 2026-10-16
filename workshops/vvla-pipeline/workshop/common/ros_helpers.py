# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 message (de)serialization helpers (PROVIDED).

Converting an ndarray to a ``sensor_msgs/Image`` (and back) and packing a
``JointState`` are pure byte plumbing - identical in every ROS project - so
they are provided. The *wiring* (nodes, publishers, subscriptions, QoS,
executors) is yours: see ``ros/``.

All imports of ROS message types happen inside the functions so this module
imports cleanly on machines without ROS 2 (and so the offline self-test can
substitute its in-memory ROS shim, ``common.fakeros``, before first use).
"""

from __future__ import annotations

import math

import numpy as np

# Gripper Torque_Limit (std_msgs/Int32, 0-1000): written by the server node's
# control loop, the only thread on the bus.
GRIPPER_TORQUE_TOPIC = "/so101/gripper_torque_limit"


def ros2_available() -> bool:
    """True when ``rclpy`` (real or shimmed) is importable."""
    try:
        import rclpy  # noqa: F401

        return True
    except ImportError:
        return False


def to_image_msg(frame_bgr: np.ndarray, node=None, frame_id: str = ""):
    """Pack a BGR ndarray into a ``sensor_msgs/Image`` (encoding ``bgr8``)."""
    from sensor_msgs.msg import Image

    msg = Image()
    if node is not None:
        msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = frame_bgr.tobytes()
    return msg


def from_image_msg(msg):
    """Decode a ``bgr8``/``rgb8`` Image message into a BGR ndarray (or None)."""
    if msg.encoding not in ("bgr8", "rgb8"):
        return None
    frame = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        msg.height, msg.width, 3
    )
    if msg.encoding == "rgb8":
        frame = frame[..., ::-1]
    return frame.copy()


def joint_state_msg(names, positions, effort=None, node=None):
    """Build a (optionally time-stamped) ``sensor_msgs/JointState``."""
    from sensor_msgs.msg import JointState

    msg = JointState()
    if node is not None:
        msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    if effort is not None:
        msg.effort = [float(e) for e in effort]
    return msg


def parse_joint_state(msg):
    """Unpack a JointState into ``(positions dict, gripper_effort or None)``.

    The server publishes the gripper servo's current in ``effort`` aligned
    with ``name``; NaN means "feedback unavailable" and maps to ``None``.
    """
    positions = {n: float(p) for n, p in zip(msg.name, msg.position)}
    effort = None
    if msg.effort:
        for n, e in zip(msg.name, msg.effort):
            if n == "gripper" and not math.isnan(e):
                effort = float(e)
    return positions, effort
