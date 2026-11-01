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
imports before ROS 2 is sourced (the model notebooks import it that way).
"""

from __future__ import annotations

import math

import numpy as np


class Spinner:
    """One rclpy node on its own executor thread, joined by :meth:`close`.

    A thread still inside ``executor.spin()`` when Python exits aborts the
    process ("terminate called without an active exception"), and
    ``executor.shutdown()`` does not end it. So the thread runs one callback
    at a time, and ``close()`` triggers a guard condition that wakes it to
    see the stop, then joins it before shutting the executor down.
    """

    def __init__(self, name: str):
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(name)
        self._wake = self.node.create_guard_condition(lambda: None)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def add_node(self, node) -> None:
        """Spin ``node`` on this thread too; its owner still destroys it."""
        self._executor.add_node(node)
        self._wake.trigger()

    def _loop(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once()
            except ExternalShutdownException:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.trigger()
        self._thread.join(timeout=2.0)
        self._executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass


def ros2_available() -> bool:
    """True when ``rclpy`` is importable (ROS 2 is sourced)."""
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


def joint_state_msg(names, positions, effort=None, node=None, velocity=None):
    """Build a (optionally time-stamped) ``sensor_msgs/JointState``."""
    from sensor_msgs.msg import JointState

    msg = JointState()
    if node is not None:
        msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    if velocity is not None:
        msg.velocity = [float(v) for v in velocity]
    if effort is not None:
        msg.effort = [float(e) for e in effort]
    return msg


def parse_joint_state(msg):
    """Unpack a JointState into ``(positions dict, gripper_effort or None)``.

    The server publishes the gripper's torque in N·m in ``effort`` aligned
    with ``name``; NaN means "feedback unavailable" and maps to ``None``.
    """
    positions = {n: float(p) for n, p in zip(msg.name, msg.position)}
    effort = None
    if msg.effort:
        for n, e in zip(msg.name, msg.effort):
            if n == "gripper" and not math.isnan(e):
                effort = float(e)
    return positions, effort


def joint_velocity(msg, name: str) -> float:
    """``name``'s velocity in a JointState; 0.0 when it carries none (or NaN)."""
    for n, v in zip(msg.name, msg.velocity):
        if n == name and not math.isnan(v):
            return float(v)
    return 0.0
