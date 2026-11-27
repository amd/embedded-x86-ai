# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Clients for the digital twin that ``python -m so101_sim.stack`` already started.

``run_pipeline.sh`` and ``run_notebooks.sh`` start that module. Section 5 of
``notebooks/03_integration/02_digital_twin.ipynb`` then connects a
:class:`~ros.arm_client.Ros2ArmClient`, the arm and view cameras, and
:class:`TwinScene`. This module does not start a launch, pick a ROS domain,
or stop the stack processes.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time

from common.ros_helpers import Spinner, from_image_msg
from ros.arm_client import Ros2ArmClient
from so101_sim.node import DIAGNOSTICS_TOPIC, GRIPPER_STATUS, PROPS_STATUS, VIEW_TOPIC, WORLD_FRAME

logger = logging.getLogger(__name__)

def twin_domain_id() -> int:
    """The ROS domain the twin runs on. The number lives in ``so101_sim.bringup``."""
    from so101_sim.bringup import twin_domain_id as domain_id

    return domain_id()


def use_private_domain(domain_id: int) -> None:
    """Pin this process to ``domain_id``. The participant is ``so101_sim.bringup``."""
    from so101_sim.bringup import ensure_dds_participant

    ensure_dds_participant(int(domain_id))


def _sensor_qos():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )


class TwinCamera:
    """Newest frame of a twin camera topic, with the time its scene was captured.

    Same ``read() -> (ok, frame)`` as the other cameras. ``last_stamp`` is
    the capture time of the frame the last ``read()`` returned, so
    :class:`~common.twin_console.FanoutCamera` and :meth:`read_after` can
    tell a frame of the settled arm from one rendered just before.
    """

    def __init__(self, role: str, topic: str, stale_after_s: float = 1.0):
        from sensor_msgs.msg import Image

        self.role = role
        self.topic = topic
        self._stale_after = float(stale_after_s)
        self._cond = threading.Condition()
        self._frame = None
        self._stamp = 0.0
        self._arrived = 0.0
        self._closed = False
        self.last_stamp: float | None = None
        self._spin = Spinner(f"twin_camera_{role}")
        self._spin.node.create_subscription(Image, topic, self._on_image, _sensor_qos())

    def _on_image(self, msg) -> None:
        frame = from_image_msg(msg)
        if frame is None:
            return
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self._cond:
            self._frame, self._stamp, self._arrived = frame, stamp, time.time()
            self._cond.notify_all()

    def read(self):
        """Newest frame, or ``(False, None)`` when the stream is stale or missing."""
        with self._cond:
            if self._frame is None or time.time() - self._arrived > self._stale_after:
                return False, None
            self.last_stamp = self._stamp
            return True, self._frame

    def read_after(self, settled_at: float):
        """First frame whose scene was captured after ``settled_at``, within the stream's staleness limit."""
        ok, frame, stamp = self.next_frame(settled_at, max(self._stale_after, 0.5))
        if ok:
            self.last_stamp = stamp
        return ok, frame

    def next_frame(self, after: float, timeout: float | None = None):
        """``(ok, frame, captured)``: the first frame captured after ``after``;
        not ok once closed, or when ``timeout`` passes first."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._frame is not None and self._stamp > after), timeout)
            if self._closed or self._frame is None or self._stamp <= after:
                return False, None, after
            return True, self._frame, self._stamp

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._spin.close()

    release = close


class TwinScene:
    """The twin's props as it publishes them (``so101_sim.node.PropPublisher``):
    each prop's ``xyz`` and ``quat`` (w, x, y, z) from its TF frame, ``grasped``
    and ``settled`` and the ``gripper`` (``torque_limit``, ``overload_torque``,
    ``tripped``) from ``/diagnostics``. A scene is one stamp: the diagnostics
    and the props' frames of one tick."""

    _KEEP = 20  # stamps of prop frames kept while their diagnostics are on the way

    def __init__(self):
        from diagnostic_msgs.msg import DiagnosticArray
        from tf2_msgs.msg import TFMessage

        self._cond = threading.Condition()
        self._latest: dict | None = None
        self._frames: dict[float, dict] = {}  # stamp -> {prop: (xyz, quat)}
        self._status: dict[float, dict] = {}  # stamp -> {status name: {key: value}}
        self._spin = Spinner("twin_scene")
        self._spin.node.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self._spin.node.create_subscription(DiagnosticArray, DIAGNOSTICS_TOPIC, self._on_diagnostics, 10)

    @staticmethod
    def _stamp(header) -> float:
        return header.stamp.sec + 1e-9 * header.stamp.nanosec

    def _on_tf(self, msg) -> None:
        with self._cond:
            for tf in msg.transforms:
                if tf.header.frame_id != WORLD_FRAME:
                    continue
                t, r = tf.transform.translation, tf.transform.rotation
                frames = self._frames.setdefault(self._stamp(tf.header), {})
                frames[tf.child_frame_id] = ([t.x, t.y, t.z], [r.w, r.x, r.y, r.z])
            while len(self._frames) > self._KEEP:
                del self._frames[min(self._frames)]
            self._complete()

    def _on_diagnostics(self, msg) -> None:
        status = {s.name: {kv.key: kv.value for kv in s.values} for s in msg.status}
        if PROPS_STATUS not in status:
            return
        with self._cond:
            self._status[self._stamp(msg.header)] = status
            self._complete()

    def _complete(self) -> None:
        """Every stamp whose diagnostics and prop frames have all arrived becomes the latest scene."""
        for stamp in sorted(self._status):
            props, gripper = self._status[stamp][PROPS_STATUS], self._status[stamp].get(GRIPPER_STATUS, {})
            names = [key[: -len(".grasped")] for key in props if key.endswith(".grasped")]
            frames = self._frames.get(stamp, {})
            if not all(name in frames for name in names):
                continue
            self._latest = {
                "stamp": stamp,
                "bodies": {
                    name: {"xyz": frames[name][0], "quat": frames[name][1], "grasped": props[f"{name}.grasped"] == "true"}
                    for name in names
                },
                "settled": props["settled"] == "true",
                "gripper": {
                    "torque_limit": int(gripper["torque_limit"]),
                    "overload_torque": int(gripper["overload_torque"]),
                    "tripped": gripper["tripped"] == "true",
                },
            }
            self._cond.notify_all()
        if self._latest is not None:
            for kept in (self._frames, self._status):
                for stamp in [s for s in kept if s <= self._latest["stamp"]]:
                    del kept[stamp]

    def latest(
        self, after: float | None = None, timeout: float | None = 2.0, settled: bool = False,
    ) -> dict | None:
        """The newest scene, or the first one stamped after ``after`` (with
        ``settled``, with every prop at rest); None when ``timeout`` passes
        first (``None``: no limit)."""

        def ready() -> bool:
            s = self._latest
            return (
                s is not None
                and (after is None or s["stamp"] > after)
                and (not settled or s["settled"])
            )

        with self._cond:
            if not self._cond.wait_for(ready, timeout):
                return None
            return self._latest

    def reading(self, body: str, after: float | None = None, timeout: float = 2.0) -> dict | None:
        """``{"xyz", "quat", "grasped", "stamp"}`` for ``body``, or None."""
        scene = self.latest(after=after, timeout=timeout)
        if scene is None or body not in scene["bodies"]:
            return None
        return {**scene["bodies"][body], "stamp": scene["stamp"]}

    def close(self) -> None:
        self._spin.close()


class TwinSession:
    """Clients of the launch ``python -m so101_sim.stack`` already started.

    ``arm`` (:class:`~ros.arm_client.Ros2ArmClient`), ``armcam`` and ``view``
    (:class:`TwinCamera`) and ``scene`` (:class:`TwinScene`) are the clients.
    :meth:`close` closes those clients. The shell owns the stack processes.
    """

    def __init__(self, cfg: dict):
        self._arm_topic = (cfg.get("cameras") or {}).get("arm", {}).get(
            "topic", "/cameras/arm/image_raw"
        )
        self.arm = self.armcam = self.view = self.scene = None
        self._closed = False
        self._require_launch()
        atexit.register(self.close)
        try:
            self.arm = Ros2ArmClient(cfg)
            self.armcam = TwinCamera("arm", self._arm_topic)
            self.view = TwinCamera("view", VIEW_TOPIC)
            self.scene = TwinScene()
        except BaseException:
            self.close()
            raise

    def _require_launch(self) -> None:
        """The twin node holds the domain lock. No holder means the launch was not started."""
        from so101_sim.domain_lock import owner

        domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        if owner(domain) is None:
            raise RuntimeError(
                "the digital twin is not running. Start it with ./run_pipeline.sh --digital-twin"
            )

    def reset_scene(self) -> bool:
        """Park the arm with the jaws open and put the props back (in the twin):
        :meth:`~common.ros2_control_arm.ControllerArm.reset_twin`."""
        return self.arm.reset_twin()

    def close(self) -> None:
        """Close the clients. Safe to call more than once. The launch keeps running."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        for part in (self.arm, self.armcam, self.view, self.scene):
            if part is None:
                continue
            try:
                part.close()
            except Exception as e:
                logger.debug("twin client close: %s", e)
