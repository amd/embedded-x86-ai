# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The digital twin behind ROS 2, for one notebook kernel (PROVIDED).

Section 5 of ``notebooks/03_integration/02_digital_twin.ipynb`` drives the
twin the way ``01_behavior.ipynb`` drives the real arm: a
:class:`~ros.arm_client.Ros2ArmClient` publishing on ``robot.command_topic``,
and a server node that owns the (simulated) bus. :class:`TwinSession` starts
that server - ``python -m so101_sim.node --notebook-twin`` - as a child of
this kernel, on a ROS domain of its own with discovery kept to this machine.
The real arm's nodes, other notebooks' demo nodes and ``app.py`` live on the
default domain and never see it.

The twin node adds what a console needs on top of the real topics (see
:mod:`so101_sim.node`): arm-camera frames stamped with the time their scene
was captured, ``/sim/scene`` (where each prop is, and whether both fingers
hold it) and ``/sim/reset_scene``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from common.motion import clamp_joints
from common.ros_helpers import from_image_msg, joint_state_msg
from ros.arm_client import Ros2ArmClient
from so101_sim.node import RESET_SERVICE, SCENE_TOPIC, VIEW_TOPIC

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_ID = 87  # override with TWIN_ROS_DOMAIN_ID


def twin_domain_id() -> int:
    """The ROS domain the twin runs on: ``TWIN_ROS_DOMAIN_ID``, else 87."""
    return int(os.environ.get("TWIN_ROS_DOMAIN_ID", DEFAULT_DOMAIN_ID))


def use_private_domain(domain_id: int) -> None:
    """Put this process's ROS 2 on ``domain_id``, discovery on this machine only.

    ROS reads both variables when ``rclpy.init()`` runs, so they must be set
    before anything in this kernel starts ROS. A kernel already running ROS
    on another domain has to be restarted.
    """
    import rclpy

    if rclpy.ok():
        current = rclpy.get_default_context().get_domain_id()
        if current != domain_id:
            raise RuntimeError(
                f"ROS 2 already runs in this kernel on domain {current}; the twin needs "
                f"domain {domain_id}. Restart the kernel and run the notebook from the top."
            )
        return
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init()


def _sensor_qos():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )


class _Spinner:
    """One rclpy node on its own executor thread, joined by :meth:`close`.

    A thread still inside ``executor.spin()`` when Python exits aborts the
    process ("terminate called without an active exception"), and
    ``executor.shutdown()`` does not end it. So the thread spins in short
    slices and ``close()`` joins it before shutting the executor down.
    """

    def __init__(self, name: str):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self.node = rclpy.create_node(name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def _loop(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass


class TwinArmClient(Ros2ArmClient):
    """:class:`Ros2ArmClient` that keeps the limits a move was sent with.

    ``go_to(..., limits=GRASP_LIMITS)`` sets ``_send_limits`` for the move.
    The base client clamps every command to the desk cap, which pulls a
    90 deg standing-grasp wrist back to 67.5 deg before the twin sees it.
    Its executor runs on a :class:`_Spinner`, so closing it leaves no
    spinning thread behind.
    """

    def _setup_ros(self, r: dict) -> None:
        from sensor_msgs.msg import JointState

        self._spin = _Spinner("twin_arm_client")
        self._node = self._spin.node
        self._pub = self._node.create_publisher(JointState, r["command_topic"], _sensor_qos())
        self._node.create_subscription(JointState, r["state_topic"], self._on_state, _sensor_qos())

    def send_joints(self, joints: dict) -> None:
        joints = clamp_joints(joints, self._send_limits)
        names = list(joints.keys())
        self._pub.publish(joint_state_msg(names, [joints[k] for k in names], node=self._node))

    def close(self) -> None:
        self._spin.close()


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
        self.last_stamp: float | None = None
        self._spin = _Spinner(f"twin_camera_{role}")
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
        """First frame whose scene was captured after ``settled_at``."""
        deadline = time.time() + max(self._stale_after, 0.5)
        with self._cond:
            while self._frame is None or self._stamp <= settled_at:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=remaining)
            self.last_stamp = self._stamp
            return True, self._frame

    def close(self) -> None:
        self._spin.close()

    release = close


class TwinScene:
    """The latest ``/sim/scene``: each prop's ``xyz``, ``quat`` and ``grasped``."""

    def __init__(self, topic: str = SCENE_TOPIC):
        from std_msgs.msg import String

        self._cond = threading.Condition()
        self._latest: dict | None = None
        self._spin = _Spinner("twin_scene")
        self._spin.node.create_subscription(String, topic, self._on_scene, _sensor_qos())

    def _on_scene(self, msg) -> None:
        data = json.loads(msg.data)
        with self._cond:
            self._latest = data
            self._cond.notify_all()

    def latest(self, after: float | None = None, timeout: float = 2.0) -> dict | None:
        """The newest scene, or the first one stamped after ``after``."""
        deadline = time.time() + timeout
        with self._cond:
            while self._latest is None or (after is not None and self._latest["stamp"] <= after):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return self._latest

    def reading(self, body: str, after: float | None = None, timeout: float = 2.0) -> dict | None:
        """``{"xyz", "quat", "grasped", "stamp"}`` for ``body``, or None."""
        scene = self.latest(after=after, timeout=timeout)
        if scene is None or body not in scene["bodies"]:
            return None
        return {**scene["bodies"][body], "stamp": scene["stamp"]}

    def close(self) -> None:
        self._spin.close()


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


class TwinSession:
    """One ``so101_sim.node --notebook-twin`` owned by this process, and its clients.

    ``arm`` (:class:`TwinArmClient`), ``armcam`` and ``view``
    (:class:`TwinCamera`) and ``scene`` (:class:`TwinScene`) are ready when
    the constructor returns. :meth:`close` stops the node; it also dies with
    this process, so a crashed kernel leaves no twin behind.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        config_path: str | os.PathLike | None = None,
        scene: dict | None = None,
        domain_id: int | None = None,
        ready_timeout_s: float = 60.0,
        discovery_s: float = 2.0,
    ):
        from common.config import DEFAULT_CONFIG

        self.domain_id = twin_domain_id() if domain_id is None else int(domain_id)
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        r = cfg["robot"]
        self._state_topic = r["state_topic"]
        self._arm_topic = (cfg.get("cameras") or {}).get("arm", {}).get(
            "topic", "/cameras/arm/image_raw"
        )
        self.arm = self.armcam = self.view = self.scene = None
        self._proc = None
        self._closed = False
        self._scene_file = None

        use_private_domain(self.domain_id)
        self._probe = _Spinner("twin_session")
        self._refuse_if_owned(discovery_s)

        from std_srvs.srv import Trigger

        self._Trigger = Trigger
        self._reset_client = self._probe.node.create_client(Trigger, RESET_SERVICE)

        cmd = [
            sys.executable, "-m", "so101_sim.node",
            "--config", str(self.config_path), "--stack", "workshop",
            "--sim-cameras", "arm", "--notebook-twin",
        ]
        if scene is not None:
            fd, name = tempfile.mkstemp(prefix="twin_scene_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(scene, f)
            self._scene_file = Path(name)
            cmd += ["--scene", name]
        fd, log_name = tempfile.mkstemp(prefix="twin_node_", suffix=".log")
        self.log_path = Path(log_name)
        with os.fdopen(fd, "w") as log:
            self._proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
            )
        atexit.register(self.close)
        try:
            self._wait_ready(ready_timeout_s, cfg)
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def _publishers(self, topic: str) -> int:
        return int(self._probe.node.count_publishers(topic))

    def _refuse_if_owned(self, discovery_s: float) -> None:
        deadline = time.time() + discovery_s
        while time.time() < deadline:
            if self._publishers(self._state_topic):
                self._probe.close()
                raise RuntimeError(
                    f"a twin already publishes {self._state_topic} on ROS domain "
                    f"{self.domain_id}. Close the notebook that runs it, or find it with: "
                    f"ROS_DOMAIN_ID={self.domain_id} ros2 node list"
                )
            time.sleep(0.1)

    def _wait_ready(self, timeout_s: float, cfg: dict) -> None:
        deadline = time.time() + timeout_s

        def remaining() -> float:
            left = deadline - time.time()
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"the twin node exited with code {self._proc.returncode}:\n{_tail(self.log_path)}"
                )
            if left <= 0:
                raise TimeoutError(
                    f"the twin node was not ready within {timeout_s:.0f} s:\n{_tail(self.log_path)}"
                )
            return left

        while not (self._publishers(self._state_topic) and self._publishers(self._arm_topic)):
            remaining()
            time.sleep(0.1)
        self.arm = TwinArmClient(cfg, state_timeout_s=min(remaining(), 10.0))
        self.armcam = TwinCamera("arm", self._arm_topic)
        self.view = TwinCamera("view", VIEW_TOPIC)
        self.scene = TwinScene()
        while not self.armcam.read()[0]:
            remaining()
            time.sleep(0.05)
        while self.scene.latest(timeout=0.1) is None:
            remaining()
        logger.info(
            "twin up on ROS domain %d (pid %d, log %s)", self.domain_id, self.pid, self.log_path
        )

    def reset_scene(self, timeout_s: float = 10.0) -> bool:
        """Park the arm with the jaws open and put the props back (in the twin)."""
        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            return False
        future = self._reset_client.call_async(self._Trigger.Request())
        deadline = time.time() + timeout_s
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.02)
        result = future.result()
        return bool(result and result.success)

    def close(self) -> None:
        """Stop the clients and the twin node. Safe to call more than once."""
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
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if self._scene_file is not None:
            self._scene_file.unlink(missing_ok=True)
        if proc is not None and proc.returncode == 0:
            self.log_path.unlink(missing_ok=True)  # kept when the node failed
        self._probe.close()
