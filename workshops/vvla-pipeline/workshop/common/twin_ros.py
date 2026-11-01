# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The digital twin behind ROS 2, for one notebook kernel (PROVIDED).

Section 5 of ``notebooks/03_integration/02_digital_twin.ipynb`` drives the
twin the way ``01_behavior.ipynb`` drives the real arm: a
:class:`~ros.arm_client.Ros2ArmClient` on ros2_control's controllers, and a
server node that owns the (simulated) bus. :class:`TwinSession` starts that
server - ``python -m so101_sim.node --notebook-twin``, its topics remapped
onto the hardware side's - and the controllers
(``launch/so101_control.launch.py``) as children of this kernel, on a ROS
domain of their own with discovery kept to this machine. The real arm's
nodes, other notebooks' demo nodes and ``app.py`` live on the default domain
and never see them.

The twin node adds what a console needs on top of the real topics (see
:mod:`so101_sim.node`): arm-camera frames stamped with the time their scene
was captured, the props (their TF frames, and on ``/diagnostics`` whether both
fingers hold them) and ``/sim/reset_scene``.
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

from common.ros_helpers import Spinner, from_image_msg
from ros.arm_client import Ros2ArmClient
from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, start_controllers, stop_controllers
from so101_sim.node import DIAGNOSTICS_TOPIC, GRIPPER_STATUS, PROPS_STATUS, VIEW_TOPIC, WORLD_FRAME

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_ID = 87  # override with TWIN_ROS_DOMAIN_ID
# Discovery stays on this machine. The profile does not turn builtin transports
# off, so FASTDDS_BUILTIN_TRANSPORTS from scripts/ryzen_ai_env.sh still applies.
_LOCALHOST_PROFILE = Path(__file__).with_name("twin_localhost.xml")
_domain_node = None  # kept so the DDS participant outlives the call


def twin_domain_id() -> int:
    """The ROS domain the twin runs on: ``TWIN_ROS_DOMAIN_ID``, else 87."""
    return int(os.environ.get("TWIN_ROS_DOMAIN_ID", DEFAULT_DOMAIN_ID))


def use_private_domain(domain_id: int) -> None:
    """Put this process's ROS 2 on ``domain_id``, discovery on this machine only.

    ROS reads the domain when ``rclpy.init()`` runs, so this must run before
    anything in this kernel starts ROS. A kernel already running ROS on
    another domain has to be restarted.

    ``twin_localhost.xml`` keeps discovery on this host (no multicast, peers
    at 127.0.0.1, other hosts ignored). ``ROS_AUTOMATIC_DISCOVERY_RANGE`` is
    not set to ``LOCALHOST``: that range disables Fast DDS builtin transports,
    and ``FASTDDS_BUILTIN_TRANSPORTS`` would no longer apply.

    Call this before the first Vitis AI session in the process. ``LARGE_DATA``
    does not discover peers if that session is opened before the DDS
    participant. ``rclpy.init`` alone does not create one; this function
    holds a node so the participant exists when it returns. The localhost
    discovery range used to hide the order, because it builds its own UDP
    transport afterwards. The pipeline's arm and camera clients start DDS
    before the models for the same reason.
    """
    global _domain_node
    import rclpy

    if rclpy.ok():
        current = rclpy.get_default_context().get_domain_id()
        if current != domain_id:
            raise RuntimeError(
                f"ROS 2 already runs in this kernel on domain {current}; the twin needs "
                f"domain {domain_id}. Restart the kernel and run the notebook from the top."
            )
    else:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST":
            del os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(_LOCALHOST_PROFILE)
        rclpy.init()
    if _domain_node is None:
        _domain_node = rclpy.create_node("twin_domain")


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


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


class TwinSession:
    """One ``so101_sim.node --notebook-twin`` and its controllers, owned by
    this process, and their clients.

    ``arm`` (:class:`~ros.arm_client.Ros2ArmClient`), ``armcam`` and ``view``
    (:class:`TwinCamera`) and ``scene`` (:class:`TwinScene`) are ready when
    the constructor returns. :meth:`close` stops the controllers and the
    node; the node also dies with this process, so a crashed kernel leaves
    no twin behind.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        config_path: str | os.PathLike | None = None,
        scene: dict | None = None,
        domain_id: int | None = None,
        ready_timeout_s: float = 60.0,
    ):
        from common.config import DEFAULT_CONFIG

        self.domain_id = twin_domain_id() if domain_id is None else int(domain_id)
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        r = cfg["robot"]
        self._arm_topic = (cfg.get("cameras") or {}).get("arm", {}).get(
            "topic", "/cameras/arm/image_raw"
        )
        self.arm = self.armcam = self.view = self.scene = None
        self._proc = self._controllers = None
        self._closed = False
        self._scene_file = None

        use_private_domain(self.domain_id)
        self._refuse_if_owned()
        self._probe = Spinner("twin_session")

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
        cmd += ["--ros-args", "-r", f"{r['command_topic']}:={HARDWARE_COMMANDS}",
                "-r", f"{r['state_topic']}:={HARDWARE_STATES}"]
        fd, log_name = tempfile.mkstemp(prefix="twin_node_", suffix=".log")
        self.log_path = Path(log_name)
        with os.fdopen(fd, "w") as log:
            # Its own session: Jupyter's stop button sends SIGINT to the kernel's
            # whole process group. The node still exits with this process.
            self._proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},  # the log is current when a test copies it
                start_new_session=True,
            )
        fd, log_name = tempfile.mkstemp(prefix="twin_controllers_", suffix=".log")
        self.controllers_log_path = Path(log_name)
        with os.fdopen(fd, "w") as log:
            self._controllers = start_controllers(self.config_path, "workshop", log=log)
        atexit.register(self.close)
        try:
            self._wait_ready(ready_timeout_s, cfg)
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    @property
    def controllers_pid(self) -> int | None:
        return None if self._controllers is None else self._controllers.pid

    def _refuse_if_owned(self) -> None:
        """The twin node holds its domain's lock (``so101_sim.domain_lock``) while it runs."""
        from so101_sim.domain_lock import DomainTaken, owner

        pid = owner(self.domain_id)
        if pid is not None:
            raise DomainTaken(self.domain_id, pid)

    def _exited(self) -> RuntimeError | None:
        if self._proc.poll() is None:
            return None
        return RuntimeError(f"the twin node exited with code {self._proc.returncode}:\n{_tail(self.log_path)}")

    def _wait_ready(self, timeout_s: float, cfg: dict) -> None:
        """The node's first joint state, then the arm client (the controllers
        up), then the first arm frame, within ``timeout_s``."""
        from sensor_msgs.msg import JointState

        from common.motion import MotionTimeout

        deadline = time.time() + timeout_s

        def remaining() -> float:
            return max(deadline - time.time(), 0.0)

        def not_ready(what: str) -> Exception:
            return self._exited() or TimeoutError(
                f"the twin node sent no {what} within {timeout_s:.0f} s:\n{_tail(self.log_path)}"
            )

        up = threading.Event()  # the first joint state, or the node's exit
        threading.Thread(target=lambda: (self._proc.wait(), up.set()), daemon=True, name="twin-exit").start()
        first = self._probe.node.create_subscription(JointState, HARDWARE_STATES, lambda _m: up.set(), _sensor_qos())
        try:
            up.wait(timeout_s)
        finally:
            self._probe.node.destroy_subscription(first)
        if self._exited() is not None or not up.is_set():
            raise not_ready("joint state")
        try:
            self.arm = Ros2ArmClient(cfg, state_timeout_s=remaining())
        except MotionTimeout as e:
            raise TimeoutError(
                f"{e}; the controllers' launch {'exited' if self._controllers.poll() is not None else 'runs'}:\n"
                f"{_tail(self.controllers_log_path)}"
            ) from None
        self.armcam = TwinCamera("arm", self._arm_topic)
        self.view = TwinCamera("view", VIEW_TOPIC)
        self.scene = TwinScene()
        if not self.armcam.next_frame(0.0, remaining())[0]:
            raise not_ready("arm camera frame")
        logger.info(
            "twin up on ROS domain %d (pid %d, log %s)", self.domain_id, self.pid, self.log_path
        )

    def reset_scene(self) -> bool:
        """Park the arm with the jaws open and put the props back (in the twin):
        :meth:`~common.ros2_control_arm.ControllerArm.reset_twin`."""
        return self.arm.reset_twin()

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
        if self._controllers is not None:
            try:
                stop_controllers(self._controllers)
            except RuntimeError as e:
                logger.warning("%s", e)
            if self._controllers.returncode == 0:
                self.controllers_log_path.unlink(missing_ok=True)  # kept when they failed
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
