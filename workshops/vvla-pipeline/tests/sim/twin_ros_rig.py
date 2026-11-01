# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier-2 helpers: the notebook twin (``so101_sim.node --notebook-twin``) over ROS 2.

Every tier-2 test runs on one private ROS domain, ``TWIN_TEST_DOMAIN_ID``
(default 93), with discovery kept to this machine. It is not the notebook's
default domain (87) and not the default domain the real arm uses, so a test
never reaches either. rclpy reads the domain once per process, so every
tier-2 module in a pytest session shares it.
"""

from __future__ import annotations

import copy
import csv
import os
import select
import shutil
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

TEST_DOMAIN_ID = int(os.environ.get("TWIN_TEST_DOMAIN_ID", "93"))

# An extended arm: from here the wrist can bend to the grasp limit without
# folding onto the forearm, as it would from the rest pose.
EXTENDED = {"shoulder_lift": -20.0, "elbow_flex": 20.0, "wrist_flex": 0.0}
BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def join_test_domain() -> None:
    """Put this process, and every twin or kernel it starts, on the test domain."""
    from common.twin_ros import use_private_domain

    os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    use_private_domain(TEST_DOMAIN_ID)


def process_gone(pid: int) -> bool:
    """``pid`` has exited (a zombie counts)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def wait_exit(pid: int) -> bool:
    """True once ``pid`` has exited: the kernel's exit event on a pidfd."""
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        readable, _, _ = select.select([fd], [], [])
    finally:
        os.close(fd)
    return bool(readable) or process_gone(pid)


def assert_domain_free(message: str, wait: bool = False) -> None:
    """No twin holds the test domain's lock; with ``wait``, a holder's exit is awaited first."""
    from so101_sim.domain_lock import owner

    pid = owner(TEST_DOMAIN_ID)
    if pid and wait:
        wait_exit(pid)
        pid = owner(TEST_DOMAIN_ID)
    assert pid is None, f"{message} (pid {pid})"


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


def twin_log_report(path, dest: Path, lines: int = 60) -> str:
    """Copy the twin node's log into ``dest``; return its last ``lines`` lines."""
    path = Path(path)
    if not path.exists():
        return f"twin node log: {path} is gone"
    shutil.copyfile(path, dest / "twin_node.log")
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    return f"twin node log -> twin_node.log, last {len(tail)} lines:\n" + "\n".join(tail)


class Probe:
    """Reads the test domain's graph, the arm server's own joint state (the
    hardware side's ``/so101/hardware/joint_states``) and the setpoints it
    receives (``/so101/hardware/joint_commands``), in calibrated units; sends
    nothing.

    Every joint state is kept for five minutes, with the time it arrived and
    the time the twin stamped it, so a failed test can save the timeline.
    """

    def __init__(self, cfg: dict):
        from sensor_msgs.msg import JointState

        from common.motion import DEFAULT_CONTROL_HZ
        from common.ros_helpers import Spinner, parse_joint_state
        from common.twin_ros import _sensor_qos
        from so101_sim.config import load_sim_config
        from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES
        from so101_sim.mapping import JointMap

        sim = load_sim_config(cfg)
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        fps = float(cfg["robot"].get("control_fps", DEFAULT_CONTROL_HZ))
        # The server's velocity is a position difference per tick: one encoder
        # step of jitter reads as one step per tick; below one and a half of
        # those a joint is still (as ControllerArm judges the jaw).
        self._still = {m: 1.5 * self._jmap.steps_to_rad(m, 1.0) * fps for m in sim["joint_map"]}
        self._velocity: dict = {}
        self._commands: list = []  # (arrival, calibrated setpoint)
        self._parse = parse_joint_state
        self._spin = Spinner("twin_test_probe")
        self.node = self._spin.node
        self._cond = threading.Condition()
        self._joints: dict | None = None
        self._joints_t = 0.0
        self._history: deque = deque(maxlen=60 * 300)
        self._listeners: list = []
        self.node.create_subscription(JointState, HARDWARE_STATES, self._on_state, _sensor_qos())
        self.node.create_subscription(JointState, HARDWARE_COMMANDS, self._on_command, _sensor_qos())

    def _on_command(self, msg) -> None:
        radians, _effort = self._parse(msg)
        with self._cond:
            self._commands.append((time.time(), {n: self._jmap.to_cal(n, v) for n, v in radians.items()}))
            self._cond.notify_all()

    def _on_state(self, msg) -> None:
        radians, _effort = self._parse(msg)
        joints = {n: self._jmap.to_cal(n, v) for n, v in radians.items()}
        velocity = dict(zip(msg.name, msg.velocity))
        now = time.time()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self._cond:
            self._joints, self._joints_t, self._velocity = joints, now, velocity
            self._history.append((now, stamp, joints))
            self._cond.notify_all()
            for wake in self._listeners:  # under the lock: none runs after its unsubscribe returns
                wake()

    def subscribe(self, callback):
        """Call ``callback()`` after every joint state from now on; returns the unsubscribe."""
        with self._cond:
            self._listeners.append(callback)

        def unsubscribe() -> None:
            with self._cond:
                self._listeners.remove(callback)

        return unsubscribe

    def failure_report(self, dest: Path, since: float | None) -> str:
        """``joints.csv`` of every joint state since ``since``, its longest gaps, and its last 4 s."""
        from common.motion import joints_text
        from so101_sim import MOTOR_NAMES

        with self._cond:
            rows = [h for h in self._history if since is None or h[0] >= since]
        if not rows:
            return "joint states: none arrived during the test"
        with open(dest / "joints.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["received", "stamp", *MOTOR_NAMES])
            for recv, stamp, j in rows:
                out.writerow([
                    f"{recv:.3f}", f"{stamp:.3f}", *(f"{j[m]:.2f}" if m in j else "" for m in MOTOR_NAMES),
                ])
        recv = np.array([r[0] for r in rows])
        sent = np.array([r[1] for r in rows])
        lines = [f"joint states: {len(rows)} over {recv[-1] - recv[0]:.1f} s -> joints.csv"]
        if len(rows) > 1:
            i, k = int(np.argmax(np.diff(recv))), int(np.argmax(np.diff(sent)))
            lines.append(
                f"longest gap: {recv[i + 1] - recv[i]:.2f} s between arrivals after {_clock(recv[i])}, "
                f"{sent[k + 1] - sent[k]:.2f} s between twin stamps after {_clock(sent[k])}"
            )
        lines.append("last 4 s, every 0.25 s:")
        shown = None
        for recv_t, _stamp, j in rows:
            if recv_t >= recv[-1] - 4.0 and (shown is None or recv_t - shown >= 0.25):
                lines.append(f"  {_clock(recv_t)} {joints_text(j)}")
                shown = recv_t
        return "\n".join(lines)

    def joints(self, after: float | None = None) -> dict:
        """The newest joint state, or the first received after ``after`` (the twin publishes every control tick)."""
        with self._cond:
            self._cond.wait_for(lambda: self._joints is not None and (after is None or self._joints_t > after))
            return dict(self._joints)

    def wait_command(self, pred, after: float, budget_s: float, what: str) -> list:
        """The setpoints the server received after ``after``, once ``pred(them)``
        holds; woken by each one. Fails when ``budget_s`` passes first."""
        with self._cond:
            ok = self._cond.wait_for(lambda: pred([c for t, c in self._commands if t > after]), budget_s)
            got = [c for t, c in self._commands if t > after]
        assert ok, f"no {what} within {budget_s:.2f} s ({len(got)} setpoints arrived)"
        return got

    def commands(self, after: float) -> list:
        """Every setpoint the server received after ``after``."""
        with self._cond:
            return [c for t, c in self._commands if t > after]

    def wait_still(self, after: float, budget_s: float) -> dict:
        """The first joint state after ``after`` in which the server reports every
        joint still (velocity), woken by each state. Fails when ``budget_s`` passes first."""

        def still() -> bool:
            v = self._velocity
            return self._joints_t > after and bool(v) and all(abs(v[m]) < self._still[m] for m in self._still if m in v)

        with self._cond:
            ok = self._cond.wait_for(still, budget_s)
            joints = dict(self._joints or {})
        assert ok, f"the server did not report the arm still within {budget_s:.2f} s; last {joints}"
        return joints

    def first_message(self, topic: str, msg_type) -> None:
        """Block until a message arrives on ``topic``."""
        from common.twin_ros import _sensor_qos

        got = threading.Event()
        sub = self.node.create_subscription(msg_type, topic, lambda _m: got.set(), _sensor_qos())
        try:
            got.wait()
        finally:
            self.node.destroy_subscription(sub)

    def topics(self) -> set[str]:
        return {name for name, _types in self.node.get_topic_names_and_types()}

    def services(self) -> set[str]:
        return {name for name, _types in self.node.get_service_names_and_types()}

    def close(self) -> None:
        self._spin.close()


def _offline_world(cfg: dict):
    from common.motion import GRIPPER_CLOSED_FLOOR, REST_POSE
    from so101_sim.world import SimWorld

    sim = dict(cfg.get("sim") or {})
    sim["gripper_map"] = {**(sim.get("gripper_map") or {}), "closed_cal": GRIPPER_CLOSED_FLOOR}
    return SimWorld({"sim": sim}, rest_pose=REST_POSE, start=False)


def default_scene_snapshot(cfg: dict) -> dict:
    """The tests' ``default_scene()``: every prop at its XML spot, turned straight."""
    world = _offline_world(cfg)
    try:
        for obj in world.objects:
            world.place(obj, world.model.body(obj).pos)
        return world.snapshot_scene()
    finally:
        world.close()


def xml_poses(cfg: dict) -> dict[str, np.ndarray]:
    """Each prop's compiled ``qpos0`` (what reset restores)."""
    world = _offline_world(cfg)
    try:
        return {
            name: world.model.qpos0[qadr : qadr + 7].copy()
            for name, (qadr, _dadr) in world._free.items()
        }
    finally:
        world.close()


def quat_close(a, b, tol: float = 1e-3) -> bool:
    return abs(abs(float(np.dot(np.asarray(a, float), np.asarray(b, float)))) - 1.0) < tol


def keyword_parse(text: str):
    """The notebook's keyword stand-in for Llama."""
    from common.fixtures import offline_parse_text
    from common.intents import Intent, ParsedCommand

    d = offline_parse_text(text)
    params = {"object": d["object"]} if d.get("object") else {}
    return ParsedCommand(Intent.from_string(d["command"]), params=params, transcript=text)


class TwinRig:
    """The section 5 console's context, on a :class:`~common.twin_ros.TwinSession`.

    The arm camera and the mount camera are wrapped in ``FanoutCamera``
    as the console wraps them, and the operator is the notebook's scripted
    stand-in (``ScriptedPose`` / ``ScriptedHands`` / ``SyntheticCamera``).
    """

    def __init__(self, cfg: dict, detector, scene: dict | None = None):
        from common.fixtures import ScriptedHands, ScriptedPose, SyntheticCamera
        from common.motion import mapper_from_config
        from common.twin_console import FanoutCamera
        from common.twin_ros import TwinSession
        from integration.dispatch import WorkshopContext, build_registry

        self.cfg = copy.deepcopy(cfg)
        self.cfg["robot"]["backend"] = "sim"
        self.twin = TwinSession(self.cfg, scene=scene)
        self.said: list[str] = []
        self.armcam = FanoutCamera(self.twin.armcam)
        self.mount = FanoutCamera(SyntheticCamera())
        self.ctx = WorkshopContext(
            cfg=self.cfg,
            arm=self.twin.arm,
            pose=ScriptedPose(),
            hands=ScriptedHands(),
            camera=self.mount,
            detector=detector,
            arm_camera=self.armcam,
            mapper=mapper_from_config(self.cfg),
            registry=build_registry(),
            headless=True,
        )
        self.ctx.say = self.said.append
        self._crashes: list[str] = []

    def said_any(self, phrase: str) -> bool:
        return any(phrase.lower() in m.lower() for m in self.said)

    def failure_report(self, dest: Path, since: float | None) -> str:
        """What the robot said, and the twin node's log."""
        said = "\n".join(f"  {m}" for m in self.said[-15:]) or "  (nothing)"
        return f"robot said:\n{said}\n\n" + twin_log_report(self.twin.log_path, dest)

    def start(self, text: str) -> threading.Thread:
        """Run ``text`` through ``handle_transcript`` on a thread, as the console's dispatcher does.

        A crash is recorded; :meth:`run` and :meth:`close` raise it, so it fails the test.
        """
        from integration.dispatch import handle_transcript

        def behave() -> None:
            try:
                handle_transcript(self.ctx, text, keyword_parse)
            except Exception as e:
                self._crashes.append(f"{text!r}: {e!r}")
                raise

        worker = threading.Thread(target=behave, daemon=True, name=text)
        worker.start()
        return worker

    def run(self, text: str) -> None:
        """Run ``text`` to the end of its behavior; each move inside it is bounded by its own servo profile."""
        from rigs import raise_crashes

        self.start(text).join()
        raise_crashes(self._crashes)

    def stop(self) -> None:
        """The console's stop button: ``handle_transcript`` with "stop", from another thread."""
        from integration.dispatch import handle_transcript

        handle_transcript(self.ctx, "stop", keyword_parse)

    def close(self) -> None:
        from rigs import raise_crashes

        self.ctx.stop_event.set()
        for cam in (self.armcam, self.mount):
            try:
                cam.close()
            except Exception:
                pass
        self.twin.close()
        raise_crashes(self._crashes)
