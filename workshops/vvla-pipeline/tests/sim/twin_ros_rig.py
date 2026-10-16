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

import os
import threading
import time

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


class Probe:
    """Reads the test domain's graph and the arm's joint state, sends nothing."""

    def __init__(self, state_topic: str):
        from sensor_msgs.msg import JointState

        from common.ros_helpers import parse_joint_state
        from common.twin_ros import _sensor_qos, _Spinner

        self._parse = parse_joint_state
        self._spin = _Spinner("twin_test_probe")
        self.node = self._spin.node
        self._cond = threading.Condition()
        self._joints: dict | None = None
        self._joints_t = 0.0
        self.node.create_subscription(JointState, state_topic, self._on_state, _sensor_qos())

    def _on_state(self, msg) -> None:
        joints, _effort = self._parse(msg)
        with self._cond:
            self._joints, self._joints_t = joints, time.time()
            self._cond.notify_all()

    def joints(self, after: float | None = None, timeout: float = 2.0) -> dict | None:
        """The newest joint state, or the first received after ``after``."""
        deadline = time.time() + timeout
        with self._cond:
            while self._joints is None or (after is not None and self._joints_t <= after):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return dict(self._joints)

    def publishers(self, topic: str) -> int:
        return int(self.node.count_publishers(topic))

    def wait_publishers(self, topic: str, count: int, timeout: float) -> bool:
        """Wait until ``topic`` has exactly ``count`` publishers."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.publishers(topic) == count:
                return True
            time.sleep(0.1)
        return self.publishers(topic) == count

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

    The arm camera and the operator camera are wrapped in ``FanoutCamera``
    as the console wraps them, and the operator is the notebook's scripted
    stand-in (``ScriptedPose`` / ``ScriptedHands`` / ``SyntheticCamera``).
    """

    def __init__(self, cfg: dict, detector, scene: dict | None = None):
        from common.fixtures import ScriptedHands, ScriptedPose, SyntheticCamera
        from common.motion import mapper_from_config
        from common.twin_console import FanoutCamera
        from common.twin_ros import TwinSession
        from integration.dispatch import WorkshopContext, build_registry

        self.cfg = cfg
        self.twin = TwinSession(cfg, scene=scene)
        self.said: list[str] = []
        self.armcam = FanoutCamera(self.twin.armcam)
        self.operator = FanoutCamera(SyntheticCamera())
        self.ctx = WorkshopContext(
            cfg=cfg,
            arm=self.twin.arm,
            pose=ScriptedPose(),
            hands=ScriptedHands(),
            camera=self.operator,
            detector=detector,
            arm_camera=self.armcam,
            mapper=mapper_from_config(cfg),
            registry=build_registry(),
            headless=True,
        )
        self.ctx.say = self.said.append

    def said_any(self, phrase: str) -> bool:
        return any(phrase.lower() in m.lower() for m in self.said)

    def start(self, text: str) -> threading.Thread:
        """Run ``text`` through ``handle_transcript`` on a thread, as the console's dispatcher does."""
        from integration.dispatch import handle_transcript

        worker = threading.Thread(
            target=handle_transcript, args=(self.ctx, text, keyword_parse), daemon=True, name=text
        )
        worker.start()
        return worker

    def run(self, text: str, timeout: float) -> None:
        worker = self.start(text)
        worker.join(timeout)
        if worker.is_alive():
            self.ctx.stop_event.set()
            worker.join(10.0)
            raise TimeoutError(f"{text!r} did not finish within {timeout:.0f} s; robot said {self.said}")

    def stop(self) -> None:
        """The console's stop button: ``handle_transcript`` with "stop", from another thread."""
        from integration.dispatch import handle_transcript

        handle_transcript(self.ctx, "stop", keyword_parse)

    def close(self) -> None:
        self.ctx.stop_event.set()
        for cam in (self.armcam, self.operator):
            try:
                cam.close()
            except Exception:
                pass
        self.twin.close()
