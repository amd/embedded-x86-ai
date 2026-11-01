# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The ros2_control robot description: limits from each stack's JOINT_LIMITS,
and the twin's geometry, so the same radians pose both models alike."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytestmark = pytest.mark.sim
pytest.importorskip("xacro")  # ROS 2 Jazzy

BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STACKS = {
    "pipeline": ("pipeline_cfg", "vla_pipeline.robot.arm_interface"),
    "workshop": ("workshop_cfg", "common.motion"),
}


def _description(request, stack: str):
    import importlib

    from so101_sim.description import robot_description

    fixture, name = STACKS[stack]
    cfg = request.getfixturevalue(fixture)
    module = importlib.import_module(name)
    urdf = robot_description(
        cfg, module.JOINT_LIMITS, joint_commands_topic="/cmd", joint_states_topic="/state",
        initial_state_timeout=7.0,
    )
    return cfg, module, ET.fromstring(urdf)


def _origin(element) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in element.get("rpy").split()]).as_matrix()
    m[:3, 3] = [float(v) for v in element.get("xyz").split()]
    return m


def _joint_frames(root, q: dict) -> dict:
    """Each joint's frame in ``base_link`` at the joint angles ``q`` (rad)."""
    from scipy.spatial.transform import Rotation

    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {"base_link": np.eye(4)}
    frames = {}

    def link(name):
        if name not in links:
            j = joints[name]
            frame = link(j.find("parent").get("link")) @ _origin(j.find("origin"))
            frames[j.get("name")] = frame
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec([0, 0, q.get(j.get("name"), 0.0)]).as_matrix()
            links[name] = frame @ turn
        return links[name]

    for child in joints:
        link(child)
    return frames


@pytest.mark.parametrize("stack", sorted(STACKS))
def test_limits_and_hardware_side_come_through_the_twin_joint_map(request, stack):
    """The limits are the stack's JOINT_LIMITS. The hardware side waits for the
    first joint state and stays quiet while every joint is within one encoder step."""
    from so101_sim.config import load_sim_config
    from so101_sim.mapping import JointMap

    cfg, module, root = _description(request, stack)
    sim = load_sim_config(cfg)
    jmap = JointMap(sim["joint_map"], sim["gripper_map"])
    for name, (lower, upper) in module.JOINT_LIMITS.items():
        limit = root.find(f"joint[@name='{name}']/limit")
        ends = sorted((jmap.to_rad(name, lower), jmap.to_rad(name, upper)))
        assert [float(limit.get("lower")), float(limit.get("upper"))] == pytest.approx(ends), name
        assert float(limit.get("effort")) == sim["current_model"]["stall_torque_nm"]
    from so101_sim.control import JOINT_STATE_STALE_S

    hardware = {p.get("name"): p.text for p in root.findall("ros2_control/hardware/param")}
    threshold = float(hardware.pop("trigger_joint_command_threshold"))
    assert hardware == {
        "joint_commands_topic": "/cmd", "joint_states_topic": "/state",
        "initial_state_timeout": "7.0",
        "joint_state_timeout": repr(float(JOINT_STATE_STALE_S)),
    }
    assert threshold == pytest.approx(sum(jmap.steps_to_rad(name, 1.0) for name in module.JOINT_LIMITS))


@pytest.mark.parametrize("q", [
    {},
    {"shoulder_pan": 0.3, "shoulder_lift": -0.4, "elbow_flex": 0.5, "wrist_flex": -0.2, "wrist_roll": 0.6,
     "gripper": 0.8},
], ids=["zero", "bent"])
def test_every_joint_turns_about_the_twin_axis(request, q):
    import mujoco

    from so101_sim.world import SimWorld

    _cfg, _module, root = _description(request, "workshop")
    frames = _joint_frames(root, q)
    world = SimWorld(start=False)
    try:
        m, d = world.model, world.data
        d.qpos[:] = 0.0
        for name, angle in q.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = angle
        mujoco.mj_forward(m, d)
        base = m.body("base").id
        rot, pos = d.xmat[base].reshape(3, 3), d.xpos[base]
        for name in (*BODY, "gripper"):
            jid = m.joint(name).id
            anchor = rot.T @ (d.xanchor[jid] - pos)
            axis = rot.T @ d.xaxis[jid]
            frame = frames[name]
            assert np.linalg.norm(frame[:3, 3] - anchor) < 1e-4, f"{name} anchor off by {frame[:3, 3] - anchor} m"
            angle = np.degrees(np.arccos(np.clip(frame[:3, 2] @ axis, -1.0, 1.0)))
            assert angle < 0.2, f"{name} axis off by {angle:.2f} deg"
    finally:
        world.close()


def test_controllers_drive_the_description_joints(request):
    import yaml

    from so101_sim.description import CONTROLLERS_FILE

    _cfg, _module, root = _description(request, "workshop")
    hardware = {j.get("name") for j in root.findall("ros2_control/joint")}
    controllers = yaml.safe_load(CONTROLLERS_FILE.read_text())
    for name in ("arm_controller", "arm_forward_controller"):
        assert tuple(controllers[name]["ros__parameters"]["joints"]) == BODY, name
    assert controllers["gripper_forward_controller"]["ros__parameters"]["joints"] == ["gripper"]
    assert controllers["gripper_action_controller"]["ros__parameters"]["joint"] == "gripper"
    assert hardware == {*BODY, "gripper"}
    assert "update_rate" not in controllers["controller_manager"]["ros__parameters"], (
        "the launch sets update_rate from robot.control_fps"
    )
