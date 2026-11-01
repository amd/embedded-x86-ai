# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline's arm over ros2_control, on the twin.

The twin and its arm server run in this process, so the checks read the
physics; ``launch/so101_control.launch.py`` runs the controllers; the
pipeline's ``Ros2ArmClient`` drives them as the orchestrator does.

Tier 2 (ROS 2): ``./run_sim_tests.sh``.
"""

from __future__ import annotations

import copy
import tempfile

import pytest

from twin_ros_rig import TEST_DOMAIN_ID, assert_domain_free, join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(240)]
pytest.importorskip("rclpy")

# Jaws low over the table, closing axis horizontal: how the pickup grasps the ball.
GRASP_POSE = {"shoulder_pan": 0.0, "shoulder_lift": 9.0, "elbow_flex": 20.0, "wrist_flex": 54.0, "wrist_roll": 0.0}


def _until(world, pred) -> None:
    """Block until ``pred()`` holds, judged after each physics pass."""
    seen = world.next_step(world.passes)
    while not pred():
        seen = world.next_step(seen)


@pytest.fixture(scope="module")
def twin(pipeline_cfg):
    from conftest import PIPE_ROOT
    from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, start_controllers, stop_controllers
    from so101_sim.domain_lock import acquire
    from so101_sim.node import build_nodes
    from vla_pipeline.robot.robot_node import Ros2ArmClient
    from vla_pipeline.robot.ros2_control_arm import Spinner

    join_test_domain()
    assert_domain_free("a twin already runs on the test domain")
    lock = acquire(TEST_DOMAIN_ID)
    cfg = copy.deepcopy(pipeline_cfg)
    cfg["robot"].update(backend="sim", command_topic=HARDWARE_COMMANDS, state_topic=HARDWARE_STATES)
    world, nodes, closers = build_nodes(cfg, stack="pipeline", view=False)
    host = Spinner("pipeline_twin_host")
    for node in nodes:
        host.add_node(node)
    log = tempfile.NamedTemporaryFile(prefix="so101_control_", suffix=".log", delete=False)
    launch = start_controllers(PIPE_ROOT / "config" / "pipeline.yaml", "pipeline", log=log)
    arm = None
    try:
        arm = Ros2ArmClient(cfg)
        yield world, arm
    finally:
        if arm is not None:
            arm.close()
        stop_controllers(launch)
        host.close()
        for close in reversed(closers):
            close()
        lock.close()


@pytest.fixture
def arm(twin):
    """The client, parked before each check."""
    from vla_pipeline.robot.arm_interface import REST_POSE

    world, client = twin
    assert client.move_to(dict(REST_POSE)) == "arrived"
    return world, client


def test_a_move_arrives_where_the_trajectory_ends(arm):
    world, client = arm
    goal = {"shoulder_pan": 20.0, "shoulder_lift": -60.0, "elbow_flex": 70.0, "wrist_flex": 40.0}
    assert client.move_to(goal) == "arrived"
    here = world.joint_cal()
    off = {k: round(here[k] - v, 2) for k, v in goal.items() if abs(here[k] - v) > 2.5}
    assert not off, f"the twin is off the goal (deg): {off}"


def test_a_stop_cancels_the_move_and_the_arm_holds(arm):
    from rigs import wait_joint_rest

    world, client = arm
    start = client.get_joints()["shoulder_pan"]
    how = client.move_to({"shoulder_pan": 60.0}, stop_check=lambda: client.get_joints()["shoulder_pan"] > start + 15.0)
    assert how == "stopped"
    wait_joint_rest(world, "shoulder_pan")
    held = world.joint_cal()["shoulder_pan"]
    assert held < 45.0, f"the arm carried on to {held:.1f} deg after the stop"


def test_a_goal_the_joints_cannot_meet_is_blocked(arm):
    """Tighter than one encoder step: the trajectory controller ends the goal as a tolerance violation."""
    _world, client = arm
    assert client.move_to({"shoulder_pan": 10.0}, tolerance_deg=1e-4) == "blocked"


def test_streamed_setpoints_drive_the_forward_controller(arm):
    from vla_pipeline.robot.arm_interface import REST_POSE

    world, client = arm
    for pan in (5.0, 10.0, 15.0, 20.0):
        client.send_joints({"shoulder_pan": pan})
    _until(world, lambda: abs(world.joint_cal()["shoulder_pan"] - 20.0) < 1.0)
    assert client.move_to(dict(REST_POSE)) == "arrived", "the trajectory controller did not take over again"


def test_a_scene_reset_leaves_the_controller_holding_the_parked_pose(arm, pipeline_cfg):
    """The reset parks the twin; the trajectory controller must then hold the
    parked pose, or its last goal pulls the arm back off it."""
    import threading

    from control_msgs.msg import JointTrajectoryControllerState

    from so101_sim.config import load_sim_config
    from so101_sim.control import ARM_CONTROLLER
    from so101_sim.mapping import JointMap
    from vla_pipeline.robot.ros2_control_arm import REPLY_S, Spinner

    world, client = arm
    assert client.move_to({"shoulder_pan": 30.0, "elbow_flex": 60.0}) == "arrived"
    assert client.reset_sim()
    sim = load_sim_config(pipeline_cfg)
    jmap = JointMap(sim["joint_map"], sim["gripper_map"])
    heard, reference = threading.Event(), {}

    def on_state(msg) -> None:
        if not heard.is_set():
            reference.update((n, jmap.to_cal(n, p)) for n, p in zip(msg.joint_names, msg.reference.positions))
            heard.set()

    probe = Spinner("reset_probe")
    try:
        probe.node.create_subscription(
            JointTrajectoryControllerState, f"/{ARM_CONTROLLER}/controller_state", on_state, 10,
        )
        assert heard.wait(REPLY_S), f"no {ARM_CONTROLLER} state within {REPLY_S} s"
    finally:
        probe.close()
    parked = world.rest_pose
    off = {k: round(v - parked[k], 2) for k, v in reference.items() if abs(v - parked[k]) > 0.5}
    assert not off, f"{ARM_CONTROLLER} holds off the parked pose (deg): {off}"


def test_the_gripper_load_and_torque_limit_cross_the_wire(arm):
    """The torque limit is the server's parameter in N·m; effort leaves the
    server in N·m and reaches behaviors as the twin's current counts."""
    from rigs import wait_joint_rest
    from so101_sim.kinematics import OPEN_GRIPPER
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    world, client = arm
    assert client.move_to({**GRASP_POSE, "gripper": OPEN_GRIPPER}) == "arrived"
    world.stage_object("sports_ball", GRASP_POSE)
    client.set_gripper_torque_limit(240)  # under Overload_Torque: the squeeze never trips
    _until(world, lambda: world.bus.read("Torque_Limit", "gripper", normalize=False) == 240)
    client.send_joints({"gripper": GRIPPER_CLOSED_FLOOR})
    _until(world, lambda: world.is_grasped("sports_ball"))
    wait_joint_rest(world, "gripper")

    def twin_counts() -> float:
        return float(world.bus.read("Present_Current", "gripper", normalize=False))

    # The server samples the current every third tick: half a second of physics
    # brings the client about ten samples of the squeeze at rest.
    t0, seen = float(world.data.time), world.passes
    while abs((client.get_gripper_load() or 0.0) - twin_counts()) > 25.0:
        assert float(world.data.time) - t0 < 0.5, (
            f"the client reads {client.get_gripper_load()} counts, the twin {twin_counts():.0f}: "
            "the effort's round trip changed it"
        )
        seen = world.next_step(seen)
    client.send_joints({"gripper": OPEN_GRIPPER})
    world.hide("sports_ball")
