# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The arm client - how behaviors drive the robot over ROS 2.

:class:`Ros2ArmClient` implements the same :class:`~common.motion.ArmClient`
API behaviors already use with :class:`~common.motion.DryRunArm`, but backed
by ros2_control's controllers on top of the server node. That symmetry is the
point: behaviors are transport-agnostic, so the composition you write in
component 3 runs unchanged against a dry-run arm, the ROS transport, or
direct serial.

The provided base, :class:`~common.ros2_control_arm.ControllerArm`, waits
until ``/joint_states`` and the controllers are live, runs a move as one
``FollowJointTrajectory`` goal to ``arm_controller`` and waits for its
result, switches the forward controller in for streamed setpoints, and sets
the gripper's torque limit parameter.

Your TODOs: the endpoints (the trajectory action client, the forward
controllers' publishers, the ``/joint_states`` subscription), publishing
clamped setpoints in radians, and parsing the state back out (radians to
degrees, and the gripper-effort-or-NaN convention).

Notebook: ``notebooks/02_ros2/02_robot.ipynb``
Round-trip test (the server and its controllers running in another terminal,
``python -m ros.node_supervisor --only arm --dry-run``):
    python -m ros.arm_client
"""

from __future__ import annotations

import argparse
import logging

from so101_sim.control import ARM_CONTROLLER, BRING_UP_S, GRIPPER_CONTROLLER, STREAM_CONTROLLER

from common.motion import GRIPPER_CLOSED_FLOOR, MOTOR_NAMES, clamp_joints
from common.ros2_control_arm import BODY, ControllerArm
from common.ros_helpers import joint_velocity, parse_joint_state

logger = logging.getLogger(__name__)


class Ros2ArmClient(ControllerArm):
    """ArmClient over the controllers; spins a private executor."""

    def __init__(self, cfg: dict, state_timeout_s: float = BRING_UP_S):
        # PROVIDED: blocks until /joint_states and the controllers are live -
        # commanding an arm whose state you can't see is how fingers get pinched.
        super().__init__(cfg, node_name="so101_arm_client", state_timeout_s=state_timeout_s)

    def _setup_ros(self, node) -> None:
        """Create the client's endpoints on ``node`` (its executor is provided).

        - ``self._trajectory``: an ``rclpy.action.ActionClient`` for
          ``control_msgs/action/FollowJointTrajectory`` on
          ``/{ARM_CONTROLLER}/follow_joint_trajectory``;
        - ``self._stream_pub`` and ``self._gripper_pub``: publishers of
          ``std_msgs/Float64MultiArray`` on ``/{STREAM_CONTROLLER}/commands``
          and ``/{GRIPPER_CONTROLLER}/commands``, depth 10;
        - a subscription to ``sensor_msgs/JointState`` on ``/joint_states``
          → ``self._on_state``, depth 10 (reliable, as joint_state_broadcaster
          publishes it).
        """
        # >>> TODO 2.7: Ros2ArmClient._setup_ros - notebooks/02_ros2/02_robot.ipynb
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self._trajectory = ActionClient(
            node, FollowJointTrajectory, f"/{ARM_CONTROLLER}/follow_joint_trajectory"
        )
        self._stream_pub = node.create_publisher(Float64MultiArray, f"/{STREAM_CONTROLLER}/commands", 10)
        self._gripper_pub = node.create_publisher(Float64MultiArray, f"/{GRIPPER_CONTROLLER}/commands", 10)
        node.create_subscription(JointState, "/joint_states", self._on_state, 10)
        # <<< TODO 2.7

    def _on_state(self, msg) -> None:
        """Hand one ``/joint_states`` message to ``self._update_state``.

        :func:`common.ros_helpers.parse_joint_state` gives the positions
        (radians) and the gripper effort (N·m) with NaN already mapped to
        ``None``; :func:`common.ros_helpers.joint_velocity` the gripper's
        speed. Convert each known motor (:data:`MOTOR_NAMES`) with
        ``self._jmap.to_cal(name, radians)``, then call
        ``self._update_state(positions, gripper_speed, effort)``.
        """
        # >>> TODO 2.9: Ros2ArmClient._on_state - notebooks/02_ros2/02_robot.ipynb
        positions, effort = parse_joint_state(msg)
        self._update_state(
            {n: self._jmap.to_cal(n, p) for n, p in positions.items() if n in MOTOR_NAMES},
            joint_velocity(msg, "gripper"),
            effort,
        )
        # <<< TODO 2.9

    def send_joints(self, joints: dict) -> None:
        """Stream absolute joint targets (degrees; partial dicts allowed).

        Clamp through :func:`common.motion.clamp_joints` FIRST, with the
        limits of the move in progress (``self._send_limits``) - the client
        clamps too, not just the server; safety layers as belt and braces.
        Then send the gripper with ``self._command_gripper(degrees)``, and
        the body to the forward controller: ``self._stream_setpoint(joints)``
        returns the whole body setpoint (None when ``joints`` names no body
        joint); publish it as a ``Float64MultiArray`` of radians in
        :data:`BODY` order.
        """
        # >>> TODO 2.8: Ros2ArmClient.send_joints - notebooks/02_ros2/02_robot.ipynb
        from std_msgs.msg import Float64MultiArray

        joints = clamp_joints(joints, self._send_limits)
        if "gripper" in joints:
            self._command_gripper(joints["gripper"])
        body = self._stream_setpoint(joints)
        if body is not None:
            self._stream_pub.publish(Float64MultiArray(data=[self._jmap.to_rad(k, body[k]) for k in BODY]))
        # <<< TODO 2.8


def main() -> None:
    """CLI: connect to the running controllers and round-trip a gripper sweep."""
    from common.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 ROS 2 client - round-trip test against the running server and controllers"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    arm = Ros2ArmClient(load_config(args.config))
    try:
        print("State:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Commanding a gripper sweep over ROS 2...")
        arm.go_to({"gripper": 60.0})
        arm.go_to({"gripper": GRIPPER_CLOSED_FLOOR})
        arm.go_to_rest()
        print("ROS 2 round-trip PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
