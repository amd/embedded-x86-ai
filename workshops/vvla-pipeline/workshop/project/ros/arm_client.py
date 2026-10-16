# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The arm client - how behaviors drive the robot over ROS 2.

:class:`Ros2ArmClient` implements the same :class:`~common.motion.ArmClient`
API behaviors already use with :class:`~common.motion.DryRunArm`, but backed
by the server node's topics. That symmetry is the point: behaviors are
transport-agnostic, so the composition you write in component 3 runs
unchanged against a dry-run arm, the ROS transport, or direct serial.

Your TODOs: the pub/sub wiring, publishing clamped commands, and parsing
state (positions + the gripper-effort-or-NaN convention) back out.

Notebook: ``notebooks/02_ros2/02_robot.ipynb``
Round-trip test (server node running in another terminal):
    python -m ros.arm_client
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from common.motion import GRIPPER_CLOSED_FLOOR, ArmClient, clamp_joints
from common.ros_helpers import GRIPPER_TORQUE_TOPIC, joint_state_msg, parse_joint_state

logger = logging.getLogger(__name__)


class Ros2ArmClient(ArmClient):
    """ArmClient over the server node's topics; spins a private executor."""

    # Commands go to the robot topic - assume real hardware behind it, so
    # live gesture mimic may use LIVE_MIMIC_LIMITS. (A dry-run server node
    # still clamps itself to the base limits, so nothing unsafe gets through.)
    is_live = True

    def __init__(self, cfg: dict, state_timeout_s: float = 5.0):
        r = cfg["robot"]
        # Mirror the server's mode: with the slew clamp off (the workshop
        # config), go_to() sends one goal and waits (move_to) instead of
        # streaming - re-writing goals every tick restarts the servo profile.
        self._direct_goal = r.get("max_relative_target", 8.0) is None
        self._state_lock = threading.Lock()
        self._state = None
        self._effort = None
        self._node = None
        self._pub = None
        self._torque_pub = None
        self._executor = None
        self._setup_ros(r)

        # PROVIDED: block until the state stream is live - commanding an arm
        # whose state you can't see is how fingers get pinched.
        deadline = time.time() + state_timeout_s
        while time.time() < deadline:
            with self._state_lock:
                if self._state is not None:
                    logger.info("Ros2ArmClient: state stream live.")
                    return
            time.sleep(0.05)
        raise TimeoutError(
            f"No JointState on {r['state_topic']} within {state_timeout_s}s - "
            "is the server node running? (python -m ros.arm_node --dry-run)"
        )

    def _setup_ros(self, r: dict) -> None:
        """Create the client node, command publisher, and state subscription.

        Init rclpy if needed; node ``so101_arm_client``; publisher of
        ``sensor_msgs/JointState`` on ``r['command_topic']`` and subscription
        on ``r['state_topic']`` → ``self._on_state`` - both with best-effort /
        keep-last / depth-1 QoS (match the server or you hear nothing);
        then spin a ``SingleThreadedExecutor`` on a daemon thread.
        """
        # >>> TODO 2.7: Ros2ArmClient._setup_ros - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.7 (Ros2ArmClient._setup_ros): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.7

    def _on_state(self, msg) -> None:
        """Cache the latest positions + gripper effort from a state message.

        Use :func:`common.ros_helpers.parse_joint_state` - it hands you the
        positions dict and the gripper effort with NaN already mapped to
        ``None``. Store both under the state lock.
        """
        # >>> TODO 2.9: Ros2ArmClient._on_state - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.9 (Ros2ArmClient._on_state): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.9

    def send_joints(self, joints: dict) -> None:
        """Publish absolute joint targets (partial dicts allowed).

        Clamp through :func:`common.motion.clamp_joints` FIRST - the client
        clamps too, not just the server; safety layers as belt and braces -
        then pack with :func:`common.ros_helpers.joint_state_msg` (pass
        ``node=self._node`` for the timestamp) and publish.
        """
        # >>> TODO 2.8: Ros2ArmClient.send_joints - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.8 (Ros2ArmClient.send_joints): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.8

    def get_joints(self) -> dict:
        """PROVIDED."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("No arm state received yet.")
            return dict(self._state)

    def get_gripper_load(self):
        """PROVIDED: latest gripper effort, or None (NaN on the wire)."""
        with self._state_lock:
            return self._effort

    def set_gripper_torque_limit(self, value: int) -> None:
        """PROVIDED: publish the gripper Torque_Limit (0-1000); the server
        node writes it on its next control tick."""
        from std_msgs.msg import Int32

        if self._torque_pub is None:
            self._torque_pub = self._node.create_publisher(Int32, GRIPPER_TORQUE_TOPIC, 10)
        self.gripper_torque_limit = int(min(max(int(value), 0), 1000))
        self._torque_pub.publish(Int32(data=self.gripper_torque_limit))

    def close(self) -> None:
        """Stop the executor and destroy the client node."""
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()


def main() -> None:
    """CLI: connect to a running server node and round-trip a gripper sweep."""
    from common.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 ROS 2 client - round-trip test against a running server"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    arm = Ros2ArmClient(load_config(args.config))
    try:
        print("State:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Commanding a gripper sweep over ROS 2...")
        arm.go_to({"gripper": 60.0}, duration_s=1.5)
        arm.go_to({"gripper": GRIPPER_CLOSED_FLOOR}, duration_s=1.5)
        arm.go_to_rest()
        print("ROS 2 round-trip PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
