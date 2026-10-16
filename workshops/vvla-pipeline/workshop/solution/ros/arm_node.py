# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The SO-101 server node - the ONE process that owns the robot bus.

Exactly one process may talk to the Feetech serial bus; this node is it.
Everything else (behaviors, the UI, your terminal) publishes joint targets
on ``/so101/joint_command`` and reads ``/so101/joint_state`` back. That
topic contract is what makes the system restartable and inspectable:
``ros2 topic echo /so101/joint_state`` works no matter what owns the bus.

Message convention (``sensor_msgs/JointState``, degrees):
- ``name``/``position`` - absolute joint targets; PARTIAL dicts are legal
  (command just the gripper and the rest holds).
- ``effort[i]`` on the state topic - the gripper servo's current for
  ``name[i] == "gripper"``; **NaN when the bus can't read it**. NaN is the
  honest "unavailable", and clients must treat it as None, not as a number.

The control loop thread (provided) is the same observe→act pattern the real
pipeline uses: one thread on the bus, no contention. Your TODOs are the ROS
wiring around it.

Notebook: ``notebooks/02_ros2/02_robot.ipynb``
Run it for real:
    python -m ros.arm_node --dry-run     # no hardware, integrates commands
    python -m ros.arm_node               # real SO-101 (sole bus owner!)
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time

from common.config import load_config
from common.motion import (
    LIVE_MIMIC_LIMITS,
    MOTOR_NAMES,
    REST_POSE,
    DryRunArm,
    clamp_joints,
    make_arm_backend,
    set_active_limits,
)
from common.ros_helpers import GRIPPER_TORQUE_TOPIC, joint_state_msg

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Int32

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]

# Read the gripper effort register every Nth control tick (an extra serial
# transaction - cheap, but no reason to pay it at full rate).
EFFORT_SAMPLE_EVERY_N_TICKS = 3


class So101ServerNode(Node):
    """Bridges an ArmClient backend (real or dry-run) to ROS 2 topics."""

    def __init__(self, cfg: dict, arm=None):
        super().__init__("so101_server_node")
        r = cfg["robot"]
        self.command_topic = r["command_topic"]
        self.state_topic = r["state_topic"]
        self._control_fps = float(r.get("control_fps", 30.0))
        self._arm = arm if arm is not None else DryRunArm()
        # Real hardware behind this node: admit the extended live gesture-
        # mimic envelope end to end. Clients clamp every scripted behavior to
        # the base JOINT_LIMITS before publishing, so only live mimic (which
        # switches its own clamp) ever uses the extra travel. A dry-run node
        # keeps the base limits.
        if getattr(self._arm, "is_live", False):
            set_active_limits(LIVE_MIMIC_LIMITS)
        self._lock = threading.Lock()
        self._target = dict(REST_POSE)
        self._last_cmd_time = 0.0
        # Direct-goal mode (config: max_relative_target: null - the pipeline's
        # setting): each goal is one absolute write the servo profiles on its
        # own, so the loop must write a goal only when it CHANGES. Re-writing
        # the same goal every tick restarts the onboard profile (twitchy,
        # stepped motion).
        self._direct_goal = r.get("max_relative_target", 8.0) is None
        self._last_sent_target = None
        self._gripper_load = math.nan  # nan = feedback unavailable
        self._tick_count = 0
        self._state_pub = None
        self._torque_limit = None  # next gripper Torque_Limit to write
        self._setup_ros()
        self.create_subscription(Int32, GRIPPER_TORQUE_TOPIC, self._on_torque_limit, 10)

        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Ready - cmd: {self.command_topic}  state: {self.state_topic}  "
            f"{self._control_fps:.0f} Hz"
        )

    def _setup_ros(self) -> None:
        """Wire the command subscription and the state publisher.

        Both ends use sensor-stream QoS (best-effort / keep-last / depth 1):
        the newest command and the newest state always win - a robot must
        never act on a backlog of stale targets.
        """
        # >>> TODO 2.4: So101ServerNode._setup_ros - notebooks/02_ros2/02_robot.ipynb
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(JointState, self.command_topic, self._on_command, qos)
        self._state_pub = self.create_publisher(JointState, self.state_topic, qos)
        # <<< TODO 2.4

    def _on_command(self, msg) -> None:
        """Fold one JointState command into the target pose.

        Zip ``msg.name``/``msg.position``, keep only known motors
        (:data:`MOTOR_NAMES` - partial commands are legal), clamp through
        :func:`clamp_joints` (the safety backstop), and update the shared
        target + freshness timestamp under the lock.
        """
        # >>> TODO 2.5: So101ServerNode._on_command - notebooks/02_ros2/02_robot.ipynb
        joints = {
            n: float(p) for n, p in zip(msg.name, msg.position) if n in MOTOR_NAMES
        }
        if not joints:
            return
        with self._lock:
            self._target.update(clamp_joints(joints))
            self._last_cmd_time = time.time()
        # <<< TODO 2.5

    def _on_torque_limit(self, msg) -> None:
        """PROVIDED: queue a gripper Torque_Limit for the control loop."""
        with self._lock:
            self._torque_limit = int(msg.data)

    def _publish_state(self, joints: dict) -> None:
        """Publish the arm's state, with gripper effort riding in ``effort``.

        Build the message with :func:`common.ros_helpers.joint_state_msg`
        over all :data:`MOTOR_NAMES` (pass ``node=self`` for the timestamp).
        ``effort`` must align with ``name``: the sampled gripper load at the
        gripper's index, NaN everywhere else - NaN, not 0, is "no reading".
        """
        # >>> TODO 2.6: So101ServerNode._publish_state - notebooks/02_ros2/02_robot.ipynb
        msg = joint_state_msg(
            MOTOR_NAMES,
            [joints.get(m, 0.0) for m in MOTOR_NAMES],
            effort=[
                self._gripper_load if m == "gripper" else math.nan for m in MOTOR_NAMES
            ],
            node=self,
        )
        self._state_pub.publish(msg)
        # <<< TODO 2.6

    def _control_loop(self) -> None:
        """PROVIDED: sequential observe→act loop (one thread on the bus)."""
        interval = 1.0 / self._control_fps
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                target = dict(self._target)
                fresh = (time.time() - self._last_cmd_time) < 1.0
                torque_limit, self._torque_limit = self._torque_limit, None
            try:
                if torque_limit is not None:
                    self._arm.set_gripper_torque_limit(torque_limit)
                # Hold the last pose silently when commands stop. In
                # direct-goal mode, write only when the goal changes so the
                # servo's onboard profile runs once, uninterrupted.
                if fresh and (
                    not self._direct_goal or target != self._last_sent_target
                ):
                    self._arm.send_joints(target)
                    self._last_sent_target = target
                joints = self._arm.get_joints()

                self._tick_count += 1
                if self._tick_count % EFFORT_SAMPLE_EVERY_N_TICKS == 0:
                    load = self._arm.get_gripper_load()
                    self._gripper_load = math.nan if load is None else float(load)

                self._publish_state(joints)
            except NotImplementedError:
                raise
            except Exception as e:  # a transient bus error must not kill the loop
                logger.warning("bus tick failed, continuing: %s", e)
            time.sleep(max(interval - (time.perf_counter() - t0), 0.0))

    def destroy_node(self) -> None:
        """Stop the control thread, park/close the arm, then destroy the ROS node."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._arm.close()  # LeRobotArm parks at REST_POSE on close
        except Exception as e:
            logger.error("Arm shutdown error: %s", e)
        super().destroy_node()


def main() -> None:
    """CLI: bring up the SO-101 server node and spin until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not ROS2_AVAILABLE:
        raise SystemExit(
            "rclpy not importable - source ROS 2 before running "
            "(offline checks: python -m ros.selftest)"
        )
    parser = argparse.ArgumentParser(description="SO-101 server node")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no hardware - integrate commands in software",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rclpy.init()
    node = So101ServerNode(cfg, arm=make_arm_backend(cfg, dry_run=args.dry_run))
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down - parking the arm.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
