# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The SO-101 server node - the ONE process that owns the robot bus.

Exactly one process may talk to the Feetech serial bus; this node is it. It
is the hardware side of ros2_control: ``topic_based_ros2_control`` (run by
``launch/so101_control.launch.py``) publishes the controllers' joint
setpoints on ``robot.command_topic`` and reads the arm back on
``robot.state_topic``; the launch remaps both onto
``/so101/hardware/joint_commands`` and ``/so101/hardware/joint_states``.
Behaviors talk to the controllers (``ros/arm_client.py``), never to this
node, and ``ros2 topic echo /joint_states`` shows the arm no matter what
owns the bus.

Message convention (``sensor_msgs/JointState``, radians):
- ``name``/``position`` - absolute joint setpoints; PARTIAL commands are
  legal (command just the gripper and the rest holds);
- ``velocity`` on the state topic - each joint's speed in rad/s, from the
  change in position since the last tick;
- ``effort[i]`` on the state topic - the gripper servo's torque in N·m for
  ``name[i] == "gripper"``, from its current; **NaN when the bus can't read
  it**. NaN is the honest "unavailable", and clients must treat it as None,
  not as a number.

The bus works in calibrated degrees; the twin's joint map
(:class:`so101_sim.mapping.JointMap`) converts at this node's topics. The
gripper's torque limit is this node's ``gripper_torque_limit_nm`` parameter.

The control loop thread (provided) is the same observe→act pattern the real
pipeline uses: one thread on the bus, no contention. It writes every fresh
setpoint - the controllers interpolate - through the bus transport's
per-tick clamp. Your TODOs are the ROS wiring around it.

Notebook: ``notebooks/02_ros2/02_robot.ipynb``
Run it for real, with its controllers:
    python -m ros.node_supervisor --only arm --dry-run   # no hardware
    python -m ros.node_supervisor --only arm             # real SO-101 (sole bus owner!)
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time

from so101_sim.control import TORQUE_PARAMETER, server_transport_cfg

from common.config import load_config
from common.motion import (
    DEFAULT_CONTROL_HZ,
    GRIPPER_TORQUE_LIMIT,
    LIVE_MIMIC_LIMITS,
    MOTOR_NAMES,
    REST_POSE,
    DryRunArm,
    clamp_joints,
    make_arm_backend,
    set_active_limits,
)
from common.ros_helpers import joint_state_msg

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]

# Read the gripper effort register every Nth control tick (an extra serial
# transaction - cheap, but no reason to pay it at full rate).
EFFORT_SAMPLE_EVERY_N_TICKS = 3


class So101ServerNode(Node):
    """Bridges an ArmClient backend (real or dry-run) to ros2_control's hardware topics.

    An injected ``arm`` must stream: build it from
    :func:`so101_sim.control.server_transport_cfg`.
    """

    def __init__(self, cfg: dict, arm=None):
        super().__init__("so101_server_node")
        from so101_sim.config import load_sim_config
        from so101_sim.mapping import JointMap

        r = cfg["robot"]
        self.command_topic = r["command_topic"]
        self.state_topic = r["state_topic"]
        self._control_fps = float(r.get("control_fps", DEFAULT_CONTROL_HZ))
        self._arm = arm if arm is not None else DryRunArm()
        # Real hardware behind this node: admit the extended live gesture-
        # mimic envelope end to end. Clients clamp every scripted behavior to
        # the base JOINT_LIMITS before publishing, so only live mimic (which
        # switches its own clamp) ever uses the extra travel. A dry-run node
        # keeps the base limits.
        if getattr(self._arm, "is_live", False):
            set_active_limits(LIVE_MIMIC_LIMITS)
        sim = load_sim_config(cfg)
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        self._current_model = sim["current_model"]
        self._stall_nm = float(sim["current_model"]["stall_torque_nm"])
        self._lock = threading.Lock()
        self._target = dict(REST_POSE)
        self._target_gen = 0  # bumped by a twin reset (reset_with)
        self._last_cmd_time = 0.0
        self._gripper_nm = math.nan  # nan = feedback unavailable
        self._tick_count = 0
        self._last_read = None  # (time, joints) of the previous tick, for velocity
        self._state_pub = None
        self._torque_limit = None  # next gripper Torque_Limit (0-1000) to write
        # PROVIDED: the gripper's torque limit, in N·m, as a ROS parameter.
        self.declare_parameter(TORQUE_PARAMETER, GRIPPER_TORQUE_LIMIT / 1000.0 * self._stall_nm)
        self.add_on_set_parameters_callback(self._on_parameters)
        self._setup_ros()

        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Ready - cmd: {self.resolve_topic_name(self.command_topic)}  "
            f"state: {self.resolve_topic_name(self.state_topic)}  {self._control_fps:.0f} Hz"
        )

    def _setup_ros(self) -> None:
        """Wire the command subscription and the state publisher.

        Both ends use sensor-stream QoS (best-effort / keep-last / depth 1):
        the newest command and the newest state always win - a robot must
        never act on a backlog of stale targets.
        """
        # >>> TODO 2.4: So101ServerNode._setup_ros - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.4 (So101ServerNode._setup_ros): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.4

    def _on_command(self, msg) -> None:
        """Fold one JointState command (radians) into the target pose (degrees).

        Zip ``msg.name``/``msg.position``, keep only known motors
        (:data:`MOTOR_NAMES` - partial commands are legal), convert each with
        ``self._jmap.to_cal(name, radians)``, clamp through
        :func:`clamp_joints` (the safety backstop), and update the shared
        target + freshness timestamp under the lock.
        """
        # >>> TODO 2.5: So101ServerNode._on_command - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.5 (So101ServerNode._on_command): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.5

    def _on_parameters(self, params):
        """PROVIDED: queue a new gripper torque limit (N·m) for the control loop, the only bus thread."""
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name != TORQUE_PARAMETER:
                continue
            nm = float(p.value)
            if not 0.0 <= nm <= self._stall_nm:
                return SetParametersResult(
                    successful=False, reason=f"{TORQUE_PARAMETER} must be within 0..{self._stall_nm} N·m",
                )
            with self._lock:
                self._torque_limit = int(round(nm / self._stall_nm * 1000.0))
        return SetParametersResult(successful=True)

    def reset_with(self, restore) -> dict:
        """PROVIDED: run ``restore`` (a twin reset that returns the pose it
        parked the arm at) under the lock, and make that pose the target, so
        the control loop does not drive the arm back to the goal before it."""
        with self._lock:
            parked = restore()
            self._target = dict(parked)
            self._last_cmd_time = time.time()
            self._target_gen += 1
        return parked

    def _publish_state(self, joints: dict, velocity) -> None:
        """Publish the arm's state in radians, with gripper effort riding in ``effort``.

        Build the message with :func:`common.ros_helpers.joint_state_msg`
        over all :data:`MOTOR_NAMES` (pass ``node=self`` for the timestamp):
        each position through ``self._jmap.to_rad(name, degrees)``, the
        control loop's ``velocity`` (rad/s, aligned with the names; None on
        the first tick), and ``effort`` aligned with ``name``: the sampled
        gripper torque ``self._gripper_nm`` at the gripper's index, NaN
        everywhere else - NaN, not 0, is "no reading".
        """
        # >>> TODO 2.6: So101ServerNode._publish_state - notebooks/02_ros2/02_robot.ipynb
        raise NotImplementedError(
            "TODO 2.6 (So101ServerNode._publish_state): implement me - see notebooks/02_ros2/02_robot.ipynb"
        )
        # <<< TODO 2.6

    def _control_loop(self) -> None:
        """PROVIDED: sequential observe→act loop (one thread on the bus)."""
        from so101_sim.servo import current_to_torque_nm

        interval = 1.0 / self._control_fps
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                target = dict(self._target)
                fresh = (time.time() - self._last_cmd_time) < 1.0
                generation = self._target_gen
                torque_limit, self._torque_limit = self._torque_limit, None
            try:
                if torque_limit is not None:
                    self._arm.set_gripper_torque_limit(torque_limit)
                # Every fresh setpoint is written: the controllers interpolate,
                # and the transport's per-tick clamp is the backstop. Hold the
                # last pose silently when setpoints stop.
                if fresh:
                    with self._lock:
                        fresh = generation == self._target_gen  # a twin reset replaced this target
                    if fresh:
                        self._arm.send_joints(target)
                joints = self._arm.get_joints()
                read_at = time.perf_counter()

                self._tick_count += 1
                if self._tick_count % EFFORT_SAMPLE_EVERY_N_TICKS == 0:
                    load = self._arm.get_gripper_load()
                    self._gripper_nm = math.nan if load is None else current_to_torque_nm(load, self._current_model)

                velocity = None
                if self._last_read is not None:
                    then, before = self._last_read
                    velocity = [
                        (self._jmap.to_rad(m, joints[m]) - self._jmap.to_rad(m, before[m])) / (read_at - then)
                        for m in MOTOR_NAMES
                    ]
                self._last_read = (read_at, joints)
                self._publish_state(joints, velocity)
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
        raise SystemExit("rclpy not importable - source /opt/ros/jazzy/setup.bash first")
    parser = argparse.ArgumentParser(description="SO-101 server node")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no hardware - integrate commands in software",
    )
    args, ros_args = parser.parse_known_args()  # --ros-args ... (the launch's remaps) go to rclpy

    cfg = load_config(args.config)
    rclpy.init(args=ros_args)
    node = So101ServerNode(cfg, arm=make_arm_backend(server_transport_cfg(cfg), dry_run=args.dry_run))
    # One thread: rclpy's MultiThreadedExecutor rebuilds its wait set in a
    # tight loop under the controllers' 60 Hz setpoints, starving the loop.
    executor = SingleThreadedExecutor()
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
