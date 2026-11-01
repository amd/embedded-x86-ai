# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 layer for the SO-101 arm, modeled on ``amd_accel_native_node``.

Two halves:

- :class:`So101ServerNode` - owns the Feetech bus (exactly one process may)
  and is ros2_control's hardware side. ``topic_based_ros2_control`` publishes
  the controllers' setpoints on ``robot.command_topic`` and reads
  ``robot.state_topic``; the launch remaps both onto ``/so101/hardware/*``.
  Both are ``sensor_msgs/JointState`` in radians, through the twin's joint
  map; the state also carries each joint's velocity and the gripper's effort
  in N·m. A single daemon control thread does the bus I/O and writes every
  setpoint as it comes - the controllers interpolate - with the same
  threading model as the production node (subscriptions on executor
  threads, one sequential control loop, rest-pose parking on shutdown). The
  gripper's torque limit is the ``gripper_torque_limit_nm`` parameter.
- :class:`Ros2ArmClient` - the :class:`~vla_pipeline.robot.ros2_control_arm.ControllerArm`
  behaviors drive the controllers with, so they stay transport-agnostic: the
  trajectory action, the forward controllers and ``/joint_states``.

Independent test
----------------
    # Terminal 1 - the server and its controllers (dry_run:=true: no motors)
    ros2 launch launch/strix_vla.launch.py dry_run:=true

    # Terminal 2 - client round-trip test
    python -m vla_pipeline.robot.robot_node --client-test
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

from vla_pipeline.robot.arm_interface import (
    DEFAULT_CONTROL_HZ,
    GRASP_LIMITS,
    GRIPPER_TORQUE_LIMIT,
    MOTOR_NAMES,
    REST_POSE,
    ArmClient,
    DryRunArm,
    clamp_joints,
)
from vla_pipeline.robot.ros2_control_arm import BODY, ControllerArm
from vla_pipeline.utils.config import load_config

from so101_sim.control import (  # noqa: E402
    ARM_CONTROLLER,
    BRING_UP_S,
    GRIPPER_CONTROLLER,
    STREAM_CONTROLLER,
    TORQUE_PARAMETER,
    server_transport_cfg,
)
from so101_sim.servo import current_to_torque_nm  # noqa: E402

try:  # ROS 2 is required; without it sourced, _require_ros2() says how to fix that
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]

# Read the gripper effort register every Nth control tick (extra serial
# transaction; at 30 Hz / N=3 this adds ~10 reads/s, negligible on the bus).
EFFORT_SAMPLE_EVERY_N_TICKS = 3

# Transient bus errors ("There is no status packet!") are retried at the
# transport layer (DirectArm._bus_retry in arm_interface). The control loop's
# job is just to survive a transaction that fails anyway: a genuinely dead link
# (power/wiring) fails every tick, so cap the log rate - warn on the first
# failure, then once every Nth after that - instead of flooding the console.
BUS_ERROR_LOG_EVERY_N = 30


def _require_ros2() -> None:
    """Raise if rclpy isn't importable (ROS 2 not sourced)."""
    if not ROS2_AVAILABLE:
        raise RuntimeError(
            "rclpy not importable. Source ROS 2 Jazzy (`source /opt/ros/jazzy/setup.bash`) "
            "before activating .venv - bootstrap.sh creates the venv with "
            "--system-site-packages so the ROS python stack stays visible."
        )


def _best_effort_qos():
    """Return the best-effort, keep-last-1 QoS profile used for joint topics."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _joint_state(node, names, positions, effort=None) -> "JointState":
    """Build a time-stamped ``JointState`` (shared by the server and client)."""
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = names
    msg.position = positions
    if effort is not None:
        msg.effort = effort
    return msg


class So101ServerNode(Node):
    """Owns the SO-101 Feetech bus; ros2_control's hardware side over ROS 2 topics.

    An injected ``arm`` must stream: build it from :func:`server_transport_cfg`.
    """

    def __init__(self, cfg: dict, dry_run: bool = False, arm: ArmClient | None = None):
        _require_ros2()
        super().__init__("so101_server_node")
        from so101_sim.config import load_sim_config
        from so101_sim.mapping import JointMap

        cfg = server_transport_cfg(cfg)
        r = cfg["robot"]
        self._control_fps = float(r.get("control_fps", DEFAULT_CONTROL_HZ))
        self._shutting_down = False
        # Opt-in loop profiling: `SO101_PROFILE=1` makes the control loop report
        # its ACTUAL rate and per-section timing once a second, so motion-quality
        # problems can be traced to a slow/irregular loop instead of guessed at.
        self._profile = os.environ.get("SO101_PROFILE", "") not in (
            "",
            "0",
            "false",
            "False",
        )

        if arm is not None:
            self._arm: ArmClient = arm
            self.get_logger().info(f"Bus transport injected: {type(arm).__name__}")
        elif dry_run or not r.get("enable_motors", True):
            self._arm = DryRunArm(verbose=False)
            self.get_logger().warn("Motors DISABLED - dry-run mode")
        elif r.get("backend", "real") == "sim":
            from vla_pipeline.robot.sim_arm import SimArmClient

            self._arm = SimArmClient(cfg)
            self.get_logger().info("Bus: MuJoCo digital twin (robot.backend=sim)")
        else:
            from vla_pipeline.robot.arm_interface import DirectArm, calibration_exists

            # Calibration is interactive; the server usually runs backgrounded
            # (no TTY), so detect a missing file up front and fail with an
            # actionable message instead of an opaque EOFError from input().
            if not calibration_exists(r["robot_id"]) and not sys.stdin.isatty():
                raise RuntimeError(
                    f"No calibration for robot_id '{r['robot_id']}', and the server "
                    "is running without a terminal so it can't calibrate "
                    "interactively.\nRun calibration once in the foreground first:\n"
                    "    python -m vla_pipeline.robot.arm_interface --calibrate\n"
                    "(run_pipeline.sh now does this automatically before launch)."
                )
            self._arm = DirectArm(r["motor_port"], r["robot_id"], r["max_relative_target"])
            self.get_logger().info(f"SO101Follower connected on {r['motor_port']}")
        # The client already clamped this move. The bus backstop is the grasp
        # envelope (wrist_flex to 90°). The desk cap (67.5°) is what the
        # client applies to ordinary moves; applying it again here pulls a
        # solved toothbrush grasp back up off the object.
        self._arm._send_limits = GRASP_LIMITS

        self._target_lock = threading.Lock()
        self._target: dict[str, float] = dict(REST_POSE)
        self._target_gen = 0
        self._last_cmd_time = 0.0
        self._gripper_load: float = math.nan  # nan = feedback unavailable
        self._tick_count = 0
        self._torque_limit: int | None = None  # next gripper Torque_Limit to write
        sim = load_sim_config(cfg)
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        self._current_model = sim["current_model"]
        self._stall_nm = float(sim["current_model"]["stall_torque_nm"])
        self._last_rad: tuple[float, list[float]] | None = None
        self.declare_parameter(TORQUE_PARAMETER, GRIPPER_TORQUE_LIMIT / 1000.0 * self._stall_nm)
        self.add_on_set_parameters_callback(self._on_parameters)

        commands = self.create_subscription(
            JointState, r["command_topic"], self._on_command, _best_effort_qos()
        )
        self._state_pub = self.create_publisher(
            JointState, r["state_topic"], _best_effort_qos()
        )

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        self.get_logger().info(
            f"Ready - cmd: {commands.topic_name}  state: {self._state_pub.topic_name}  "
            f"{self._control_fps:.0f} Hz"
        )

    def reset_with(self, restore) -> dict:
        """Run ``restore`` (a twin reset that returns the pose it parked the arm
        at) under the target lock, and make that pose the target, so the
        control loop does not drive the arm back to the goal before it."""
        with self._target_lock:
            parked = restore()
            self._target = dict(parked)
            self._last_cmd_time = time.time()
            self._target_gen += 1
        return parked

    def _on_command(self, msg: "JointState") -> None:
        """Merge an incoming setpoint (radians) into the current target."""
        joints = {
            n: self._jmap.to_cal(n, float(p)) for n, p in zip(msg.name, msg.position) if n in MOTOR_NAMES
        }
        if not joints:
            return
        with self._target_lock:
            self._target.update(clamp_joints(joints, GRASP_LIMITS))
            self._last_cmd_time = time.time()

    def _on_parameters(self, params):
        """Queue a new gripper torque limit (N·m) for the control loop, the only bus thread."""
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name != TORQUE_PARAMETER:
                continue
            nm = float(p.value)
            if not 0.0 <= nm <= self._stall_nm:
                return SetParametersResult(
                    successful=False, reason=f"{TORQUE_PARAMETER} must be within 0..{self._stall_nm} N·m",
                )
            with self._target_lock:
                self._torque_limit = int(round(nm / self._stall_nm * 1000.0))
        return SetParametersResult(successful=True)

    def _control_loop(self) -> None:
        """Sequential observe → act loop (one thread on the bus, no contention).

        Both bus transactions - the command *write* and the state *read* - run
        inside one guard. Transient serial errors are retried in the transport
        (DirectArm._bus_retry); if a transaction still fails here, the tick is
        logged and skipped rather than letting the exception escape, kill this
        daemon thread, and leave the arm uncommanded for the rest of the run.

        When ``SO101_PROFILE`` is set, the loop reports its real rate and where
        each tick's time goes once a second - the timing itself is always
        collected (perf_counter is ~free next to the serial I/O) but only logged
        when profiling, so this is a no-op on normal runs.
        """
        interval = 1.0 / self._control_fps
        bus_errors = 0
        # Rolling 1 s profile window.
        p_t0 = time.perf_counter()
        p_last = p_t0
        p_n = p_send = p_read = 0
        p_send_s = p_read_s = p_worst = 0.0
        while self._running:
            t0 = time.perf_counter()
            p_dt = t0 - p_last
            p_last = t0
            if p_dt > p_worst:
                p_worst = p_dt
            with self._target_lock:
                target = dict(self._target)
                fresh = (time.time() - self._last_cmd_time) < 1.0
                generation = self._target_gen
                torque_limit, self._torque_limit = self._torque_limit, None

            try:
                if torque_limit is not None:
                    self._arm.set_gripper_torque_limit(torque_limit)
                if fresh:  # hold last pose silently when setpoints stop
                    with self._target_lock:
                        if generation != self._target_gen:
                            fresh = False  # a scene reset replaced this goal
                    if fresh:
                        _s = time.perf_counter()
                        self._arm.send_joints(target)
                        p_send_s += time.perf_counter() - _s
                        p_send += 1

                _s = time.perf_counter()
                joints = self._arm.get_joints()
                p_read_s += time.perf_counter() - _s
                p_read += 1

                # Sample gripper effort at a reduced rate (same thread → no
                # bus contention; nan keeps the field honest when missing).
                self._tick_count += 1
                if self._tick_count % EFFORT_SAMPLE_EVERY_N_TICKS == 0:
                    load = self._arm.get_gripper_load()
                    self._gripper_load = math.nan if load is None else float(load)

                read_at = time.perf_counter()
                rad = [self._jmap.to_rad(m, joints[m]) for m in MOTOR_NAMES]
                load = self._gripper_load
                grip_nm = math.nan if math.isnan(load) else current_to_torque_nm(load, self._current_model)
                msg = _joint_state(
                    self, MOTOR_NAMES, rad, effort=[grip_nm if m == "gripper" else math.nan for m in MOTOR_NAMES],
                )
                if self._last_rad is not None:
                    then, before = self._last_rad
                    msg.velocity = [(a - b) / (read_at - then) for a, b in zip(rad, before)]
                self._last_rad = (read_at, rad)
                self._state_pub.publish(msg)
                bus_errors = 0
            except Exception as e:  # transport retries exhausted, or shutdown race
                bus_errors += 1
                if not self._shutting_down and bus_errors % BUS_ERROR_LOG_EVERY_N == 1:
                    self.get_logger().warning(
                        f"bus tick failed (x{bus_errors}), continuing: {e}"
                    )

            p_n += 1
            now = time.perf_counter()
            if now - p_t0 >= 1.0:
                if self._profile and not self._shutting_down:
                    self.get_logger().info(
                        "PROFILE loop=%.1fHz (target %.0f)  send=%.1fms  read=%.1fms  worst_tick=%.0fms" % (
                            p_n / (now - p_t0),
                            self._control_fps,
                            1000.0 * p_send_s / max(p_send, 1),
                            1000.0 * p_read_s / max(p_read, 1),
                            1000.0 * p_worst,
                        )
                    )
                p_t0 = now
                p_n = p_send = p_read = 0
                p_send_s = p_read_s = p_worst = 0.0

            elapsed = now - t0
            time.sleep(max(interval - elapsed, 0.0))

    def destroy_node(self) -> None:
        """Stop the control loop, park the arm, and release the bus.

        Join that loop until it has left the bus. Closing on a timeout raced
        the rest goal with a write the loop could still make.
        """
        self._shutting_down = True
        self._running = False
        if self._control_thread.is_alive():
            self._control_thread.join()
        try:
            self._arm.close()  # parks at REST_POSE, then closes the port
        except Exception as e:
            msg = str(e)
            if "Torque_Enable" in msg:
                # Torque-off is disabled on disconnect (see DirectArm.__init__), so
                # this normally won't fire. If it does, a servo didn't ack a torque
                # write during close - non-fatal here (the port is closing), but
                # worth surfacing rather than crashing the shutdown.
                logger.warning(
                    "Torque write failed during shutdown (non-fatal): %s", msg
                )
            else:
                logger.error("Arm shutdown error: %s", e)
        super().destroy_node()


class Ros2ArmClient(ControllerArm):
    """The pipeline's arm over ros2_control (:class:`~vla_pipeline.robot.ros2_control_arm.ControllerArm`)."""

    def __init__(self, cfg: dict, state_timeout_s: float = BRING_UP_S):
        _require_ros2()
        super().__init__(cfg, node_name="so101_arm_client", state_timeout_s=state_timeout_s)

    def _setup_ros(self, node) -> None:
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from std_msgs.msg import Float64MultiArray

        self._trajectory = ActionClient(node, FollowJointTrajectory, f"/{ARM_CONTROLLER}/follow_joint_trajectory")
        self._stream_pub = node.create_publisher(Float64MultiArray, f"/{STREAM_CONTROLLER}/commands", 10)
        self._gripper_pub = node.create_publisher(Float64MultiArray, f"/{GRIPPER_CONTROLLER}/commands", 10)
        node.create_subscription(JointState, "/joint_states", self._on_state, 10)

    def _on_state(self, msg) -> None:
        positions, effort, speed = {}, None, 0.0
        for i, name in enumerate(msg.name):
            if name not in MOTOR_NAMES:
                continue
            positions[name] = self._jmap.to_cal(name, msg.position[i])
            if name == "gripper":
                if len(msg.velocity) > i and math.isfinite(msg.velocity[i]):
                    speed = float(msg.velocity[i])
                if len(msg.effort) > i and math.isfinite(msg.effort[i]):
                    effort = float(msg.effort[i])
        self._update_state(positions, speed, effort)

    def send_joints(self, joints: dict) -> None:
        from std_msgs.msg import Float64MultiArray

        joints = clamp_joints(joints, self._send_limits)
        if "gripper" in joints:
            self._command_gripper(joints["gripper"])
        body = self._stream_setpoint(joints)
        if body is not None:
            self._stream_pub.publish(Float64MultiArray(data=[self._jmap.to_rad(k, body[k]) for k in BODY]))

    def reset_sim(self) -> bool:
        """Park the arm and put the twin's props back (:meth:`reset_twin`)."""
        return self.reset_twin()


# =============================================================================
# Standalone component tests
# =============================================================================


def _run_server(cfg: dict, dry_run: bool, ros_args: list[str]) -> None:
    """Spin up rclpy, run So101ServerNode until interrupted, and shut down cleanly."""
    rclpy.init(args=ros_args)
    node = So101ServerNode(cfg, dry_run=dry_run)
    # One thread: the callbacks are short (the bus has its own thread), and
    # rclpy's MultiThreadedExecutor rebuilds its wait set in a tight loop under
    # the controllers' 60 Hz setpoints, which starves the control loop.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down (keyboard interrupt)...")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _run_client_test(cfg: dict) -> None:
    """Round-trip test: connect a Ros2ArmClient and exercise the gripper over ROS 2."""
    arm = Ros2ArmClient(cfg)
    try:
        print("State:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Commanding gripper sweep over ROS 2...")
        arm.go_to({"gripper": 60.0}, duration_s=1.5)
        arm.go_to({"gripper": 10.0}, duration_s=1.5)
        arm.go_to_rest()
        print("ROS 2 round-trip PASSED")
    finally:
        arm.close()


def main() -> None:
    """CLI entry point: run the server node or a client round-trip test."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 ROS 2 node / client - component test"
    )
    parser.add_argument("--config", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--server", action="store_true", help="run the bus-owning server node"
    )
    mode.add_argument(
        "--client-test",
        action="store_true",
        help="round-trip test against a running server",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="server without hardware"
    )
    args, ros_args = parser.parse_known_args()  # --ros-args ... (the launch's remaps) go to rclpy

    cfg = load_config(args.config)
    if args.server:
        _run_server(cfg, args.dry_run, ros_args)
    else:
        _run_client_test(cfg)


if __name__ == "__main__":
    main()
