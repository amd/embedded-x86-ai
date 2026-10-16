# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 layer for the SO-101 arm, modeled on ``amd_accel_native_node``.

Two halves:

- :class:`So101ServerNode` - owns the Feetech bus (exactly one process may).
  Subscribes ``robot.command_topic`` (``sensor_msgs/JointState``, degrees),
  publishes ``robot.state_topic`` at ``control_fps``. A single daemon control
  thread does the bus I/O - the same threading model as the production node
  (subscriptions on executor threads, one sequential control loop, rest-pose
  parking on shutdown).
- :class:`Ros2ArmClient` - implements the :class:`ArmClient` API on top of
  those topics so behaviors are transport-agnostic. It spins its own
  background executor, so the orchestrator stays a plain Python program.

New in this revision: the server samples the gripper's effort/current
(:meth:`ArmClient.get_gripper_load`, runtime-detected) every few control
ticks and publishes it in ``JointState.effort``; ``Ros2ArmClient`` exposes it
back through ``get_gripper_load()`` so the feedback-based grip works
identically over ROS 2 and over the direct serial transport.

Independent test
----------------
    # Terminal 1 - start the server node (or use --dry-run without hardware)
    python -m vla_pipeline.robot.robot_node --server [--dry-run]

    # Terminal 2 - client round-trip test over DDS
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
    GRASP_LIMITS,
    MOTOR_NAMES,
    REST_POSE,
    ArmClient,
    DryRunArm,
    clamp_joints,
)
from vla_pipeline.utils.config import load_config

try:  # ROS 2 is optional at import time so non-ROS machines can run dry tests
    import rclpy
    from rclpy.executors import (
        ExternalShutdownException,
        MultiThreadedExecutor,
        SingleThreadedExecutor,
    )
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Int32

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]

# Read the gripper effort register every Nth control tick (extra serial
# transaction; at 30 Hz / N=3 this adds ~10 reads/s, negligible on the bus).
EFFORT_SAMPLE_EVERY_N_TICKS = 3

# Transient bus errors ("There is no status packet!") are retried at the
# transport layer (LeRobotArm._bus_retry in common.motion). The control loop's
# job is just to survive a transaction that fails anyway: a genuinely dead link
# (power/wiring) fails every tick, so cap the log rate - warn on the first
# failure, then once every Nth after that - instead of flooding the console.
BUS_ERROR_LOG_EVERY_N = 30

# Gripper Torque_Limit (std_msgs/Int32, 0-1000). Reliable: one write, not a stream.
GRIPPER_TORQUE_TOPIC = "/so101/gripper_torque_limit"


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
    """Owns the SO-101 Feetech bus; bridges it to ROS 2 topics."""

    def __init__(self, cfg: dict, dry_run: bool = False, arm: ArmClient | None = None):
        _require_ros2()
        super().__init__("so101_server_node")
        r = cfg["robot"]
        self._control_fps = float(r.get("control_fps", 30.0))
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
            from vla_pipeline.robot.arm_interface import LeRobotArm, calibration_exists

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
            self._arm = LeRobotArm(
                r["motor_port"], r["robot_id"], r.get("max_relative_target", 8.0)
            )
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
        # Direct-goal mode (clamp disabled): the servo profiles its own way to
        # each goal, so the loop must write a Goal_Position only when it CHANGES
        # - re-writing the same goal every tick would restart that onboard
        # profile and reintroduce the stepping. _last_sent_target tracks the last
        # value actually written so unchanged ticks are skipped.
        self._direct_goal = r.get("max_relative_target", 8.0) is None
        self._last_sent_target: dict[str, float] | None = None
        self._torque_limit: int | None = None  # next gripper Torque_Limit to write

        self.create_subscription(
            JointState, r["command_topic"], self._on_command, _best_effort_qos()
        )
        self.create_subscription(Int32, GRIPPER_TORQUE_TOPIC, self._on_torque_limit, 10)
        if getattr(self._arm, "world", None) is not None:
            from std_srvs.srv import Trigger

            self.create_service(Trigger, "/so101/reset_scene", self._on_reset_scene)
        self._state_pub = self.create_publisher(
            JointState, r["state_topic"], _best_effort_qos()
        )

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        self.get_logger().info(
            f"Ready - cmd: {r['command_topic']}  state: {r['state_topic']}  "
            f"{self._control_fps:.0f} Hz"
        )

    def _on_reset_scene(self, _request, response):
        """Park the twin and put its props back on their starting spots."""
        from vla_pipeline.robot.arm_interface import restore_twin

        with self._target_lock:
            parked = restore_twin(self._arm)
            self._target = dict(parked)
            self._last_cmd_time = time.time()
            self._last_sent_target = None
            self._target_gen += 1
        response.success = True
        response.message = "arm parked, objects back on the table"
        return response

    def _on_command(self, msg: "JointState") -> None:
        """Merge an incoming JointState command into the current target."""
        joints = {
            n: float(p) for n, p in zip(msg.name, msg.position) if n in MOTOR_NAMES
        }
        if not joints:
            return
        with self._target_lock:
            self._target.update(clamp_joints(joints, GRASP_LIMITS))
            self._last_cmd_time = time.time()

    def _on_torque_limit(self, msg: "Int32") -> None:
        """Queue a gripper Torque_Limit for the control loop (the only bus thread)."""
        with self._target_lock:
            self._torque_limit = int(msg.data)

    def _control_loop(self) -> None:
        """Sequential observe → act loop (one thread on the bus, no contention).

        Both bus transactions - the command *write* and the state *read* - run
        inside one guard. Transient serial errors are retried in the transport
        (LeRobotArm._bus_retry); if a transaction still fails here, the tick is
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
                if fresh:  # hold last pose silently when commands stop
                    with self._target_lock:
                        if generation != self._target_gen:
                            fresh = False  # a scene reset replaced this goal
                    # Direct-goal mode: write only when the goal changes so the
                    # servo's onboard profile runs once, uninterrupted. Streaming
                    # mode writes every tick (each tick carries a new setpoint).
                    if fresh and (not self._direct_goal or target != self._last_sent_target):
                        _s = time.perf_counter()
                        self._arm.send_joints(target)
                        p_send_s += time.perf_counter() - _s
                        p_send += 1
                        self._last_sent_target = target

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

                msg = _joint_state(
                    self,
                    MOTOR_NAMES,
                    [joints[m] for m in MOTOR_NAMES],
                    effort=[
                        self._gripper_load if m == "gripper" else math.nan
                        for m in MOTOR_NAMES
                    ],
                )
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
                        "PROFILE loop=%.1fHz (target %.0f)  send=%.1fms  read=%.1fms  "
                        "worst_tick=%.0fms",
                        p_n / (now - p_t0),
                        self._control_fps,
                        1000.0 * p_send_s / max(p_send, 1),
                        1000.0 * p_read_s / max(p_read, 1),
                        1000.0 * p_worst,
                    )
                p_t0 = now
                p_n = p_send = p_read = 0
                p_send_s = p_read_s = p_worst = 0.0

            elapsed = now - t0
            time.sleep(max(interval - elapsed, 0.0))

    def set_coordinated_speeds(self, travels: dict[str, float]) -> None:
        """Scale per-joint servo speed so joints finish together (speed ∝ travel).
        No-op if the arm doesn't support it (e.g. dry-run)."""
        fn = getattr(self._arm, "set_coordinated_speeds", None)
        if callable(fn):
            fn(travels)

    def reset_joint_speeds(self) -> None:
        """Restore the default direct-mode speed on the joints last coordinated."""
        fn = getattr(self._arm, "set_joint_speeds", None)
        if callable(fn):
            fn(None)

    def destroy_node(self) -> None:
        """Stop the control loop, park the arm, and release the bus before the base class teardown."""
        self._shutting_down = True
        self._running = False
        if self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        try:
            self._arm.close()  # parks at REST_POSE, then closes the port
        except Exception as e:
            msg = str(e)
            if "Torque_Enable" in msg:
                # Torque-off is disabled on disconnect (see LeRobotArm.__init__), so
                # this normally won't fire. If it does, a servo didn't ack a torque
                # write during close - non-fatal here (the port is closing), but
                # worth surfacing rather than crashing the shutdown.
                logger.warning(
                    "Torque write failed during shutdown (non-fatal): %s", msg
                )
            else:
                logger.error("Arm shutdown error: %s", e)
        super().destroy_node()


class Ros2ArmClient(ArmClient):
    """ArmClient over the server node's topics; spins a private executor."""

    def __init__(self, cfg: dict, state_timeout_s: float = 5.0):
        _require_ros2()
        r = cfg["robot"]
        # Mirror the server's mode: with the clamp off, go_to() sends one goal and
        # waits (move_to) instead of streaming an interpolated trajectory.
        self._direct_goal = r.get("max_relative_target", 8.0) is None
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("so101_arm_client")
        self._pub = self._node.create_publisher(
            JointState, r["command_topic"], _best_effort_qos()
        )
        self._torque_pub = self._node.create_publisher(Int32, GRIPPER_TORQUE_TOPIC, 10)
        self._state_lock = threading.Lock()
        self._state: dict[str, float] | None = None
        self._effort: float | None = None
        self._node.create_subscription(
            JointState, r["state_topic"], self._on_state, _best_effort_qos()
        )

        # One subscription callback to service, so a single-threaded executor is
        # the right fit - and it avoids rclpy's "MultiThreadedExecutor is used
        # with a single thread" warning seen in the logs.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_stop = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True, name="so101_arm_client")
        self._spin_thread.start()

        deadline = time.time() + state_timeout_s
        while time.time() < deadline:
            with self._state_lock:
                if self._state is not None:
                    logger.info("Ros2ArmClient: state stream live.")
                    return
            time.sleep(0.05)
        self.close()
        raise TimeoutError(
            f"No JointState on {r['state_topic']} within {state_timeout_s}s - "
            "is the server node running? (python -m vla_pipeline.robot.robot_node --server)"
        )

    def _spin(self) -> None:
        """Spin in short slices until :meth:`close`. A thread still inside
        ``executor.spin()`` when Python exits aborts the process ("terminate
        called without an active exception"), so ``close()`` must join it."""
        while not self._spin_stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                return

    def _on_state(self, msg: "JointState") -> None:
        """Update the cached joint state and gripper effort from an incoming JointState."""
        state = {n: float(p) for n, p in zip(msg.name, msg.position)}
        effort: float | None = None
        if msg.effort:
            for n, e in zip(msg.name, msg.effort):
                if n == "gripper" and not math.isnan(e):
                    effort = float(e)
        with self._state_lock:
            self._state = state
            self._effort = effort

    def reset_sim(self) -> bool:
        """Ask the digital-twin server to park the arm and restore the props."""
        from std_srvs.srv import Trigger

        if not hasattr(self, "_reset_client"):
            self._reset_client = self._node.create_client(Trigger, "/so101/reset_scene")
        if not self._reset_client.wait_for_service(timeout_sec=0.5):
            return False
        future = self._reset_client.call_async(Trigger.Request())
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        result = future.result() if future.done() else None
        return bool(result and result.success)

    def send_joints(self, joints: dict[str, float]) -> None:
        """Publish joints as a JointState command.

        ``go_to(limits=GRASP_LIMITS)`` has already set ``_send_limits``.
        Clamping with the desk cap here drops a 90° wrist to 67.5° before
        the server ever sees the grasp.
        """
        joints = clamp_joints(joints, self._send_limits)
        names = list(joints.keys())
        self._pub.publish(_joint_state(self._node, names, [joints[k] for k in names]))

    def get_joints(self) -> dict[str, float]:
        """Return the most recently received joint state."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("No arm state received yet.")
            return dict(self._state)

    def get_gripper_load(self) -> float | None:
        """Return the most recently received gripper effort, if any."""
        with self._state_lock:
            return self._effort

    def set_gripper_torque_limit(self, value: int) -> None:
        """Publish the gripper Torque_Limit; the server writes it on its next tick."""
        msg = Int32()
        msg.data = int(min(max(int(value), 0), 1000))
        self._torque_pub.publish(msg)
        self.gripper_torque_limit = msg.data

    def close(self) -> None:
        """Stop and join the spin thread, then shut down the executor and destroy the node."""
        self._spin_stop.set()
        self._spin_thread.join(timeout=2.0)
        self._executor.shutdown()
        self._node.destroy_node()


# =============================================================================
# Standalone component tests
# =============================================================================


def _run_server(cfg: dict, dry_run: bool) -> None:
    """Spin up rclpy, run So101ServerNode until interrupted, and shut down cleanly."""
    rclpy.init()
    node = So101ServerNode(cfg, dry_run=dry_run)
    executor = MultiThreadedExecutor(num_threads=2)
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.server:
        _run_server(cfg, args.dry_run)
    else:
        _run_client_test(cfg)


if __name__ == "__main__":
    main()
