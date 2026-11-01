# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The arm over ros2_control.

``launch/so101_control.launch.py`` runs the controllers on top of the arm
server (:mod:`so101_sim.control` names them and starts the launch).
:class:`ControllerArm` drives them; a subclass wires its ROS endpoints and
message conversions (``Ros2ArmClient`` in :mod:`vla_pipeline.robot.robot_node`):

- a move (:meth:`~ControllerArm.move_to`, and ``go_to``) is one
  ``FollowJointTrajectory`` goal to ``arm_controller``, timed by the servo
  profile planner (:mod:`vla_pipeline.robot.servo_profile`) and carrying the caller's
  tolerance; the controller's result says whether the joints got there;
- a streamed setpoint (:meth:`~ControllerArm.send_joints`: gesture mimic,
  person tracking) goes to ``arm_forward_controller``, switched in on the
  first one; the next move switches ``arm_controller`` back;
- the gripper is ``gripper_forward_controller``;
- the state is ``/joint_states`` from ``joint_state_broadcaster``.

Joint values are radians on the ROS side and calibrated units for behaviors,
through the twin's joint map. The gripper's effort is N·m on the wire and
``Present_Current`` counts for behaviors, through the datasheet line in
:mod:`so101_sim.servo`. Its torque limit is the arm server's
``gripper_torque_limit_nm`` parameter.

The workshop keeps its own copy (``workshop/common/ros2_control_arm.py``);
``tests/sim/test_copies.py`` fails when the two drift apart.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import abstractmethod

from so101_sim.control import (
    ARM_CONTROLLER,
    BRING_UP_S,
    GRIPPER_CONTROLLER,
    JOINT_STATE_STALE_S,
    SERVER_NODE,
    STREAM_CONTROLLER,
    TORQUE_PARAMETER,
    goal_time_s,
)

from vla_pipeline.robot.arm_interface import (
    MOTOR_NAMES, REST_POSE, ArmClient, MotionTimeout, clamp_joints, joints_text,
)

logger = logging.getLogger(__name__)

BODY = tuple(MOTOR_NAMES[:5])
# A controller's or service's reply takes one controller update and a local DDS
# hop (1.3 ms at most, scripts/ros2_control_poc.py): this bound only catches a
# controller manager or arm server that is gone. It is the hardware side's
# joint-state silence bound, so the two cannot drift apart.
REPLY_S = JOINT_STATE_STALE_S


class Spinner:
    """One rclpy node on its own executor thread, joined by :meth:`close`.

    A thread still inside ``executor.spin()`` when Python exits aborts the
    process ("terminate called without an active exception"), and
    ``executor.shutdown()`` does not end it. So the thread runs one callback
    at a time, and ``close()`` triggers a guard condition that wakes it to
    see the stop, then joins it before shutting the executor down.
    """

    def __init__(self, name: str):
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(name)
        self._wake = self.node.create_guard_condition(lambda: None)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def add_node(self, node) -> None:
        """Spin ``node`` on this thread too; its owner still destroys it."""
        self._executor.add_node(node)
        self._wake.trigger()

    def _loop(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once()
            except ExternalShutdownException:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.trigger()
        self._thread.join(timeout=2.0)
        self._executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass


class ControllerArm(ArmClient):
    """:class:`~vla_pipeline.robot.arm_interface.ArmClient` over ros2_control; spins a private node.

    A subclass provides:

    - :meth:`_setup_ros`: ``self._trajectory``, the ``FollowJointTrajectory``
      action client of ``arm_controller``; ``self._stream_pub`` and
      ``self._gripper_pub``, ``std_msgs/Float64MultiArray`` publishers on the
      forward controllers' ``commands`` topics; and a ``/joint_states``
      subscription to :meth:`_on_state`;
    - :meth:`_on_state`: one ``/joint_states`` message to :meth:`_update_state`;
    - :meth:`send_joints`: a streamed setpoint, through :meth:`_command_gripper`
      and :meth:`_stream_setpoint`.
    """

    is_live = True
    # go_to() is one trajectory goal: the controller interpolates the move.
    _direct_goal = True

    def __init__(self, cfg: dict, node_name: str = "so101_arm_client", state_timeout_s: float = BRING_UP_S):
        from control_msgs.action import FollowJointTrajectory
        from controller_manager_msgs.msg import ControllerManagerActivity
        from controller_manager_msgs.srv import SwitchController
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float64MultiArray
        from std_srvs.srv import Trigger

        from so101_sim.config import load_sim_config
        from so101_sim.mapping import JointMap
        from so101_sim.node import RESET_SERVICE

        sim = load_sim_config(cfg)
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        self._current_model = sim["current_model"]
        self._stall_nm = float(sim["current_model"]["stall_torque_nm"])
        control_fps = float(cfg["robot"]["control_fps"])
        # The server's position-difference velocity reads one encoder step per
        # tick while still; below one and a half of those the jaw has stopped.
        self._jaw_still_rad_s = 1.5 * self._jmap.steps_to_rad("gripper", 1.0) * control_fps
        self._goal_time_s = goal_time_s()
        self._FollowJointTrajectory = FollowJointTrajectory
        self._SwitchController = SwitchController
        self._Float64MultiArray = Float64MultiArray

        self._cond = threading.Condition()
        self._state: dict | None = None  # calibrated units
        self._jaw_speed = 0.0  # rad/s
        self._effort_nm: float | None = None
        self._state_t = 0.0  # when the newest state arrived (monotonic)
        self._controllers: dict[str, str] = {}  # name -> state label, from the controller manager
        self._streaming = False
        self._stream: dict | None = None  # the last streamed body setpoint
        self._command_lock = threading.RLock()
        self._trajectory = self._stream_pub = self._gripper_pub = None
        self._spin = Spinner(node_name)
        node = self._spin.node
        self._switch = node.create_client(SwitchController, "/controller_manager/switch_controller")
        self._params = AsyncParameterClient(node, SERVER_NODE)
        # Only a twin serves it; made now, so it is discovered with the server.
        self._reset = node.create_client(Trigger, RESET_SERVICE)
        self._Trigger = Trigger
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", self._on_activity, latched)
        try:
            self._setup_ros(node)
            self._wait(
                lambda: self._state is not None and self._controllers_ready() and self._trajectory.server_is_ready()
                and self._switch.service_is_ready() and self._params.services_are_ready(),
                state_timeout_s,
                "/joint_states, the controllers, the trajectory action, the controller switch "
                "and the arm server's parameters",
            )
        except BaseException:
            self.close()
            raise
        self._streaming = self._controllers[STREAM_CONTROLLER] == "active"
        logger.info("ControllerArm: state stream and controllers live.")

    # -- the subclass's part --------------------------------------------------
    @abstractmethod
    def _setup_ros(self, node) -> None:
        """Create the trajectory action client, the forward controllers'
        publishers and the ``/joint_states`` subscription on ``node``."""

    @abstractmethod
    def _on_state(self, msg) -> None:
        """Convert one ``/joint_states`` message and pass it to :meth:`_update_state`."""

    @abstractmethod
    def send_joints(self, joints: dict) -> None:
        """Stream a setpoint: the body to the forward controller, the gripper to its own."""

    # -- state ---------------------------------------------------------------
    def _update_state(self, positions: dict, jaw_speed: float, effort_nm: float | None) -> None:
        """Store a state (calibrated units; the jaw's speed in rad/s; the gripper's
        effort in N·m, None when unreadable) and wake every wait."""
        with self._cond:
            self._state = {**(self._state or {}), **positions}
            self._jaw_speed, self._effort_nm = jaw_speed, effort_nm
            self._state_t = time.monotonic()
            self._cond.notify_all()

    def _on_activity(self, msg) -> None:
        with self._cond:
            self._controllers = {c.name: c.state.label for c in msg.controllers}
            self._cond.notify_all()

    def _controllers_ready(self) -> bool:
        """The gripper controller active, one arm controller active and the other loaded."""
        arm = (self._controllers.get(ARM_CONTROLLER), self._controllers.get(STREAM_CONTROLLER))
        return self._controllers.get(GRIPPER_CONTROLLER) == "active" and arm in (
            ("active", "inactive"), ("inactive", "active"),
        )

    def _wait(self, pred, budget_s: float, what: str, stop_check=None) -> bool:
        """Wake on each joint state until ``pred()``; False when ``stop_check`` fires first."""
        with self._cond:
            ok = self._cond.wait_for(lambda: pred() or bool(stop_check and stop_check()), budget_s)
            if not ok:
                raise MotionTimeout(f"no {what} within {budget_s:.2f} s")
            return pred()

    def state_time(self) -> float:
        """When the newest ``/joint_states`` message arrived (monotonic seconds)."""
        with self._cond:
            return self._state_t

    def wait_state(self, t: float, stop_check=None) -> float | None:
        """The first ``/joint_states`` message to arrive at or after ``t``; a
        state stream that stops raises :class:`MotionTimeout`."""
        budget = max(t - time.monotonic(), 0.0) + REPLY_S
        if not self._wait(lambda: self._state_t >= t, budget, "joint state", stop_check):
            return None
        with self._cond:
            return self._state_t

    def get_joints(self) -> dict:
        with self._cond:
            if self._state is None:
                raise RuntimeError("No arm state received yet.")
            return dict(self._state)

    def get_gripper_load(self):
        """The gripper's ``Present_Current`` counts, from its effort in N·m."""
        from so101_sim.servo import torque_nm_to_current

        with self._cond:
            effort = self._effort_nm
        return None if effort is None else torque_nm_to_current(effort, self._current_model)

    # -- controllers ---------------------------------------------------------
    def _call(self, future, what: str):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(REPLY_S):
            raise MotionTimeout(f"no reply to {what} within {REPLY_S:.1f} s")
        return future.result()

    def _use_stream(self, streaming: bool) -> None:
        """Switch between the trajectory controller and the forward controller."""
        with self._command_lock:
            if streaming == self._streaming:
                return
            on, off = (STREAM_CONTROLLER, ARM_CONTROLLER) if streaming else (ARM_CONTROLLER, STREAM_CONTROLLER)
            request = self._SwitchController.Request(
                activate_controllers=[on], deactivate_controllers=[off],
                strictness=self._SwitchController.Request.STRICT, activate_asap=True,
            )
            if not self._call(self._switch.call_async(request), f"switching to {on}").ok:
                raise RuntimeError(f"the controller manager refused to switch to {on}")
            self._streaming = streaming
            self._stream = None

    def _command_gripper(self, cal: float) -> None:
        """Command the gripper's forward controller (calibrated units)."""
        self._gripper_pub.publish(self._Float64MultiArray(data=[self._jmap.to_rad("gripper", cal)]))

    def _stream_setpoint(self, joints: dict) -> dict | None:
        """The whole body setpoint, in :data:`BODY` order, that ``joints`` makes:
        the body joints it names, the last streamed (or measured) value for the
        rest; None when it names none. Switches the forward controller in."""
        with self._command_lock:
            body = {k: v for k, v in joints.items() if k in BODY}
            if not body:
                return None
            self._use_stream(True)
            base = self._stream or self.get_joints()
            self._stream = {k: body.get(k, base[k]) for k in BODY}
            return dict(self._stream)

    def move_to(
        self, target: dict, tolerance_deg: float = 2.5, stop_check=None, limits: dict | None = None,
    ) -> str:
        """One trajectory goal for the body, a command for the gripper; returns
        ``"arrived"``, ``"blocked"`` (the controller ended the goal outside
        ``tolerance_deg``) or ``"stopped"`` (``stop_check`` fired; the arm holds
        where it is). A gripper-only move ends when the jaw stops."""
        from builtin_interfaces.msg import Duration
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint

        from vla_pipeline.robot.servo_profile import shared_profiler

        with self._command_lock:
            previous = self._send_limits
            self._send_limits = limits
            try:
                target = clamp_joints(target, limits)
                start = self.get_joints()
                body = {k: v for k, v in target.items() if k in BODY}
                eta_s = float(shared_profiler().plan(start, body or target).eta_s)
                sent_at = time.time()
                if "gripper" in target:
                    self._command_gripper(target["gripper"])
                if body:
                    self._use_stream(False)
                    goal = self._FollowJointTrajectory.Goal()
                    goal.trajectory.joint_names = list(body)
                    point = JointTrajectoryPoint()
                    point.positions = [self._jmap.to_rad(k, v) for k, v in body.items()]
                    point.time_from_start = Duration(
                        sec=int(eta_s),
                        nanosec=int(round((eta_s - int(eta_s)) * 1e9)) % 1_000_000_000,
                    )
                    goal.trajectory.points = [point]
                    goal.goal_tolerance = [
                        JointTolerance(name=k, position=math.radians(tolerance_deg))
                        for k in body
                    ]
                    how = self._run(goal, eta_s, stop_check)
                else:
                    how = self._settle_jaw(
                        start["gripper"], target["gripper"], eta_s, tolerance_deg, stop_check
                    )
                # The controller judged the end on a state newer than ours: callers read the arm next.
                self.wait_state(time.monotonic())
                joints = self.get_joints()
                worst = max(body or list(target), key=lambda k: abs(joints.get(k, target[k]) - target[k]))
                logger.log(
                    logging.WARNING if how == "blocked" else logging.INFO,
                    "move_to %s in %.2f s (servo profile %.2f s): %.1f deg left on %s%s",
                    how, time.time() - sent_at, eta_s,
                    abs(joints.get(worst, target[worst]) - target[worst]), worst,
                    "" if how == "arrived" else f" | at {joints_text(joints)} | goal {joints_text(target)}",
                )
                return how
            finally:
                self._send_limits = previous

    def _run(self, goal, eta_s: float, stop_check) -> str:
        """Send ``goal`` and wait for the controller's result, cancelling it on a stop."""
        from control_msgs.action import FollowJointTrajectory

        handle = self._call(self._trajectory.send_goal_async(goal), "the trajectory goal")
        if not handle.accepted:
            raise RuntimeError("arm_controller rejected the trajectory goal")
        result = handle.get_result_async()
        finished = threading.Event()

        def done(_f) -> None:
            finished.set()
            with self._cond:
                self._cond.notify_all()

        result.add_done_callback(done)
        budget = eta_s + self._goal_time_s + REPLY_S
        try:
            stopped = not self._wait(finished.is_set, budget, "trajectory result", stop_check)
        except MotionTimeout:
            self._cancel_goal(handle, "trajectory budget expired")
            raise
        if stopped:
            self._cancel_goal(handle, "stop requested")
            return "stopped"
        result_msg = result.result()
        from action_msgs.msg import GoalStatus
        status = result_msg.status
        code = result_msg.result.error_code
        # An unmet goal is aborted, with GOAL_TOLERANCE_VIOLATED. Succeeded is the
        # only status whose SUCCESSFUL code means the arm arrived.
        if status == GoalStatus.STATUS_CANCELED:
            return "stopped"
        if code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED and status in (
            GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_SUCCEEDED,
        ):
            return "blocked"
        if status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"arm_controller ended trajectory with action status {status} "
                f"and error code {code}: {result_msg.result.error_string}"
            )
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            return "arrived"
        raise RuntimeError(
            f"arm_controller ended the trajectory with error code {code}: {result_msg.result.error_string}"
        )

    def _cancel_goal(self, handle, reason: str) -> None:
        """Cancel a live trajectory and reject an unsuccessful cancellation."""
        response = self._call(handle.cancel_goal_async(), f"cancelling the trajectory ({reason})")
        if response.return_code != response.ERROR_NONE:
            raise RuntimeError(
                f"arm_controller refused to cancel the trajectory ({reason}); "
                f"return code {response.return_code}"
            )

    def _settle_jaw(self, start: float, goal: float, eta_s: float, tolerance_deg: float, stop_check) -> str:
        """Wait until the jaw is within ``tolerance_deg`` of ``goal``, or has come
        ``tolerance_deg`` closer to it and stopped (on an object, or short of a
        readback that need not converge). A jaw squeezing an object jitters by an
        encoder step a tick, so its speed alone says nothing until it has moved."""
        toward = math.copysign(1.0, goal - start)

        def still() -> bool:
            jaw = self._state["gripper"]
            if abs(jaw - goal) <= tolerance_deg:
                return True
            return (jaw - start) * toward >= tolerance_deg and abs(self._jaw_speed) < self._jaw_still_rad_s

        if not self._wait(still, eta_s + self._goal_time_s + REPLY_S, "end of the gripper move", stop_check):
            self._command_gripper(self.get_joints()["gripper"])
            return "stopped"
        return "arrived"

    def set_gripper_torque_limit(self, value: int) -> None:
        """Cap the gripper at ``value`` (0-1000) of stall torque: the arm server's
        ``gripper_torque_limit_nm`` parameter, written on its next control tick."""
        from rclpy.parameter import Parameter

        value = int(min(max(int(value), 0), 1000))
        nm = value / 1000.0 * self._stall_nm
        results = self._call(self._params.set_parameters([Parameter(TORQUE_PARAMETER, value=nm)]), TORQUE_PARAMETER)
        if not all(r.successful for r in results.results):
            raise RuntimeError(f"the arm server refused {TORQUE_PARAMETER}={nm:.3f}")
        self.gripper_torque_limit = value

    def reset_twin(self) -> bool:
        """Park through the controllers, then ask the twin to put the props back
        (``/sim/reset_scene``); False when no twin serves it. The controller
        must end up holding the parked pose: a twin snapped there behind its
        back is pulled back to the controller's last goal."""
        from so101_sim.world import parked_pose

        if not self._reset.service_is_ready():
            return False
        parked = parked_pose(REST_POSE)
        how = self.move_to(parked)
        result = self._call(self._reset.call_async(self._Trigger.Request()), "the twin reset")
        if how != "arrived":
            # The controller holds where the park move ended, not the parked pose.
            self.move_to(parked)
        return bool(result.success)

    def close(self) -> None:
        self._spin.close()
