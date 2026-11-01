#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ros2_control on the digital twin: the measurements for the rollout decision.

``launch/so101_control.launch.py`` runs robot_state_publisher, the controller
manager and the controllers against the twin's arm server (pipeline stack,
its topics remapped onto the hardware side). Then:

1. moves - the twin in its own process, as in production. Each keyframe move
   is a one-point FollowJointTrajectory timed by ``servo_profile``'s plan:
   when the action ended, where the arm was then, and when it settled;
2. CPU - ros2_control_node's and the twin's share of one core during the moves;
3. latency - a command's stamp to its arrival on the hardware topic;
4. grasp - the twin in this process, so a ball can be staged between the
   jaws, closed on by each gripper controller: when the forward controller's
   jaw stopped, and what the action controller reported and when. The report
   gives this process's sim time per wall second beside it.

Source ROS, then:

    .venv/bin/python scripts/ros2_control_poc.py [--domain 88] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from so101_sim import control  # noqa: E402

CONFIG = ROOT / "config" / "pipeline.yaml"
BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SETTLED_DEG = 0.5
# Jaws low over the table, closing axis horizontal: how the pickup grasps the ball.
GRASP_POSE = {"shoulder_pan": 0.0, "shoulder_lift": 9.0, "elbow_flex": 20.0, "wrist_flex": 54.0, "wrist_roll": 0.0}


def _moves(cfg: dict) -> list[tuple[str, dict]]:
    """Park, the scan sweep, a reach down to the table, the wave and the dance keyframes."""
    from vla_pipeline.behaviors.dance import DANCE_MOVES
    from vla_pipeline.behaviors.simple import WAVE_UP
    from vla_pipeline.robot.arm_interface import REST_POSE, clamp_joints

    def body(pose: dict) -> dict:
        return {k: v for k, v in pose.items() if k in BODY}

    pp = cfg["behaviors"]["pick_place"]
    scan = clamp_joints({**REST_POSE, **pp["scan_pose"]})
    moves = [("rest to wave", body(clamp_joints({**REST_POSE, **WAVE_UP}))), ("park", body(REST_POSE)),
             ("rest to scan", body(scan))]
    for pan in pp["scan_pans"][1:]:
        moves.append((f"scan pan {pan:+.0f}", {"shoulder_pan": float(pan)}))
    over = clamp_joints({**scan, "shoulder_lift": -15.0, "elbow_flex": 20.0, "wrist_flex": 54.0})
    moves += [("reach over", body(over)), ("down to the table", {"shoulder_lift": 9.0}), ("park", body(REST_POSE))]
    for name, keyframes in DANCE_MOVES.items():
        for i, keyframe in enumerate(keyframes):
            moves.append((f"{name} {i}", {k: v for k, v in keyframe.items() if k in BODY}))
    moves.append(("park", body(REST_POSE)))
    return moves


class Probe:
    """The measuring side: action clients, the hardware topics, the controller manager's activity."""

    def __init__(self, jmap, commands_topic: str, states_topic: str):
        from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
        from controller_manager_msgs.msg import ControllerManagerActivity
        from controller_manager_msgs.srv import SwitchController
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        from vla_pipeline.robot.robot_node import _best_effort_qos

        self.jmap = jmap
        self.node = Node("ros2_control_poc")
        self.cond = threading.Condition()
        self.state: dict[str, float] | None = None  # calibrated units
        self.first_state_at: float | None = None
        self.seq = 0
        self.controllers: dict[str, str] = {}
        self.latency: list[float] = []
        self.record_latency = False
        self.trajectory = ActionClient(self.node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.gripper_action = ActionClient(
            self.node, ParallelGripperCommand, "/gripper_action_controller/gripper_cmd"
        )
        self.gripper_forward = self.node.create_publisher(
            Float64MultiArray, "/gripper_forward_controller/commands", 10
        )
        self.switch = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.node.create_subscription(JointState, states_topic, self._on_state, _best_effort_qos())
        self.node.create_subscription(JointState, commands_topic, self._on_command, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(
            ControllerManagerActivity, "/controller_manager/activity", self._on_activity, latched
        )

    def _on_state(self, msg) -> None:
        state = {n: self.jmap.to_cal(n, p) for n, p in zip(msg.name, msg.position)}
        with self.cond:
            if self.first_state_at is None:
                self.first_state_at = time.monotonic()
            self.state, self.seq = state, self.seq + 1
            self.cond.notify_all()

    def _on_command(self, msg) -> None:
        if self.record_latency:
            sent = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            self.latency.append(time.time() - sent)

    def _on_activity(self, msg) -> None:
        with self.cond:
            self.controllers = {c.name: c.state.label for c in msg.controllers}
            self.cond.notify_all()

    def wait(self, pred, budget_s: float, what: str):
        """One bounded wait for an expected event; raises when it does not come."""
        with self.cond:
            if not self.cond.wait_for(pred, timeout=budget_s):
                raise TimeoutError(f"{what}: not within {budget_s:.1f} s")

    def joints(self) -> dict:
        with self.cond:
            return dict(self.state)

    def close(self) -> None:
        self.trajectory.destroy()
        self.gripper_action.destroy()
        self.node.destroy_node()

    def call(self, future, budget_s: float, what: str):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(budget_s):
            raise TimeoutError(f"{what}: no answer within {budget_s:.1f} s")
        return future.result()


def _error_deg(state: dict, goal: dict) -> float:
    return max(abs(state[k] - v) for k, v in goal.items())


def run_move(probe: Probe, name: str, goal: dict) -> dict:
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    from vla_pipeline.robot.servo_profile import shared_profiler

    start = probe.joints()
    plan = shared_profiler().plan(start, goal)
    eta = float(plan.eta_s)
    msg = FollowJointTrajectory.Goal()
    msg.trajectory.joint_names = list(goal)
    point = JointTrajectoryPoint()
    point.positions = [probe.jmap.to_rad(k, v) for k, v in goal.items()]
    point.time_from_start = Duration(sec=int(eta), nanosec=int(round((eta - int(eta)) * 1e9)))
    msg.trajectory.points = [point]
    sent = time.monotonic()
    handle = probe.call(probe.trajectory.send_goal_async(msg), 5.0, f"{name}: goal acceptance")
    if not handle.accepted:
        raise RuntimeError(f"{name}: goal rejected")
    result = probe.call(handle.get_result_async(), 2.0 * eta + 5.0, f"{name}: result")
    ended = time.monotonic() - sent
    at_end = _error_deg(probe.joints(), goal)
    settled = None
    try:
        probe.wait(lambda: _error_deg(probe.state, goal) < SETTLED_DEG, eta + 1.0, f"{name}: settle")
        settled = time.monotonic() - sent
    except TimeoutError:
        pass
    return {
        "move": name, "planned_s": round(eta, 3), "action_s": round(ended, 3),
        "error_at_end_deg": round(at_end, 2),
        "settled_s": None if settled is None else round(settled, 3),
        "final_error_deg": round(_error_deg(probe.joints(), goal), 2),
        "error_code": int(result.result.error_code),
    }


def _gripper_rad(probe: Probe, cal: float) -> float:
    return probe.jmap.to_rad("gripper", cal)


def grasp_forward(probe: Probe, world, closed: float, opened: float) -> dict:
    from std_msgs.msg import Float64MultiArray

    probe.gripper_forward.publish(Float64MultiArray(data=[_gripper_rad(probe, opened)]))
    probe.wait(lambda: abs(probe.state["gripper"] - opened) < 1.0, 5.0, "forward: jaws open")
    world.stage_object("sports_ball", GRASP_POSE)
    probe.gripper_forward.publish(Float64MultiArray(data=[_gripper_rad(probe, closed)]))
    sent = time.monotonic()
    last = {"jaw": probe.joints()["gripper"], "seq": probe.seq, "still": 0}

    def stopped() -> bool:
        if probe.seq == last["seq"]:
            return False
        jaw = probe.state["gripper"]
        last["still"] = last["still"] + 1 if abs(jaw - last["jaw"]) < 0.05 else 0
        last["jaw"], last["seq"] = jaw, probe.seq
        return last["still"] >= 5 and jaw > closed + 1.0

    probe.wait(stopped, 5.0, "forward: jaw stops on the ball")
    return {"controller": "gripper_forward_controller", "jaw_stopped_s": round(time.monotonic() - sent, 3),
            "jaw": round(last["jaw"], 1), "commanded": closed, "grasped": bool(world.is_grasped("sports_ball"))}


def grasp_action(probe: Probe, world, closed: float, opened: float) -> dict:
    from control_msgs.action import ParallelGripperCommand
    from controller_manager_msgs.srv import SwitchController
    from std_msgs.msg import Float64MultiArray

    probe.gripper_forward.publish(Float64MultiArray(data=[_gripper_rad(probe, opened)]))
    probe.wait(lambda: abs(probe.state["gripper"] - opened) < 1.0, 5.0, "action: jaws open")
    request = SwitchController.Request(
        activate_controllers=["gripper_action_controller"],
        deactivate_controllers=["gripper_forward_controller"],
        strictness=SwitchController.Request.STRICT,
    )
    reply = probe.call(probe.switch.call_async(request), 10.0, "switch to the gripper action controller")
    if not reply.ok:
        raise RuntimeError("switch_controller refused the gripper action controller")
    world.stage_object("sports_ball", GRASP_POSE)
    goal = ParallelGripperCommand.Goal()
    goal.command.name = ["gripper"]
    goal.command.position = [_gripper_rad(probe, closed)]
    sent = time.monotonic()
    handle = probe.call(probe.gripper_action.send_goal_async(goal), 5.0, "gripper goal acceptance")
    if not handle.accepted:
        raise RuntimeError("gripper goal rejected")
    result = probe.call(handle.get_result_async(), 10.0, "gripper result").result
    return {"controller": "gripper_action_controller", "result_s": round(time.monotonic() - sent, 3),
            "stalled": bool(result.stalled), "reached_goal": bool(result.reached_goal),
            "jaw": round(probe.joints()["gripper"], 1), "commanded": closed,
            "grasped": bool(world.is_grasped("sports_ball"))}


def _stop(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGINT)
    proc.wait()


def _wait_exit(pid: int, budget_s: float) -> None:
    """Block until ``pid`` (not necessarily our child) has exited."""
    import select

    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if not select.select([fd], [], [], budget_s)[0]:
            raise TimeoutError(f"pid {pid} still runs {budget_s:.0f} s after SIGINT")
    finally:
        os.close(fd)


class Session:
    """The probe on its own executor, plus the launch; ``extra`` nodes join the executor."""

    def __init__(self, jmap, cfg_path: Path, extra=()):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self.probe = Probe(jmap, control.HARDWARE_COMMANDS, control.HARDWARE_STATES)
        self.executor = SingleThreadedExecutor()
        for n in (*extra, self.probe.node):
            self.executor.add_node(n)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.launch = control.start_controllers(cfg_path, "pipeline")
        self.probe.wait(
            lambda: self.probe.state is not None
            and self.probe.controllers.get("arm_controller") == "active"
            and self.probe.controllers.get("gripper_forward_controller") == "active",
            control.BRING_UP_S, "controllers active",
        )

    def controller_manager(self):
        import psutil

        return next(p for p in psutil.Process(self.launch.pid).children(recursive=True)
                    if "ros2_control_node" in " ".join(p.cmdline()))

    def close(self) -> None:
        import rclpy

        control.stop_controllers(self.launch)
        self.executor.shutdown()
        self.probe.close()
        rclpy.try_shutdown()


def measure_moves(cfg: dict, cfg_path: Path, jmap, profile_server: bool = False,
                  py_spy: str | None = None) -> dict:
    """The twin in its own process, as in production; its server's topics remapped onto the hardware side."""
    import psutil

    env = dict(os.environ, SO101_PROFILE="1") if profile_server else None
    sampler = ["py-spy", "record", "--threads", "--format", "raw", "--rate", "200", "-o", py_spy, "--"] if py_spy else []
    started = time.monotonic()
    twin = subprocess.Popen(
        [*sampler, sys.executable, "-m", "so101_sim.node", "--config", str(cfg_path), "--stack", "pipeline",
         "--no-view", "--ros-args",
         "-r", f"{cfg['robot']['command_topic']}:={control.HARDWARE_COMMANDS}",
         "-r", f"{cfg['robot']['state_topic']}:={control.HARDWARE_STATES}"],
        cwd=str(ROOT), start_new_session=True, env=env,
    )
    session = None
    twin_pid = twin.pid
    try:
        session = Session(jmap, cfg_path)
        probe = session.probe
        twin_proc = psutil.Process(twin.pid)
        if py_spy:
            twin_proc = twin_proc.children()[0]
            twin_pid = twin_proc.pid
        cm = session.controller_manager()
        cm.cpu_percent(None)
        twin_proc.cpu_percent(None)
        probe.record_latency = True
        moves = [run_move(probe, name, goal) for name, goal in _moves(cfg)]
        probe.record_latency = False
        lat = sorted(1000.0 * v for v in probe.latency)
        return {
            "server_first_state_s": round(probe.first_state_at - started, 3),
            "cpu_percent_of_one_core": {"ros2_control_node": round(cm.cpu_percent(None), 1),
                                        "twin": round(twin_proc.cpu_percent(None), 1)},
            "command_latency_ms": {"n": len(lat), "median": round(statistics.median(lat), 2),
                                   "p95": round(lat[int(0.95 * (len(lat) - 1))], 2), "max": round(lat[-1], 2)},
            "moves": moves,
        }
    finally:
        if session is not None:
            session.close()
        _stop(twin)
        _wait_exit(twin_pid, 10.0)


def measure_grasp(cfg: dict, cfg_path: Path, jmap, domain: int) -> dict:
    """The twin in this process, so a ball can be staged between the jaws."""
    from so101_sim.domain_lock import acquire
    from so101_sim.kinematics import OPEN_GRIPPER
    from so101_sim.node import build_nodes
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    lock = acquire(domain)
    import rclpy

    rclpy.init()
    in_process = dict(cfg, robot={**cfg["robot"], "command_topic": control.HARDWARE_COMMANDS,
                                  "state_topic": control.HARDWARE_STATES})
    world, nodes, closers = build_nodes(in_process, stack="pipeline", config_path=str(cfg_path), view=False)
    session = None
    try:
        session = Session(jmap, cfg_path, extra=nodes)
        sim0, wall0 = float(world.data.time), time.monotonic()
        reached = run_move(session.probe, "to the grasp pose", GRASP_POSE)
        if reached["settled_s"] is None:
            raise RuntimeError(f"the arm did not settle at the grasp pose: {reached}")
        grasps = [
            grasp_forward(session.probe, world, GRIPPER_CLOSED_FLOOR, OPEN_GRIPPER),
            grasp_action(session.probe, world, GRIPPER_CLOSED_FLOOR, OPEN_GRIPPER),
        ]
        ratio = (float(world.data.time) - sim0) / (time.monotonic() - wall0)
        return {"sim_time_per_wall_second": round(ratio, 3), "grasps": grasps}
    finally:
        if session is not None:
            session.close()
        for close in reversed(closers):
            close()
        lock.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, default=88, help="private ROS domain for the measurement")
    ap.add_argument("--out", default=str(ROOT / "_artifacts" / "ros2_control_poc" / "report.json"))
    ap.add_argument("--profile-server", action="store_true",
                    help="SO101_PROFILE=1 in the twin: its server logs loop rate and bus timing once a second")
    ap.add_argument("--py-spy", default=None, metavar="FILE",
                    help="run the moves' twin under py-spy record (per-thread collapsed stacks to FILE)")
    args = ap.parse_args()

    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    import yaml

    from so101_sim.config import load_sim_config
    from so101_sim.mapping import JointMap


    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["robot"]["backend"] = "sim"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out.with_suffix(".config.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg))
    sim = load_sim_config(cfg)
    jmap = JointMap(sim["joint_map"], sim["gripper_map"])

    report = {"control_fps": cfg["robot"]["control_fps"]}
    report["moves_phase"] = measure_moves(cfg, cfg_path, jmap, args.profile_server, args.py_spy)
    out.write_text(json.dumps(report, indent=1))
    report["grasp_phase"] = measure_grasp(cfg, cfg_path, jmap, args.domain)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
