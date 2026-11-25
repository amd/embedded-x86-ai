# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ros2_control for the SO-101: the names both stacks agree on, and its launch.

``launch/so101_control.launch.py`` runs the controllers on top of an arm
server (the pipeline's, the workshop's, or the twin hosting either). The
arm server and each stack's client (``ControllerArm``) meet on the names
here: the hardware side's topics, the controllers, the server node and its
torque parameter. The launch files, the twin and both stacks import them
from this one place.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from so101_sim.parent_death import STOP_S

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_FILE = REPO_ROOT / "launch" / "so101_control.launch.py"
# topic_based_ros2_control, built by bootstrap.sh into the colcon workspace ros_ws/
BRIDGE_PREFIX = REPO_ROOT / "ros_ws" / "install" / "topic_based_ros2_control"
# The arm server's topics as the hardware side sees them; the launch remaps them there.
HARDWARE_COMMANDS = "/so101/hardware/joint_commands"
HARDWARE_STATES = "/so101/hardware/joint_states"
ARM_CONTROLLER = "arm_controller"
STREAM_CONTROLLER = "arm_forward_controller"
# The launch's ``gripper:=`` choices; the clients drive the forward one.
GRIPPER_CONTROLLERS = {"forward": "gripper_forward_controller", "action": "gripper_action_controller"}
GRIPPER_CONTROLLER = GRIPPER_CONTROLLERS["forward"]
SERVER_NODE = "so101_server_node"
TORQUE_PARAMETER = "gripper_torque_limit_nm"
# How long the hardware side waits, on activation, for the arm server's first
# joint state. The twin's server published it 0.7-0.95 s after its process
# started (scripts/ros2_control_poc.py); the real server also connects the bus.
SERVER_START_S = 10.0
# After that, a joint state this old means the arm server stopped publishing.
# The controller manager then deactivates the controllers, so the last sample
# is not republished as a live /joint_states. The arm clients use this same
# bound for a reply from a controller that has gone.
JOINT_STATE_STALE_S = 2.0
# From the server's first state until every controller is loaded: 0.3 s with
# so101_sim.controller_bringup on the twin, on an idle host (2026-11-25), which
# no lost reply lengthens. ROS's spawner took 1.0 s, and 10 s more for every
# reply that did not arrive. The bound covers a host that loads the pipeline's
# models at the same time.
CONTROLLERS_UP_S = 5.0
# How long a client started with the stack waits for it to come up.
BRING_UP_S = SERVER_START_S + CONTROLLERS_UP_S
# The arm server writes the controllers' setpoints as they come; LeRobot's
# per-tick clamp stays its backstop, at LeRobot's default when the config sets none.
STREAM_CLAMP_DEG = 8.0


def server_transport_cfg(cfg: dict) -> dict:
    """``cfg`` with the per-tick clamp the arm server's bus transport streams under."""
    robot = dict(cfg.get("robot") or {})
    if robot.get("max_relative_target") is None:
        robot["max_relative_target"] = STREAM_CLAMP_DEG
    return {**cfg, "robot": robot}


def bridge_prefix_path() -> str:
    """``AMENT_PREFIX_PATH`` with the bridge first, for the process that loads its plugin."""
    if not BRIDGE_PREFIX.is_dir():
        raise RuntimeError(f"{BRIDGE_PREFIX} is missing: run ./bootstrap.sh (section 4e builds it)")
    prefixes = [str(BRIDGE_PREFIX), *filter(None, os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep))]
    return os.pathsep.join(dict.fromkeys(prefixes))


def child_domain_env(base: dict | None = None) -> dict:
    """Environment for a child process that publishes or subscribes.

    The child reads ``ROS_DOMAIN_ID``. This process's participant may already
    be on a domain that variable no longer names.
    """
    env = dict(os.environ if base is None else base)
    try:
        import rclpy
    except ImportError:
        return env
    if rclpy.ok():
        env["ROS_DOMAIN_ID"] = str(rclpy.get_default_context().get_domain_id())
    return env


def start_controllers(config_path, stack: str, gripper: str = "forward", log=None) -> subprocess.Popen:
    """``launch/so101_control.launch.py`` with this Python, in a process group of
    its own, shut down (SIGINT) when this process exits however it exits."""
    from so101_sim.ros_launch import python_args

    launch = python_args(LAUNCH_FILE, f"config:={config_path}", f"stack:={stack}", f"gripper:={gripper}")
    return subprocess.Popen(
        [sys.executable, "-m", "so101_sim.parent_death", sys.executable, *launch],
        cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True, env=child_domain_env(),
    )


def stop_controllers(proc: subprocess.Popen, budget_s: float = STOP_S) -> None:
    """SIGINT the launch's process group; kill it when it has not exited within ``budget_s``."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise RuntimeError(f"the controllers' launch did not exit within {budget_s:.0f} s of SIGINT") from None


def goal_time_s() -> float:
    """``arm_controller``'s ``constraints.goal_time`` from the controllers file."""
    import yaml

    from so101_sim.description import CONTROLLERS_FILE

    params = yaml.safe_load(CONTROLLERS_FILE.read_text())[ARM_CONTROLLER]["ros__parameters"]
    return float(params["constraints"]["goal_time"])
