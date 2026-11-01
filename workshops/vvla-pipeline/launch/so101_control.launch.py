# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: ros2_control for the SO-101 over the arm server's topics.

robot_state_publisher publishes the description, ``ros2_control_node`` runs
the controllers at ``robot.control_fps``, and ``topic_based_ros2_control``
talks to the arm server (or the twin) on ``/so101/hardware/joint_commands`` and
``/so101/hardware/joint_states``, in radians; the controller manager finds it
in ``ros_ws/`` (bootstrap.sh builds it). Run it with the venv's Python, after
sourcing ROS:

    .venv/bin/python $(which ros2) launch launch/so101_control.launch.py
    ... launch/so101_control.launch.py gripper:=action
    ... launch/so101_control.launch.py config:=workshop/solution/config/workshop.yaml stack:=workshop
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO_ROOT = Path(__file__).resolve().parents[1]


def _joint_limits(stack: str) -> dict:
    """The host stack's ``JOINT_LIMITS``, from its own copy of the arm layer."""
    root = REPO_ROOT / "workshop" / "solution" if stack == "workshop" else REPO_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if stack == "workshop":
        from common.motion import JOINT_LIMITS
    else:
        from vla_pipeline.robot.arm_interface import JOINT_LIMITS
    return JOINT_LIMITS


def _nodes(context):
    import yaml

    from so101_sim.control import (
        ARM_CONTROLLER,
        GRIPPER_CONTROLLERS,
        HARDWARE_COMMANDS,
        HARDWARE_STATES,
        SERVER_START_S,
        STREAM_CONTROLLER,
        bridge_prefix_path,
    )
    from so101_sim.description import CONTROLLERS_FILE, robot_description

    config = LaunchConfiguration("config").perform(context)
    stack = LaunchConfiguration("stack").perform(context)
    gripper = LaunchConfiguration("gripper").perform(context)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    urdf = robot_description(
        cfg, _joint_limits(stack), joint_commands_topic=HARDWARE_COMMANDS, joint_states_topic=HARDWARE_STATES,
        initial_state_timeout=SERVER_START_S,
    )
    controllers = str(CONTROLLERS_FILE)
    spawn = ["--controller-manager", "/controller_manager", "--param-file", controllers]
    inactive = [STREAM_CONTROLLER, *(name for key, name in GRIPPER_CONTROLLERS.items() if key != gripper)]
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": urdf}],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[controllers, {"update_rate": int(round(float(cfg["robot"]["control_fps"])))}],
            remappings=[("~/robot_description", "/robot_description")],
            additional_env={"AMENT_PREFIX_PATH": bridge_prefix_path()},
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", ARM_CONTROLLER, GRIPPER_CONTROLLERS[gripper], *spawn],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[*inactive, "--inactive", *spawn],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """robot_state_publisher, the controller manager, and its controllers."""
    from so101_sim.control import GRIPPER_CONTROLLERS

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(REPO_ROOT / "config" / "pipeline.yaml"),
                description="pipeline.yaml or workshop.yaml",
            ),
            DeclareLaunchArgument(
                "stack",
                default_value="pipeline",
                choices=["pipeline", "workshop"],
                description="whose JOINT_LIMITS the description uses",
            ),
            DeclareLaunchArgument(
                "gripper",
                default_value="forward",
                choices=sorted(GRIPPER_CONTROLLERS),
                description="the active gripper controller; the other is loaded inactive, as is the streaming arm one",
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
