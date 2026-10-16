# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: both camera nodes + the SO-101 server node.

The repo is run from source (uv venv), not installed as an ament package, so
nodes are launched as ``python -m`` processes - same modules you run by hand
for component tests. Per-camera parameters (device, resolution, rotation,
topic) come from ``config/pipeline.yaml`` via each node's ``--config``.

    ros2 launch launch/strix_vla.launch.py
    ros2 launch launch/strix_vla.launch.py dry_run:=true     # no motors
    ros2 launch launch/strix_vla.launch.py digital_twin:=true  # MuJoCo twin, no arm
    ros2 launch launch/strix_vla.launch.py config:=/path/to/pipeline.yaml

The orchestrator (``python -m vla_pipeline.main``) is intentionally NOT part
of this launch: it wants a TTY + preview windows. Start it in its own
terminal once the nodes are up.
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # the venv's python (launch must run inside .venv)


def generate_launch_description() -> LaunchDescription:
    """Build the LaunchDescription for the two camera nodes and the SO-101 server node."""
    config = LaunchConfiguration("config")
    dry_run = LaunchConfiguration("dry_run")
    sim = LaunchConfiguration("digital_twin")
    real_arm = PythonExpression(
        ["'", dry_run, "' != 'true' and '", sim, "' != 'true'"]
    )
    dry_arm = PythonExpression(["'", dry_run, "' == 'true' and '", sim, "' != 'true'"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(REPO_ROOT / "config" / "pipeline.yaml"),
                description="pipeline YAML with cameras/robot parameters",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="run the arm server without motors",
            ),
            DeclareLaunchArgument(
                "digital_twin",
                default_value="false",
                description="MuJoCo digital twin replaces the arm server + arm camera",
            ),
            # Operator camera (faces the human) - /cameras/operator/image_raw
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "vla_pipeline.vision.camera_node",
                    "--role",
                    "operator",
                    "--config",
                    config,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="camera_operator",
            ),
            # Arm camera (end effector) - /cameras/arm/image_raw
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "vla_pipeline.vision.camera_node",
                    "--role",
                    "arm",
                    "--config",
                    config,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="camera_arm",
                condition=UnlessCondition(sim),
            ),
            # Digital twin: arm server + arm camera rendered on the gripper (one
            # process - they share the physics) + /sim/view/image_raw
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "so101_sim.node",
                    "--config",
                    config,
                    "--stack",
                    "pipeline",
                    "--sim-cameras",
                    "arm",
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="so101_sim",
                condition=IfCondition(sim),
            ),
            # SO-101 bus owner - /so101/joint_command, /so101/joint_state
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "vla_pipeline.robot.robot_node",
                    "--server",
                    "--config",
                    config,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="so101_server",
                condition=IfCondition(real_arm),
            ),
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "vla_pipeline.robot.robot_node",
                    "--server",
                    "--dry-run",
                    "--config",
                    config,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="so101_server_dry",
                condition=IfCondition(dry_arm),
            ),
        ]
    )
