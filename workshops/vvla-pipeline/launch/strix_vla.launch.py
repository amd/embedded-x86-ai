# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: both camera nodes, the SO-101 server node and its controllers.

The repo is run from source (uv venv), not installed as an ament package, so
nodes are launched as ``python -m`` processes - same modules you run by hand
for component tests. Per-camera parameters (device, resolution, rotation,
topic) come from ``config/pipeline.yaml`` via each node's ``--config``. The
arm server (or the digital twin, which hosts it) runs under ros2_control:
``launch/so101_control.launch.py`` is included, and the server's topics are
remapped onto the hardware side's ``/so101/hardware/*``.

    ros2 launch launch/strix_vla.launch.py
    ros2 launch launch/strix_vla.launch.py dry_run:=true     # no motors
    ros2 launch launch/strix_vla.launch.py digital_twin:=true  # MuJoCo twin, no arm
    ros2 launch launch/strix_vla.launch.py config:=/path/to/pipeline.yaml
    ros2 launch launch/strix_vla.launch.py cameras:=false    # the arm only

Run it with the venv's Python, after sourcing ROS
(``.venv/bin/python $(which ros2) launch ...``).

The orchestrator (``python -m vla_pipeline.main``) is intentionally NOT part
of this launch: it wants a TTY + preview windows. Start it in its own
terminal once the nodes are up.
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # the venv's python (launch must run inside .venv)


def _arm(context):
    """The bus owner - the twin or the arm server - on the hardware topics, and its controllers."""
    if LaunchConfiguration("arm").perform(context) != "true":
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, LAUNCH_FILE
    from vla_pipeline.utils.config import load_config

    config = LaunchConfiguration("config").perform(context)
    dry_run = LaunchConfiguration("dry_run").perform(context) == "true"
    twin = LaunchConfiguration("digital_twin").perform(context) == "true"
    robot = load_config(config)["robot"]
    remaps = [
        "--ros-args",
        "-r", f"{robot['command_topic']}:={HARDWARE_COMMANDS}",
        "-r", f"{robot['state_topic']}:={HARDWARE_STATES}",
    ]
    if twin:
        # The twin hosts the arm server and renders the arm camera (they share the physics).
        cmd = [PY, "-m", "so101_sim.node", "--config", config, "--stack", "pipeline", "--sim-cameras", "arm"]
        name = "so101_sim"
    else:
        cmd = [PY, "-m", "vla_pipeline.robot.robot_node", "--server", *(["--dry-run"] if dry_run else []),
               "--config", config]
        name = "so101_server_dry" if dry_run else "so101_server"
    return [
        ExecuteProcess(cmd=[*cmd, *remaps], cwd=str(REPO_ROOT), output="screen", name=name),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(LAUNCH_FILE)),
            launch_arguments={"config": config, "stack": "pipeline"}.items(),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Build the LaunchDescription for the two camera nodes, the SO-101 server node and its controllers."""
    config = LaunchConfiguration("config")
    sim = LaunchConfiguration("digital_twin")
    cameras = LaunchConfiguration("cameras")
    arm_camera = PythonExpression(["'", cameras, "' == 'true' and '", sim, "' != 'true'"])

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
            DeclareLaunchArgument(
                "arm",
                default_value="true",
                description="run the bus owner and its controllers (robot.use_ros2)",
            ),
            DeclareLaunchArgument(
                "cameras",
                default_value="true",
                description="run the camera nodes (cameras.<role>.use_ros2)",
            ),
            # Mount camera (faces the human) - /cameras/mount/image_raw
            ExecuteProcess(
                cmd=[
                    PY,
                    "-m",
                    "vla_pipeline.vision.camera_node",
                    "--role",
                    "mount",
                    "--config",
                    config,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
                name="camera_mount",
                condition=IfCondition(cameras),
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
                condition=IfCondition(arm_camera),
            ),
            # SO-101 bus owner (the twin with digital_twin:=true) and ros2_control
            OpaqueFunction(function=_arm),
        ]
    )
