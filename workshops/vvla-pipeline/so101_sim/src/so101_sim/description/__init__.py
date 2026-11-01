# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The SO-101 robot description for ros2_control.

``so101.urdf.xacro`` holds the geometry and the hardware side
(``topic_based_ros2_control``); ``controllers.yaml`` the controllers. The
joint limits are the host stack's ``JOINT_LIMITS``, turned into radians by the
twin's joint map: LeRobot's homing pose is the URDF and MJCF zero, and the
gripper's calibrated span maps linearly onto the jaw angle.
"""

from __future__ import annotations

from pathlib import Path

DESCRIPTION_DIR = Path(__file__).resolve().parent
XACRO_FILE = DESCRIPTION_DIR / "so101.urdf.xacro"
CONTROLLERS_FILE = DESCRIPTION_DIR / "controllers.yaml"


def robot_description(
    cfg: dict | None,
    joint_limits: dict,
    *,
    joint_commands_topic: str,
    joint_states_topic: str,
    initial_state_timeout: float,
) -> str:
    """The URDF ``robot_state_publisher`` and ``ros2_control_node`` load.

    ``joint_limits`` maps every motor to ``(lower, upper)`` in calibrated
    units, as ``JOINT_LIMITS`` does. ``cfg`` is the host config: its ``sim:``
    section holds the joint map, the servos' top speed and their stall torque.
    ``initial_state_timeout`` bounds how long the hardware side waits, on
    activation, for the arm server's first joint state. After that, a joint
    state older than :data:`so101_sim.control.JOINT_STATE_STALE_S` is rejected.
    """
    import xacro

    from so101_sim.config import load_sim_config
    from so101_sim.control import JOINT_STATE_STALE_S
    from so101_sim.mapping import JointMap

    sim = load_sim_config(cfg)
    jmap = JointMap(sim["joint_map"], sim["gripper_map"])
    top_speed = float(sim["servo"]["max_speed_raw"])
    args = {
        "joint_commands_topic": joint_commands_topic,
        "joint_states_topic": joint_states_topic,
        "effort": repr(float(sim["current_model"]["stall_torque_nm"])),
        "initial_state_timeout": repr(float(initial_state_timeout)),
        "joint_state_timeout": repr(float(JOINT_STATE_STALE_S)),
        # The hardware side republishes the commands while the summed
        # |state - command| exceeds this: one encoder step on every joint.
        "trigger_joint_command_threshold": repr(sum(jmap.steps_to_rad(name, 1.0) for name in joint_limits)),
    }
    for name, (lower, upper) in joint_limits.items():
        a, b = jmap.to_rad(name, lower), jmap.to_rad(name, upper)
        args[f"{name}_lower"] = repr(min(a, b))
        args[f"{name}_upper"] = repr(max(a, b))
        args[f"{name}_velocity"] = repr(jmap.steps_to_rad(name, top_speed))
    return xacro.process_file(str(XACRO_FILE), mappings=args).toxml()
