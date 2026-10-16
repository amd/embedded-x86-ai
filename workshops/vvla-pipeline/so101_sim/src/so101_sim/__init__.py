# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MuJoCo digital twin of the SO-101 arm used by the VVLA pipeline + workshop.

The twin is driven exactly like the real rig: the arm speaks the LeRobot
``SO101Follower`` / Feetech-bus API (calibrated degrees, 0-100 gripper,
``Goal_Position`` / ``Goal_Velocity`` / ``Acceleration`` writes,
``Present_Position`` / ``Present_Current`` / ``Present_Load`` reads), the
wrist camera is a MuJoCo camera on the gripper, and the gripper current
comes from the simulated actuator torque.

Typical use::

    from so101_sim import shared_world, SimArm, SimCamera
    world = shared_world()            # one physics world per process
    arm = SimArm(world)               # LeRobot-style arm on that world
    cam = SimCamera(world, "arm")     # 640x480 BGR wrist camera
    arm.send_joints({"gripper": 90})
    ok, frame = cam.read()

Importing this package does not import MuJoCo; the GL backend is chosen
on first use (see :mod:`so101_sim.gl`).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "SimWorld",
    "SimArm",
    "SimBus",
    "SimFollower",
    "SimCamera",
    "RenderService",
    "SimRecorder",
    "SegDetector",
    "Kinematics",
    "shared_world",
    "close_shared_world",
    "load_sim_config",
    "MOTOR_NAMES",
]

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_LAZY = {
    "SimWorld": "so101_sim.world",
    "shared_world": "so101_sim.world",
    "close_shared_world": "so101_sim.world",
    "SimArm": "so101_sim.arm",
    "SimBus": "so101_sim.arm",
    "SimFollower": "so101_sim.arm",
    "SimCamera": "so101_sim.camera",
    "RenderService": "so101_sim.render",
    "SimRecorder": "so101_sim.record",
    "SegDetector": "so101_sim.detect",
    "Kinematics": "so101_sim.kinematics",
    "load_sim_config": "so101_sim.config",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'so101_sim' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod), name)
