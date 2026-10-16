# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The notebook, the sim tests, and ``run_pipeline.sh --digital-twin`` must
reset the twin the same way.

The notebook button and the pytest helper are the same three lines:
``world.reset(parked)`` then ``arm.send_joints(parked)``. The pipeline
console, when the arm owns the world, calls that through
``TwinConsole._restore_twin``. ``run_pipeline.sh`` does not: ``use_ros2``
makes the orchestrator a ``Ros2ArmClient``, so the console calls
``reset_sim()`` and ``So101ServerNode._on_reset_scene`` resets the world the
control loop is driving.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim

_AWAY = {
    "shoulder_pan": 40.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 40.0,
    "wrist_flex": 10.0,
    "wrist_roll": 10.0,
    "gripper": 40.0,
}


def _notebook_reset(arm) -> None:
    """The reset button in ``02_digital_twin.ipynb``, and the pytest helper."""
    from common.motion import restore_twin

    arm.world  # the notebook's world and the arm's world are the same object
    state = getattr(arm, "shared_state", None)
    if isinstance(state, dict):
        state.pop("grip_lock", None)
        state.pop("grip_lock_pos", None)
    restore_twin(arm)


def _service_reset(arm) -> None:
    """``So101ServerNode._on_reset_scene``, then the control loop's next send."""
    from vla_pipeline.robot.robot_node import So101ServerNode

    node = SimpleNamespace(
        _arm=arm,
        _target_lock=threading.Lock(),
        _target={},
        _last_cmd_time=0.0,
        _last_sent_target={"stale": 1.0},
        _target_gen=0,
    )
    response = SimpleNamespace(success=False, message="")
    So101ServerNode._on_reset_scene(node, None, response)
    assert response.success is True
    with node._target_lock:
        target = dict(node._target)
        fresh = (time.time() - node._last_cmd_time) < 1.0
        sent = node._last_sent_target
    if fresh and target != sent:
        arm.send_joints(target)


def _pipeline_console_reset(server_arm) -> dict:
    """``run_pipeline.sh``: the console owns no world, so reset is the ROS service."""
    from vla_pipeline.console import TwinConsole

    state = {"grip_lock": True, "grip_lock_pos": 40.0}
    client = SimpleNamespace(
        world=None,
        shared_state=state,
        reset_sim=lambda: _service_reset(server_arm) or True,
    )
    ctx = SimpleNamespace(
        arm=client,
        stop_event=threading.Event(),
        shared_state=state,
        say=lambda _message: None,
    )
    box = SimpleNamespace(ctx=ctx)
    box._restore_twin = lambda: TwinConsole._restore_twin(box)
    TwinConsole.reset_scene(box)
    return state


def _arm(stack: str, cfg: dict):
    from so101_sim.world import SimWorld

    if stack == "workshop":
        from common.motion import REST_POSE, SimArmClient
    else:
        from vla_pipeline.robot.sim_arm import REST_POSE, SimArmClient

    sim = dict(cfg.get("sim") or {})
    world = SimWorld({"sim": sim}, rest_pose=REST_POSE, start=False)
    arm = SimArmClient(cfg, world)
    arm.shared_state = {"grip_lock": True, "grip_lock_pos": 40.0}
    return arm


def _disturb(arm) -> None:
    arm.world.place("toothbrush", (0.35, 0.15, 0.25), (0.5, 0.5, 0.5, 0.5))
    arm.send_joints(_AWAY)
    arm.world.bus.write("Torque_Limit", "gripper", 240)


def _snap(arm) -> dict:
    world = arm.world
    with world.lock:
        props = {
            name: world.data.qpos[qadr : qadr + 7].copy()
            for name, (qadr, _dadr) in world._free.items()
        }
    return {
        "joints": arm.world.joint_cal(),
        "goal": arm.world.goal_cal(),
        "props": props,
        "torque": arm.world.bus.read("Torque_Limit", "gripper"),
    }


def _differ(a: dict, b: dict) -> list[str]:
    lines = []
    for name in a["joints"]:
        if abs(a["joints"][name] - b["joints"][name]) > 0.05:
            lines.append(
                f"joint {name}: {a['joints'][name]:.1f} vs {b['joints'][name]:.1f}"
            )
        if abs(a["goal"][name] - b["goal"][name]) > 0.05:
            lines.append(
                f"goal {name}: {a['goal'][name]:.1f} vs {b['goal'][name]:.1f}"
            )
    for name in a["props"]:
        if not (abs(a["props"][name] - b["props"][name]) < 1e-4).all():
            lines.append(f"prop {name} qpos differs")
    if a["torque"] != b["torque"]:
        lines.append(f"gripper Torque_Limit {a['torque']} vs {b['torque']}")
    return lines


def test_run_pipeline_reset_calls_the_service_not_the_local_world():
    """``use_ros2`` arm has no world, so the console reset is the ROS service."""
    from vla_pipeline.console import TwinConsole

    calls = []
    arm = SimpleNamespace(world=None, reset_sim=lambda: calls.append("service") or True)
    assert TwinConsole._restore_twin(SimpleNamespace(ctx=SimpleNamespace(arm=arm))) is True
    assert calls == ["service"]


def test_notebook_test_and_pipeline_resets_leave_the_same_twin(workshop_cfg, pipeline_cfg):
    """One disturbed twin, reset the way each entry point resets it."""
    from common.motion import REST_POSE as workshop_pose
    from vla_pipeline.robot.arm_interface import REST_POSE as pipeline_pose

    assert workshop_pose is pipeline_pose
    assert workshop_pose["wrist_flex"] == 22.5
    assert workshop_pose["gripper"] == 38.0
    made = []
    try:
        notebook = _arm("workshop", workshop_cfg)
        in_process = _arm("pipeline", pipeline_cfg)
        server = _arm("pipeline", pipeline_cfg)
        made.extend((notebook, in_process, server))
        for arm in made:
            _disturb(arm)
        _notebook_reset(notebook)
        # The in-process branch is what the console does when the arm owns the
        # world. run_pipeline.sh does not take it.
        from vla_pipeline.console import TwinConsole

        assert TwinConsole._restore_twin(SimpleNamespace(ctx=SimpleNamespace(arm=in_process)))
        client_state = _pipeline_console_reset(server)
        notebook_snap = _snap(notebook)
        diffs = []
        for label, other in (
            ("pipeline in-process", _snap(in_process)),
            ("run_pipeline service", _snap(server)),
        ):
            found = _differ(notebook_snap, other)
            diffs.extend(f"{label}: {line}" for line in found)
        if client_state.get("grip_lock") is not None:
            diffs.append("run_pipeline client grip_lock was not cleared")
        assert not diffs, "reset paths diverged\n" + "\n".join(diffs)
        assert "grip_lock" not in notebook.shared_state
    finally:
        for arm in made:
            arm.world.close()
