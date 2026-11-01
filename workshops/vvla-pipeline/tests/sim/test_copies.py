# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline's copies of the workshop's code still match it.

The pipeline runs on its own and the workshop is a lesson students edit, so
each keeps its own copy of the arm layer, the servo profile planner, the
ros2_control client, the manipulation behaviors, the gesture mapper, the
console's camera fan-out and the stop keywords. Every definition both copies
hold is compared as code: docstrings, and the names each copy uses for the
other's modules (and ``DirectArm`` for the workshop's ``LeRobotArm``), are set
aside. A change to one copy fails here until the other matches it.

The scenarios that run on both stacks (``test_scenarios.py``) check the same
copies by what the arm does.
"""

from __future__ import annotations

import ast
import copy
import difflib
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

ROOT = Path(__file__).resolve().parents[2]
EVERY = "every definition"

# (workshop module, pipeline module): the definitions both copies hold.
MIRRORS = {
    ("common.motion", "vla_pipeline.robot.arm_interface"): (
        "GRIPPER_CLOSED_FLOOR", "GRIPPER_TORQUE_LIMIT", "REST_POSE", "joints_text", "JOINT_LIMITS",
        "_PIPELINE_BACKSTOP", "GRASP_LIMITS", "_LIVE_LIMIT_SCALE", "_extended_limits", "LIVE_MIMIC_LIMITS",
        "_active_limits", "set_active_limits", "clamp_joints", "OVERLOAD_BIT", "MotionTimeout", "settled",
        "ArmClient", "DryRunArm", "BUS_RETRIES", "BUS_RETRY_DELAY_S", "_SPEED_REGISTERS", "_ACCEL_REGISTERS",
        "DEFAULT_CONTROL_HZ", "LeRobotArm", "_EFFORT_EVERY_N_TICKS", "OWNER_REPLY_S", "BusOwnerArm",
        "bus_owner",
    ),
    ("common.motion", "vla_pipeline.robot.sim_arm"): ("sim_world", "SimArmClient"),
    ("common.motion", "vla_pipeline.utils.gesture_map"): ("_LowPass", "OneEuro", "_axis", "MimicMapper"),
    ("common.motion", "vla_pipeline.behaviors.dance"): ("DANCE_MOVES",),
    ("common.motion", "vla_pipeline.behaviors.simple"): ("WAVE_UP", "WAVE_LEFT", "WAVE_RIGHT"),
    ("common.servo_profile", "vla_pipeline.robot.servo_profile"): EVERY,
    ("common.ros2_control_arm", "vla_pipeline.robot.ros2_control_arm"): EVERY,
    ("common.ros_helpers", "vla_pipeline.robot.ros2_control_arm"): ("Spinner",),
    ("common.manipulation", "vla_pipeline.behaviors.manipulation"): EVERY,
    ("common.twin_console", "vla_pipeline.console"): (
        "IDLE_AFTER_S", "FanoutCamera", "ModelTap", "_fit", "_title", "_bar", "PROCESSOR", "provider_name",
    ),
    ("common.intents", "vla_pipeline.llm.intents"): ("STOP_KEYWORDS",),
    ("common.intents", "vla_pipeline.main"): ("_STOP_RE", "is_stop"),
}

# Definitions one copy has and the other does not, and why.
ONLY_IN = {
    "common.motion": {
        "make_arm_backend": "the workshop's backend factory; the pipeline's are make_arm and make_bus_arm",
        "mapper_from_config": "reads the workshop's mimic: section; the pipeline's reads behaviors.gesture_mimic",
    },
    "common.manipulation": {
        "register_manipulation": "fills the workshop's registry; the pipeline registers handlers in main.py",
    },
    "vla_pipeline.robot.ros2_control_arm": {
        "Spinner": "the workshop's is common.ros_helpers.Spinner, compared there",
    },
}

# Each pipeline module, and DirectArm, under the workshop's name.
MODULE_NAMES = {
    "vla_pipeline.robot.arm_interface": "common.motion",
    "vla_pipeline.robot.servo_profile": "common.servo_profile",
    "vla_pipeline.robot.ros2_control_arm": "common.ros2_control_arm",
    "vla_pipeline.behaviors.manipulation": "common.manipulation",
    "vla_pipeline.vision.yolo_pose_npu": "common.imaging",
}
NAMES = {"DirectArm": "LeRobotArm"}
SKIP = {"logger"}


def _path(module: str) -> Path:
    if module.startswith("common."):
        return ROOT / "workshop" / "common" / (module.split(".", 1)[1] + ".py")
    return ROOT / (module.replace(".", "/") + ".py")


def _name(node) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _definitions(module: str) -> dict:
    tree = ast.parse(_path(module).read_text(), filename=str(_path(module)))
    return {name: node for node in tree.body if (name := _name(node)) and name not in SKIP}


class _Normalize(ast.NodeTransformer):
    """Drop docstrings; give the pipeline's names the workshop's."""

    def __init__(self, rename: bool):
        self.names = NAMES if rename else {}
        self.modules = MODULE_NAMES if rename else {}

    def _scope(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        node.name = self.names.get(node.name, node.name)
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.modules.get(node.module, node.module)
        return node


def _normal(node, rename: bool):
    return ast.fix_missing_locations(_Normalize(rename).visit(copy.deepcopy(node)))


def _mirrored(workshop: str, pipeline: str) -> list[str]:
    names = MIRRORS[(workshop, pipeline)]
    if names != EVERY:
        return list(names)
    ours, theirs = _definitions(workshop), _definitions(pipeline)
    theirs = {NAMES.get(n, n) for n in theirs}
    return sorted((set(ours) | theirs) - set(ONLY_IN.get(workshop, {})) - set(ONLY_IN.get(pipeline, {})))


@pytest.mark.parametrize("workshop,pipeline", sorted(MIRRORS), ids=lambda m: m)
def test_the_copies_match(workshop, pipeline):
    ours = _definitions(workshop)
    theirs = {NAMES.get(n, n): node for n, node in _definitions(pipeline).items()}
    missing, differ = [], []
    for name in _mirrored(workshop, pipeline):
        if name not in ours or name not in theirs:
            where = workshop if name not in ours else pipeline
            missing.append(f"{name} is not in {where}")
            continue
        a, b = _normal(ours[name], rename=False), _normal(theirs[name], rename=True)
        if ast.dump(a) != ast.dump(b):
            diff = difflib.unified_diff(
                ast.unparse(a).splitlines(), ast.unparse(b).splitlines(),
                f"{workshop}.{name}", f"{pipeline}.{name}", lineterm="", n=2,
            )
            differ.append("\n".join(diff))
    assert not missing and not differ, "the copies drifted apart:\n" + "\n".join(missing + differ)


@pytest.mark.parametrize(
    "workshop", ["common.motion", "common.servo_profile", "common.ros2_control_arm", "common.manipulation"],
)
def test_every_workshop_definition_has_a_pipeline_copy_or_a_reason(workshop):
    mirrored = set().union(*(_mirrored(w, p) for w, p in MIRRORS if w == workshop))
    unaccounted = set(_definitions(workshop)) - mirrored - set(ONLY_IN.get(workshop, {}))
    assert not unaccounted, (
        f"{workshop} defines {sorted(unaccounted)} with no pipeline copy: copy it to the pipeline "
        "and list it in MIRRORS, or say in ONLY_IN why only the workshop has it"
    )


def test_the_twin_defaults_are_the_stacks_arm():
    """The twin's own rest pose and jaw closed point, for a world built without a stack's."""
    from common.motion import GRIPPER_CLOSED_FLOOR as workshop_floor
    from common.motion import REST_POSE as workshop_rest
    from so101_sim.config import DEFAULTS
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR as pipeline_floor
    from vla_pipeline.robot.arm_interface import REST_POSE as pipeline_rest

    assert DEFAULTS["rest_pose"] == workshop_rest == pipeline_rest
    assert DEFAULTS["gripper_map"]["closed_cal"] == workshop_floor == pipeline_floor
