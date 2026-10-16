# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The shared look-then-move pickup, without a full scene.

A settled frame is stored with the joints that took it. A clipped box loses
to a smaller one the table locate will accept. One empty frame does not end
the sweep. The descent stays inside 5 mm of the vertical. A slipped grasp
locates the object again before the second solve.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

pytestmark = pytest.mark.sim

MODULES = ["common.manipulation"]
POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 40.0,
    "wrist_flex": 20.0,
    "wrist_roll": 0.0,
    "gripper": 70.0,
}
NEAR = {**POSE, "shoulder_lift": -57.3, "elbow_flex": 63.2, "wrist_flex": 67.5}


class _Det:
    def __init__(self, box, score=0.9):
        self.box = tuple(float(v) for v in box)
        self.score = score
        x1, y1, x2, y2 = self.box
        self.area = max(0.0, x2 - x1) * max(0.0, y2 - y1)


class _Arm:
    def __init__(self):
        self.pose = dict(POSE)
        self.returned_at = []

    def go_to(self, target, duration_s=0.0, stop_check=None, limits=None):
        self.pose.update(target)
        self.returned_at.append(time.time())

    def move_to(self, target, stop_check=None, limits=None, **_):
        self.pose.update(target)

    def get_joints(self):
        return dict(self.pose)

    def send_joints(self, target):
        self.pose.update(target)

    def set_gripper_torque_limit(self, value):
        pass

    def get_gripper_load(self):
        return None

    def grip_report(self):
        return None

    def go_to_rest(self):
        self.pose = dict(POSE)


class _Camera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.seen = []

    def read_after(self, settled_at):
        self.seen.append(settled_at)
        return True, self.frames.pop(0)

    def ok(self):
        return True


def _frame(kind: int) -> np.ndarray:
    image = np.zeros((100, 100, 3), np.uint8)
    image[0, 0, 0] = kind
    return image


def _find(frame, name):
    kind = int(frame[0, 0, 0])
    if kind == 0:
        return [], "sports ball"
    if kind == 2:
        # The larger box is cut by the left edge. The smaller one is usable.
        return [_Det((0, 5, 90, 90), 0.99), _Det((20, 20, 45, 45), 0.8)], "sports ball"
    return [_Det((20, 20, 60, 60))], "sports ball"


def _pp():
    return {
        "min_box_area_frac": 0.002,
        "scan_pose": dict(POSE),
        "scan_pans": [0.0, -14.0, 39.0],
        "scan_near_pose": dict(NEAR),
        "scan_near_pans": [0.0, -14.0],
        "ambiguous_ratio": 2.0,
        "reach": "ik",
        "retries": 1,
        "wrist_align_classes": ["toothbrush"],
    }


def _mod(name):
    return importlib.import_module(name)


@pytest.mark.parametrize("module", MODULES)
def test_frame_is_captured_after_the_move_and_paired_with_those_joints(module, caplog):
    mod = _mod(module)
    arm, camera = _Arm(), _Camera([_frame(0), _frame(1), _frame(1)])
    views, status, _, _ = mod._sweep_views(
        arm, camera, _find, "ball", dict(POSE), [0.0, -14.0, 39.0], _pp(), None,
    )

    assert status == "ok"
    assert len(arm.returned_at) == 2
    assert [view["joints"]["shoulder_pan"] for view in views] == [-14.0]
    assert [view["settled_at"] for view in views] == camera.seen[1:]
    assert camera.seen[1] >= arm.returned_at[1]
    assert "scan sighting: sports ball" in caplog.text
    assert "joints pan -14.0" in caplog.text


@pytest.mark.parametrize("module", MODULES)
def test_clipped_larger_box_loses_to_a_valid_smaller_one(module):
    mod = _mod(module)
    arm, camera = _Arm(), _Camera([_frame(2), _frame(1)])
    views, status, _, _ = mod._sweep_views(
        arm, camera, _find, "ball", dict(POSE), [0.0, -14.0], _pp(), None,
    )

    assert status == "ok"
    assert len(arm.returned_at) == 1
    ranked = mod._rank_views(views)
    assert ranked[0]["det"].box == (20.0, 20.0, 45.0, 45.0)
    assert all(view["det"].box[0] > 1 for view in views)
    assert not any(view["det"].box == (0.0, 5.0, 90.0, 90.0) for view in views)


@pytest.mark.parametrize("module", MODULES)
def test_ungraspable_sighting_does_not_end_the_sweep(module, caplog):
    mod = _mod(module)
    arm, camera = _Arm(), _Camera([_frame(1), _frame(1)])
    tried = []

    def accept(batch):
        tried.append(batch[0]["joints"]["shoulder_pan"])
        return len(tried) == 2

    views, status, _, _ = mod._sweep_views(
        arm, camera, _find, "ball", dict(POSE), [0.0, -14.0, 39.0], _pp(), None, accept=accept,
    )

    assert status == "ok"
    assert tried == [0.0, -14.0]
    assert [view["joints"]["shoulder_pan"] for view in views] == [-14.0]
    assert "not graspable at pan 0.0" in caplog.text


@pytest.mark.parametrize("module", MODULES)
def test_one_empty_frame_does_not_end_the_sweep(module):
    mod = _mod(module)
    arm, camera = _Arm(), _Camera([_frame(0), _frame(1), _frame(0)])
    views, status, _, _ = mod._sweep_views(
        arm, camera, _find, "ball", dict(POSE), [0.0, -14.0, 39.0], _pp(), None,
    )

    assert status == "ok"
    assert len(arm.returned_at) == 2
    assert len(views) == 1
    assert views[0]["order"] == 1
    assert views[0]["joints"]["shoulder_pan"] == -14.0


@pytest.mark.parametrize("module", MODULES)
def test_near_sweep_runs_only_when_the_high_sweep_stored_nothing(module, caplog):
    mod = _mod(module)
    pp = _pp()
    empty = _Camera([_frame(0), _frame(0), _frame(0), _frame(1), _frame(0)])
    arm = _Arm()
    views, status, _, _ = mod._search_views(arm, empty, _find, "ball", pp, None, empty)
    assert status == "ok" and views
    assert "scan near pan 0.0" in caplog.text
    assert any(abs(item["joints"]["shoulder_lift"] - NEAR["shoulder_lift"]) < 1e-6 for item in views)

    caplog.clear()
    hit = _Camera([_frame(1), _frame(0), _frame(0)])
    views, status, _, _ = mod._search_views(arm, hit, _find, "ball", pp, None, hit)
    assert status == "ok" and views
    assert "scan near pan" not in caplog.text


def test_ball_descent_stays_within_5_mm_of_the_vertical():
    from so101_sim.kinematics import joint_line, shared_kinematics

    kin = shared_kinematics(43)
    solved = kin.solve_grasp(
        np.array([0.28, 0.06, 0.015]), above_m=0.06, width=0.03, table_z=-0.005, wrist_roll=0.0,
    )
    assert solved is not None
    pre, grasp = solved
    for module in MODULES:
        mod = _mod(module)
        path = mod._descent_poses(kin, pre, grasp, 0.03, -0.005)
        assert path == joint_line(pre, grasp)
        assert mod._path_within(kin, pre, path, 0.03, -0.005)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("second_xy,second_pan", [
    ((0.28, 0.08), 40.0),  # the object moved
    ((0.204, 0.0), 11.0),  # it did not: the first plan failed, so the fresh one still runs
], ids=["moved", "unmoved"])
def test_a_slipped_grasp_grasps_from_the_fresh_fix(module, monkeypatch, second_xy, second_pan):
    mod = _mod(module)
    frames = [_frame(1), _frame(1)]
    arm, camera = _Arm(), _Camera(frames)
    events = []
    plans = []

    def plan_from_view(owner, view):
        events.append("plan")
        shifted = len(plans) > 0
        pan = second_pan if shifted else 10.0
        xyz = np.array([*second_xy, 0.01] if shifted else [0.20, 0.00, 0.01])
        pose = {**POSE, "shoulder_pan": pan, "shoulder_lift": -20.0 if shifted else -10.0}
        made = mod._GraspPlan([dict(pose), {**pose, "shoulder_lift": pose["shoulder_lift"] - 8.0}], xyz)
        plans.append(made)
        return made

    def close(arm, g, stop_check=None, start_from=None, ease=True):
        assert not ease, "the pickup eased the grip while the object was on the table"
        events.append("close")
        return (len(plans) >= 2, 40.0)

    monkeypatch.setattr(mod, "_plan_from_view", plan_from_view)
    monkeypatch.setattr(mod, "close_with_feedback", close)
    monkeypatch.setattr(mod, "holding", lambda arm, g, hold: len(plans) >= 2)
    monkeypatch.setattr(mod, "_offer", lambda *args, **kwargs: "offered")

    if module.startswith("common"):
        held_arm, held_camera = arm, camera

        class _Finder:
            find = staticmethod(_find)

        class _Ctx:
            cfg = {"pick_place": _pp(), "grip": {}}
            arm = held_arm
            arm_camera = held_camera
            camera = held_camera
            shared_state = {}
            detector = _Finder()

            def say(self, message):
                events.append(message)

            def stop_check(self):
                return False

        class _Cmd:
            object_name = "ball"

        assert mod.run_pick_place(_Ctx(), _Cmd())
    else:
        cfg = {"behaviors": {"pick_place": _pp(), "grip": {}}, "cameras": {"arm": {}}}

        class _Detector:
            find = staticmethod(_find)

        assert mod.run_pick_place(
            cfg, arm, _Detector(), None, camera, None, "ball", {}, None,
            lambda message: events.append(message),
        )

    assert events.index("plan") < events.index("The grasp slipped - trying once more.")
    assert events.index("The grasp slipped - trying once more.") < events.index("plan", events.index("plan") + 1)
    assert len(plans) == 2
    assert plans[1].xyz[0] == pytest.approx(second_xy[0])
    assert arm.get_joints()["shoulder_pan"] == pytest.approx(second_pan)
