# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``so101_sim.kinematics``: locating objects from arm-camera boxes, grasp IK,
and the hand-eye solve behind ``scripts/calibrate_arm_camera.py``.

Localization runs on the twin: the arm looks from ``scan_pose`` (panned so the
object sits between the jaws), the box comes from the twin's segmentation, and the
result is compared with where the object really is.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytestmark = pytest.mark.sim

BALL_SPOTS = [(0.28, 0.06), (0.22, -0.04), (0.32, 0.0), (0.25, 0.10)]


@pytest.fixture(scope="module")
def kin():
    from so101_sim.kinematics import Kinematics

    return Kinematics()


@pytest.mark.parametrize("angle", [-70.0, -35.0, 0.0, 28.0, 72.0])
def test_principal_axis_in_box(angle):
    from so101_sim.kinematics import axis_error_deg, principal_axis_in_box

    frame = np.full((240, 320, 3), [115, 145, 175], np.uint8)
    yy, xx = np.indices(frame.shape[:2])
    a = math.radians(angle)
    along = (xx - 160) * math.cos(a) + (yy - 120) * math.sin(a)
    across = -(xx - 160) * math.sin(a) + (yy - 120) * math.cos(a)
    mask = (np.abs(along) <= 65) & (np.abs(across) <= 8)
    frame[mask] = [220, 120, 25]
    ys, xs = np.nonzero(mask)
    box = (xs.min() - 5, ys.min() - 5, xs.max() + 6, ys.max() + 6)

    axis = principal_axis_in_box(frame, box)

    assert axis is not None
    assert abs(axis_error_deg(axis["angle_deg"], angle)) < 1.0
    assert axis["elongation"] > 4.0


def test_principal_axis_rejects_square_or_clipped_crop():
    from so101_sim.kinematics import principal_axis_in_box

    frame = np.full((120, 160, 3), 100, np.uint8)
    frame[40:80, 60:100] = 220
    assert principal_axis_in_box(frame, (55, 35, 105, 85)) is None
    assert principal_axis_in_box(frame, (0, 35, 105, 85)) is None


def _aimed_scan_pose(kin, cfg, p) -> dict:
    """scan_pose with the pan that puts ``p`` at the image x between the jaws."""
    pp = cfg["behaviors"]["pick_place"]
    jaws_x = -0.24  # normalized image x of an object sitting between the jaws
    best = None
    for pan in np.arange(-60.0, 40.0, 0.5):
        pose = {**pp["scan_pose"], "shoulder_pan": float(pan)}
        u = kin.project(p, pose)[0, 0]
        err = abs(u / 640 * 2 - 1 - jaws_x)
        if best is None or err < best[0]:
            best = (err, pose)
    return best[1]


def _look(twin, obj, pos, pose):
    """Arm at ``pose``, only ``obj`` on the table at ``pos``: ``(class, box, (w, h), true centre)``."""
    from so101_sim.camera import SimCamera
    from so101_sim.detect import SegDetector

    world = twin.world
    world.reset(pose)
    for o in world.objects:
        if o != obj:
            world.hide(o)
    world.place(obj, pos)
    cam = SimCamera(world, "arm", renderer=twin.renderer, segmentation=True)
    try:
        time.sleep(0.3)  # let the object settle on the table
        ok, frame = cam.read(fresh=True)
        assert ok
        dets = [d for d in SegDetector(cam).detect(frame) if d[3] == obj]
        assert dets, f"{obj} not in view"
        return dets[0][0], dets[0][2], frame.shape[1::-1], world.body_pos(obj)
    finally:
        cam.release()


@pytest.mark.parametrize(
    "obj,xy",
    [("sports_ball", s) for s in BALL_SPOTS]
    + [("bottle", (0.2, 0.2)), ("book", (0.22, -0.15)), ("toothbrush", (0.12, -0.24))],
)
def test_locate_on_table(twins, pipeline_cfg, kin, obj, xy):
    from so101_sim.kinematics import FLAT_CLASSES, ROUND_CLASSES

    twin = twins("pipeline", pipeline_cfg)
    pos = np.array([xy[0], xy[1], twin.world.model.body(obj).pos[2]])
    pose = _aimed_scan_pose(kin, pipeline_cfg, pos)
    cls, box, size, truth = _look(twin, obj, pos, pose)

    shape = "sphere" if cls in ROUND_CLASSES else "flat" if cls in FLAT_CLASSES else "bottle" if cls == "bottle" else "upright"
    loc = kin.locate_on_table(box, pose, image_size=size, shape=shape)
    assert loc is not None, f"no fix for box {box}"
    err = loc["xyz"] - truth
    xy_tol = 0.025 if shape == "flat" else 0.005 if shape == "bottle" else 0.01
    assert np.linalg.norm(err[:2]) < xy_tol, (
        f"{obj} at {np.round(truth, 3)} located at {np.round(loc['xyz'], 3)}"
    )
    if obj == "sports_ball":
        assert abs(err[2]) < 0.01
        assert abs(loc["width"] - 0.04) < 0.005


def test_box_cut_by_the_image_side_gives_no_fix(kin):
    assert kin.locate_on_table((0, 100, 80, 200), {}, image_size=(640, 480)) is None
    assert kin.locate_on_table((200, 400, 300, 480), {}, image_size=(640, 480)) is None


@pytest.mark.parametrize("r,bearing", [(0.16, 0), (0.20, 25), (0.24, -20), (0.28, 10), (0.30, -5)])
def test_ik_round_trip(kin, r, bearing):
    sx, sy = kin.world.fk({}).body("shoulder").xpos[:2]
    p = np.array([sx + r * math.cos(math.radians(bearing)), sy + r * math.sin(math.radians(bearing)), 0.03])
    plan = kin.solve_grasp(p, above_m=0.06, width=0.04)
    assert plan is not None, f"{np.round(p, 3)} should be in reach"
    pre, grasp = plan
    w = kin.world
    assert np.linalg.norm(w.grasp_point(grasp, 0.04) - p) < 0.003
    assert np.linalg.norm(w.grasp_point(pre, 0.04) - (p + [0, 0, 0.06])) < 0.003
    assert abs(w.jaw_axis(grasp)[2]) < 0.1, "closing axis not horizontal at the grasp"
    # at the edge of reach the wrist limit tilts the pre-grasp a little
    assert abs(w.jaw_axis(pre)[2]) <= 0.3
    for pose in (pre, grasp):
        assert kin.lowest_point(pose) > -0.005, "arm below the table"


def test_descent_waypoints_stay_on_the_vertical(kin):
    p = np.array([0.22, -0.04, 0.021])
    pre, grasp = kin.solve_grasp(p, above_m=0.06, width=0.04)
    path = kin.waypoints(pre, grasp, 0.04)
    assert len(path) >= 4 and path[-1] == grasp
    top = kin.world.grasp_point(pre, 0.04)
    for pose in path:
        q = kin.world.grasp_point(pose, 0.04)
        assert np.linalg.norm(q[:2] - top[:2]) < 0.003, "descent leaves the vertical line"


@pytest.mark.parametrize("yaw", [20.0, 60.0, 112.9, 160.0])
def test_ik_closes_across_the_axis(kin, yaw):
    from common.motion import GRASP_LIMITS

    p = np.array([0.12, -0.24, 0.011])
    axis = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
    pose, err, _ = kin.ik(p, 0.024, wrist_roll=0.0, limits=GRASP_LIMITS, object_axis=axis)
    lo, hi = GRASP_LIMITS["wrist_roll"]
    assert lo <= pose["wrist_roll"] <= hi, "no roll inside the arm's wrist range"
    assert err < 0.003
    assert abs(float(kin.world.jaw_axis(pose) @ axis)) < 0.02, "jaws don't close across the axis"


def test_analytic_grasp_jacobian_matches_forward_difference(kin):
    q = np.array([23.0, -41.0, 62.0, 37.0, -54.0])
    target = np.array([0.22, -0.08, 0.025])
    reference = q.copy()
    residual, jac, _ = kin._grasp_eval(q, target, 0.032, 75.0, [0.4, 0.9165, 0.0], reference)
    for j in range(5):
        dq = np.zeros(5)
        dq[j] = 1e-3
        moved, _, _ = kin._grasp_eval(q + dq, target, 0.032, 75.0, [0.4, 0.9165, 0.0], reference)
        numeric = (moved[:5] - residual[:5]) / dq[j]
        assert np.allclose(jac[:5, j], numeric, atol=1e-6)


def _goal_list(kin, joints, box, size, shape):
    """Locate, solve, then the joint-space goals the behavior sends."""
    from so101_sim.kinematics import joint_line

    from common.motion import GRASP_LIMITS

    loc = kin.locate_on_table(box, joints, image_size=size, shape=shape)
    assert loc is not None, f"no fix for {shape} box {box}"
    p = loc["xyz"].copy()
    if shape == "flat":
        p[2] = max(p[2], -0.005 + 0.011)
    obstacle = None
    limits = None
    if shape == "bottle":
        obstacle = (*loc["xyz"][:2], 0.5 * loc["width"], -0.005 + loc["height"])
        limits = GRASP_LIMITS
    elif shape == "flat":
        limits = GRASP_LIMITS
    plan = kin.solve_grasp(
        p, above_m=0.06, width=loc["width"], table_z=-0.005, limits=limits, obstacle=obstacle,
    )
    assert plan is not None, f"no grasp of {shape} at {np.round(p, 3)}"
    pre, grasp = plan
    keys = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    goals = [*joint_line(joints, pre), *joint_line(pre, grasp)]
    return [[pose[k] for k in keys] for pose in goals]


@pytest.mark.parametrize(
    "obj,xy,shape",
    [
        ("sports_ball", (0.28, 0.06), "sphere"),
        ("bottle", (0.2, 0.2), "bottle"),
        ("toothbrush", (0.12, -0.24), "flat"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_grasp_path_is_deterministic(twins, pipeline_cfg, kin, obj, xy, shape):
    """Same joints and the same box: the same goals. A two-pixel shift keeps the elbow."""
    twin = twins("pipeline", pipeline_cfg)
    pos = np.array([xy[0], xy[1], twin.world.model.body(obj).pos[2]])
    joints = _aimed_scan_pose(kin, pipeline_cfg, pos)
    _cls, box, size, _truth = _look(twin, obj, pos, joints)
    first = _goal_list(kin, joints, box, size, shape)
    second = _goal_list(kin, joints, box, size, shape)
    assert first == second
    shifted = (box[0] + 2, box[1], box[2] + 2, box[3])
    other = _goal_list(kin, joints, shifted, size, shape)
    assert (first[-1][2] > 0) == (other[-1][2] > 0)


def test_oriented_sqp_is_deterministic_and_does_not_flip_roll_branch(kin):
    from common.motion import GRASP_LIMITS

    rng = np.random.default_rng(17)
    for _ in range(12):
        p = np.array([rng.uniform(0.16, 0.28), rng.uniform(-0.16, 0.14), 0.011])
        yaw = rng.uniform(0.0, math.pi)
        axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        a = kin.solve_grasp(p, width=0.024, wrist_roll=0.0, object_axis=axis, limits=GRASP_LIMITS)
        b = kin.solve_grasp(p, width=0.024, wrist_roll=0.0, object_axis=axis, limits=GRASP_LIMITS)
        assert (a is None) == (b is None)
        if a is None:
            continue
        for pa, pb in zip(a, b):
            assert np.allclose([pa[k] for k in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                "wrist_flex", "wrist_roll")],
                               [pb[k] for k in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                "wrist_flex", "wrist_roll")], atol=1e-8)
            assert abs(float(kin.world.jaw_axis(pa) @ axis)) < 0.02
        assert abs(a[0]["wrist_roll"] - a[1]["wrist_roll"]) < 5.0


def test_ik_out_of_reach(kin):
    assert kin.solve_grasp([0.50, 0.0, 0.02]) is None
    assert kin.solve_grasp([0.25, 0.0, -0.05]) is None  # below the table


def test_mount_config_round_trip(kin):
    from so101_sim.kinematics import matrix_to_rpy

    R0, t0 = kin.default_mount
    R, t = kin.mount({"pos": t0.tolist(), "rpy": matrix_to_rpy(R0)})
    assert np.allclose(R, R0, atol=1e-9) and np.allclose(t, t0)
    assert kin.mount({"pos": None, "rpy": None}) is kin.default_mount


def test_default_camera_model_projects_like_the_twin(twins, pipeline_cfg, kin):
    twin = twins("pipeline", pipeline_cfg)
    pos = np.array([0.28, 0.06, 0.015])
    pose = _aimed_scan_pose(kin, pipeline_cfg, pos)
    _, box, _, truth = _look(twin, "sports_ball", pos, pose)
    u, v = kin.project(truth, pose)[0]
    assert abs(u - 0.5 * (box[0] + box[2])) < 5 and abs(v - 0.5 * (box[1] + box[3])) < 5


def test_hand_eye_recovers_a_known_mount(kin):
    from so101_sim.kinematics import rpy_to_matrix, solve_mount

    rng = np.random.default_rng(0)
    R0, t0 = kin.default_mount
    R_true = R0 @ rpy_to_matrix([3.0, -2.0, 4.0])
    t_true = t0 + [0.004, -0.003, 0.005]
    board_R, board_t = rpy_to_matrix([0.0, 0.0, 90.0]), np.array([0.25, -0.04, -0.005])
    grip, board = [], []
    for _ in range(10):
        joints = {"shoulder_pan": rng.uniform(-10, 10), "shoulder_lift": rng.uniform(-62, -40),
                  "elbow_flex": rng.uniform(50, 80), "wrist_flex": rng.uniform(30, 65),
                  "wrist_roll": rng.uniform(-25, 25)}
        Rg, tg = kin.gripper_pose(joints)
        Rc, tc = Rg @ R_true, tg + Rg @ t_true
        # board in camera, with ~0.3 mm / 0.1 deg of PnP noise
        Rb = rpy_to_matrix(rng.normal(0, 0.1, 3)) @ Rc.T @ board_R
        tb = Rc.T @ (board_t - tc) + rng.normal(0, 3e-4, 3)
        grip.append((Rg, tg))
        board.append((Rb, tb))
    R, t, spread = solve_mount(grip, board)
    angle = math.degrees(math.acos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1)))
    assert np.linalg.norm(t - t_true) < 0.002
    assert angle < 0.5
    assert spread < 0.003
