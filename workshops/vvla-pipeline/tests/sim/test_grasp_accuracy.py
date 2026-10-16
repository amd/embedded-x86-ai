# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The scan view the pickup keeps, then the top-down grasp.

The sweep stops at the first pan whose box solves. The hand must rise clear
before it moves over the object and come down centred on it. Beside a lying
object it then moves sideways until the fixed finger is ``fixed_jaw_gap_m``
away; around a standing one it closes with the same small gap to each inner
face. The jaw holds a little past where contact stopped it, not at the floor.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.sim

_CASES = [
    ("pipeline", "sports_ball", "sports ball"),
    ("pipeline", "bottle", "bottle"),
    ("workshop", "sports_ball", "sports ball"),
    ("workshop", "bottle", "bottle"),
]


def _stack(stack: str):
    if stack == "pipeline":
        from vla_pipeline.behaviors.grip import close_with_feedback
        from vla_pipeline.behaviors.pick_place import _box_usable, _plan_from_view, _shape_of
        from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR
        from vla_pipeline.robot.sim_arm import SimArmClient

        def plan(cfg, view):
            return _plan_from_view(cfg, view)

        def pp(cfg):
            return cfg["behaviors"]["pick_place"]

        def grip(cfg):
            return cfg["behaviors"]["grip"]

        return plan, pp, _box_usable, _shape_of, SimArmClient, close_with_feedback, grip, GRIPPER_CLOSED_FLOOR
    from common.manipulation import _box_usable, _plan_from_view, _shape_of, close_with_feedback
    from common.motion import GRIPPER_CLOSED_FLOOR, SimArmClient

    def plan(cfg, view):
        return _plan_from_view(SimpleNamespace(cfg=cfg), view)

    def pp(cfg):
        return cfg["pick_place"]

    def grip(cfg):
        return cfg["grip"]

    return plan, pp, _box_usable, _shape_of, SimArmClient, close_with_feedback, grip, GRIPPER_CLOSED_FLOOR


def _surface_gap(m, d, geom: int, obj: int) -> float:
    """Gap (m) from the triangles of mesh ``geom`` to the sphere or cylinder ``obj``.

    ``mj_geomDistance`` measures a mesh by its convex hull, and the hull of
    the wrist part spans the jaw opening a lying object sits in.
    """
    import mujoco

    mesh = int(m.geom_dataid[geom])
    v0, f0 = m.mesh_vertadr[mesh], m.mesh_faceadr[mesh]
    verts = m.mesh_vert[v0:v0 + m.mesh_vertnum[mesh]] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
    verts = (verts - d.geom_xpos[obj]) @ d.geom_xmat[obj].reshape(3, 3)
    tris = verts[m.mesh_face[f0:f0 + m.mesh_facenum[mesh]]]
    radius, half = float(m.geom_size[obj][0]), float(m.geom_size[obj][1])
    cylinder = m.geom_type[obj] == mujoco.mjtGeom.mjGEOM_CYLINDER

    def sdf(p):
        if not cylinder:
            return np.linalg.norm(p, axis=-1) - radius
        q = np.stack([np.hypot(p[..., 0], p[..., 1]) - radius, np.abs(p[..., 2]) - half], -1)
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)

    best = float(sdf(tris).min())
    if best > 0.005:
        return best
    edge = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=-1).max(-1)
    near = sdf(tris.mean(1)) - edge < best
    n = min(60, int(np.ceil(edge[near].max() / 0.0005)))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    keep = i + j <= n
    w = np.stack([i[keep], j[keep], n - i[keep] - j[keep]], -1) / n
    return min(best, float(sdf(np.einsum("pk,tkd->tpd", w, tris[near])).min()))


def _gaps(world, pose, body: str) -> tuple[float, float]:
    """Minimum gap (m) from the visible finger's surface, and from any colliding
    jaw, to ``body``."""
    import mujoco

    m = world.model
    d = world.fk(pose)
    gid = m.geom(f"{body}_collision").id
    visual = 1e9
    for i in range(m.ngeom):
        owner = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if not m.geom_contype[i] and owner == "gripper":
            visual = min(visual, _surface_gap(m, d, i, gid))
    return visual, _collision_gap(world, pose, body)


def _collision_gap(world, pose, body: str) -> float:
    """Minimum gap (m) from any colliding jaw geom to ``body``."""
    import mujoco

    m = world.model
    d = world.fk(pose)
    gid = m.geom(f"{body}_collision").id
    gap = 1e9
    for i in range(m.ngeom):
        owner = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if m.geom_contype[i] and owner in ("gripper", "moving_jaw_so101_v1"):
            gap = min(gap, float(mujoco.mj_geomDistance(m, d, i, gid, 0.25, None)))
    return gap


def _face_gaps(world, pose, centre, half: float) -> tuple[float, float]:
    """Gap (m) from the object surface to the fixed pad and to the moving pad."""
    from so101_sim.world import _FIXED_PAD_X

    d = world.fk(pose)
    b = d.body("gripper")
    local = b.xmat.reshape(3, 3).T @ (np.asarray(centre, float) - b.xpos)
    moving = b.xmat.reshape(3, 3).T @ (d.geom("moving_jaw_box3").xpos - b.xpos)
    gap_fixed = float(local[0] - half - _FIXED_PAD_X)
    gap_moving = float(moving[0] - (local[0] + half))
    return gap_fixed, gap_moving


def _accepted(twin, cfg, stack: str, body: str, resolved: str, quat=(1.0, 0.0, 0.0, 0.0)):
    """The first scan pan whose box the pickup would grasp, and where the body really is."""
    from so101_sim.camera import SimCamera
    from so101_sim.detect import SegDetector
    from so101_sim.kinematics import joint_line, shared_kinematics

    plan_from, pp_of, usable, shape_of, *_, floor = _stack(stack)
    world = twin.world
    pp = pp_of(cfg)
    shape = shape_of(resolved)
    scan = dict(pp["scan_pose"])
    truth = np.array(world.model.body(body).pos, float)
    kin = shared_kinematics(floor)
    cam = SimCamera(world, "arm", renderer=twin.renderer, segmentation=True)
    try:
        for pan in pp["scan_pans"]:
            world.reset({**scan, "shoulder_pan": float(pan)})
            for other in world.objects:
                if other != body:
                    world.hide(other)
            world.place(body, truth, quat)
            time.sleep(0.2)
            ok, frame = cam.read(fresh=True)
            if not ok or frame is None:
                continue
            dets = [d for d in SegDetector(cam).detect(frame) if d[3] == body]
            if not dets:
                continue
            box = dets[0][2]
            height, width = frame.shape[:2]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area / float(height * width) < float(pp["min_box_area_frac"]):
                continue
            if not usable(box, shape, width, height):
                continue
            joints = dict(world.joint_cal())
            det = SimpleNamespace(box=box, area=area, score=dets[0][1], name=resolved)
            plan = plan_from(cfg, {
                "frame": frame, "det": det, "resolved": resolved, "joints": joints,
            })
            if plan is None:
                continue
            pre = {**plan.poses[0], "gripper": 75.0}
            here = {**joints, "gripper": 75.0}
            above = joint_line(here, pre)
            down = [{**pose, "gripper": 75.0} for pose in plan.poses[1:]]
            if plan.beside is not None:
                down += joint_line(plan.poses[-1], plan.beside)
            return plan, [*above, *down], joints, truth, float(pan), kin
    finally:
        cam.release()
    return None, None, None, truth, None, kin


@pytest.mark.parametrize("stack,body,resolved", _CASES)
def test_approach_clears_and_arrives_at_the_grasp(twins, pipeline_cfg, workshop_cfg, stack, body, resolved):
    from so101_sim.kinematics import OPEN_GRIPPER

    cfg = pipeline_cfg if stack == "pipeline" else workshop_cfg
    twin = twins(stack, cfg)
    plan, path, _joints, truth, pan, _kin = _accepted(twin, cfg, stack, body, resolved)
    assert plan is not None, f"{stack} found no graspable view of the {body}"
    err = float(np.linalg.norm((plan.xyz - truth)[:2]))
    assert err < 0.005, (
        f"{stack} {body} at pan {pan:.0f}: centre {np.round(plan.xyz, 4)} is {err * 1000:.1f} mm "
        f"from {np.round(truth, 4)}"
    )
    twin.world.place(body, truth)
    closed = [pose for pose in path if abs(float(pose["gripper"]) - OPEN_GRIPPER) > 0.5]
    assert not closed, f"{stack} {body}: the jaw closes before the hand is at the grasp"
    visual_min = collision_min = 1e9
    for pose in path:
        visual, collision = _gaps(twin.world, pose, body)
        visual_min = min(visual_min, visual)
        collision_min = min(collision_min, collision)
    assert visual_min >= 0.0, (
        f"{stack} {body}: the visible finger enters the object by {-visual_min * 1000:.1f} mm on the way in"
    )
    assert collision_min > 0.0, (
        f"{stack} {body}: a jaw collides with the object {-collision_min * 1000:.1f} mm before the close"
    )

    top = twin.world.grasp_point(plan.poses[0], 0.04)
    for pose in plan.poses:
        point = twin.world.grasp_point(pose, 0.04)
        lateral = float(np.linalg.norm(point[:2] - top[:2]))
        assert lateral <= 0.005, f"{stack} {body}: the descent wanders {lateral * 1000:.1f} mm off the centre"
        off_object = float(np.linalg.norm(point[:2] - truth[:2]))
        assert off_object <= 0.008, (
            f"{stack} {body}: the descent is {off_object * 1000:.1f} mm from the object"
        )

    half = float(twin.world.model.geom(f"{body}_collision").size[0])
    gap_fixed, gap_moving = _face_gaps(twin.world, plan.grasp, truth, half)
    assert gap_fixed > 0.001 and gap_moving > 0.001, (
        f"{stack} {body}: a face is already on the object (fixed {gap_fixed * 1000:.1f} mm, "
        f"moving {gap_moving * 1000:.1f} mm)"
    )
    if body == "bottle":
        assert abs(gap_fixed - gap_moving) <= 0.004, (
            f"{stack} {body}: the gaps differ by {abs(gap_fixed - gap_moving) * 1000:.1f} mm "
            f"(fixed {gap_fixed * 1000:.1f}, moving {gap_moving * 1000:.1f})"
        )
    else:
        want = float(_stack(stack)[1](cfg)["fixed_jaw_gap_m"])
        assert abs(gap_fixed - want) <= 0.002 and gap_moving > gap_fixed, (
            f"{stack} {body}: the fixed finger is {gap_fixed * 1000:.1f} mm from the object, "
            f"want {want * 1000:.1f} (moving {gap_moving * 1000:.1f} mm)"
        )


@pytest.mark.parametrize("stack", ["pipeline", "workshop"])
@pytest.mark.parametrize("lying", ["straight", "turned"])
def test_toothbrush_descent_clears_it(twins, pipeline_cfg, workshop_cfg, stack, lying):
    """The jaws turn across the toothbrush and come down centred over it, then
    the hand moves sideways until the fixed finger is beside it.

    ``straight`` is how the lesson lays the brush; ``turned`` is its pose in
    the scene file, which reset restores.
    """
    from so101_sim.kinematics import joint_line
    from so101_sim.world import _FIXED_FINGER

    cfg = pipeline_cfg if stack == "pipeline" else workshop_cfg
    twin = twins(stack, cfg)
    m = twin.world.model
    quat = (1.0, 0.0, 0.0, 0.0) if lying == "straight" else tuple(float(v) for v in m.body("toothbrush").quat)
    plan, path, _joints, truth, pan, _kin = _accepted(twin, cfg, stack, "toothbrush", "toothbrush", quat)
    assert plan is not None, f"{stack} found no graspable view of the {lying} toothbrush"
    twin.world.place("toothbrush", truth, quat)
    sideways = len(joint_line(plan.poses[-1], plan.beside)) if plan.beside is not None else 0
    down = min(_collision_gap(twin.world, pose, "toothbrush") for pose in path[: len(path) - sideways])
    assert down > 0.0, (
        f"{stack} {lying} toothbrush at pan {pan:.0f}: a jaw lands on it "
        f"({-down * 1000:.1f} mm into it) on the way down"
    )
    assert plan.beside is not None, f"{stack} {lying} toothbrush: no sideways move to the fixed finger"

    import mujoco

    d = twin.world.fk(plan.beside)
    brush = m.geom("toothbrush_collision").id
    fixed = min(
        float(mujoco.mj_geomDistance(m, d, i, brush, 0.25, None))
        for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i]))) in _FIXED_FINGER
    )
    moving = min(
        float(mujoco.mj_geomDistance(m, d, i, brush, 0.25, None))
        for i in range(m.ngeom)
        if m.geom_contype[i] and m.geom_bodyid[i] == m.body("moving_jaw_so101_v1").id
    )
    assert fixed > -0.001 and moving > fixed, (
        f"{stack} {lying} toothbrush: beside it the fixed finger is {fixed * 1000:+.1f} mm and the "
        f"moving finger {moving * 1000:+.1f} mm from it"
    )


def _hold_fns(stack: str):
    if stack == "pipeline":
        from vla_pipeline.behaviors.grip import ease_grip, holding
    else:
        from common.manipulation import ease_grip, holding
    return holding, ease_grip


@pytest.mark.parametrize("stack,body,resolved", _CASES)
def test_close_holds_the_object(twins, pipeline_cfg, workshop_cfg, stack, body, resolved):
    """The close runs after the hand is down, both jaws keep the object up to
    the pre-grasp without it moving in the gripper, the grip eases to the
    hold limit before the overload protection trips, and ``holding()`` tells
    a held object from empty jaws."""
    from rigs import LIFT_SLIP_TOL_M, Trace

    cfg = pipeline_cfg if stack == "pipeline" else workshop_cfg
    twin = twins(stack, cfg)
    plan, _approach, _joints, truth, _pan, _kin = _accepted(twin, cfg, stack, body, resolved)
    assert plan is not None, f"{stack} found no graspable view of the {body}"
    *_, Arm, close, grip_of, floor = _stack(stack)
    world = twin.world
    world.reset(plan.grasp)
    for other in world.objects:
        if other != body:
            world.hide(other)
    world.place(body, truth)
    arm = Arm(cfg, world=world)
    try:
        gripped, hold = close(arm, grip_of(cfg), start_from=75.0, ease=False)
        assert gripped, f"{stack} reported no contact on the {body}"
        time.sleep(0.3)
        assert world.is_grasped(body), f"{stack} {body} is not held by both jaws"
        moved = float(np.linalg.norm(world.body_pos(body)[:2] - truth[:2]))
        assert moved < 0.02, f"{stack} {body} was shoved {moved * 1000:.0f} mm across the table"
        if body == "bottle":
            _w, x, y, _z = (float(v) for v in world.body_quat(body))
            up_z = 1.0 - 2.0 * (x * x + y * y)
            assert up_z > 0.95, f"{stack} bottle tipped (up_z {up_z:.3f})"
        assert hold > floor + 5.0, f"{stack} {body}: hold {hold:.1f} is the closed floor {floor:.0f}"

        holding, ease_grip = _hold_fns(stack)
        z0 = float(world.body_pos(body)[2])
        trace = Trace(world, time.monotonic, watch=(body,))
        trace.start()
        t0 = time.monotonic()
        arm.go_to({**plan.poses[0], "gripper": hold})
        ease_grip(arm, grip_of(cfg))
        time.sleep(2.0)
        trace.stop()
        assert trace.first_overload() is None, f"{stack}: the gripper overload protection tripped"
        rise = float(world.body_pos(body)[2]) - z0
        assert rise > 0.03, f"{stack} {body} rose only {rise * 1000:.0f} mm"
        assert world.is_grasped(body), f"{stack} {body} is not held by both fingers after the lift"
        moved = trace.drift(body, t0)
        assert moved <= LIFT_SLIP_TOL_M, (
            f"{stack} {body} moved {moved * 1000:.1f} mm in the gripper on the way up"
        )
        assert holding(arm, grip_of(cfg), hold), f"{stack}: holding() lost the lifted {body}"
        world.hide(body)
        time.sleep(0.5)
        assert not holding(arm, grip_of(cfg), hold), f"{stack}: holding() with empty jaws"
    finally:
        world.reset()
