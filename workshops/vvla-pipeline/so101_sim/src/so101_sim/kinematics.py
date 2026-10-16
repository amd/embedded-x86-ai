# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Where is the object, and which joint angles reach it - for the real arm and the twin.

The SO-101 MJCF gives forward kinematics in calibrated degrees (the same
units the LeRobot follower reports), so this module is not twin-only:

* :meth:`Kinematics.camera_pose` - the wrist camera's pose from the joint
  angles: FK to the gripper, then the camera's mount on the gripper.
* :meth:`Kinematics.locate_on_table` - an object detector box turned into a
  3D position: the camera rays through the box's pixels are intersected with
  the table plane (objects rest on the table, so one camera is enough).
* :meth:`Kinematics.solve_grasp` - bounded five-joint SQP for a top-down
  grasp, using MuJoCo's analytic Jacobians: the point between the jaws on
  the object, the closing axis horizontal, every arm part above the table.
  An elongated-object axis also constrains the jaws to close across it.

Camera frames follow OpenCV (x right, y down, z forward); a mount is the
camera frame in the ``gripper`` body frame - what ``cv2.calibrateHandEye``
returns for an eye-in-hand camera. Until the real camera is calibrated the
twin's ``wrist_cam`` (pose and intrinsics) is the default.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from so101_sim.world import _FIXED_PAD_X, TABLE_Z

__all__ = [
    "Kinematics",
    "TABLE_Z",
    "ROUND_CLASSES",
    "FLAT_CLASSES",
    "principal_axis_in_box",
    "axis_error_deg",
    "joint_line",
    "rpy_to_matrix",
    "matrix_to_rpy",
    "solve_mount",
    "shared_kinematics",
]

IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
IK5_JOINTS = (*IK_JOINTS, "wrist_roll")
OPEN_GRIPPER = 75.0
# Detector classes with geometry that needs a different table-box model.
ROUND_CLASSES = frozenset({"sports ball", "orange", "apple"})
FLAT_CLASSES = frozenset({"book", "toothbrush"})

_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras look down -z with y up
_CLEARANCE_BODIES = ("lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")
_MIN_HOLD_M = 0.015  # a tall object's grasp point stays at least this far below its top
_AXIS_WEIGHT = 0.05  # metres of grasp-point error worth one unit of closing-axis tilt
_SEEDS = ((9.0, 20.0, 54.0), (-15.0, 20.0, 54.0), (30.0, -20.0, 75.0), (-40.0, 60.0, 60.0))
_SQP_TRUST_DEG = 10.0
_SQP_REG = 2e-6
_BOX_GRID = np.array(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3)).reshape(3, -1).T  # points through a geom's box
_RING = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
_golden = math.pi * (3.0 - math.sqrt(5.0))
_SPHERE = np.array([
    [math.cos(_golden * i) * math.sqrt(1 - z * z), math.sin(_golden * i) * math.sqrt(1 - z * z), z]
    for i, z in enumerate(np.linspace(-1, 1, 600))
])


def axis_error_deg(angle: float, target: float) -> float:
    """Shortest signed error between two unoriented image axes (degrees)."""
    return (float(angle) - float(target) + 90.0) % 180.0 - 90.0


def principal_axis_in_box(
    frame,
    box,
    *,
    colour_threshold: float = 35.0,
    min_pixels: int = 40,
    min_elongation: float = 1.6,
) -> dict | None:
    """Long image axis of an elongated object inside an axis-aligned box.

    YOLO supplies only ``xyxy``.  The object's silhouette is approximated by
    pixels that differ from the crop's border colour, then PCA supplies an
    unoriented angle in ``[-90, 90)``.  ``None`` is returned when the box is
    clipped, too small, or not convincingly elongated.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    x1f, y1f, x2f, y2f = (float(v) for v in box)
    if x1f <= 1 or y1f <= 1 or x2f >= width - 1 or y2f >= height - 1:
        return None
    x1, y1 = max(0, int(math.floor(x1f))), max(0, int(math.floor(y1f)))
    x2, y2 = min(width, int(math.ceil(x2f))), min(height, int(math.ceil(y2f)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2, :3].astype(float)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0)
    background = np.median(border, axis=0)
    ys, xs = np.nonzero(np.linalg.norm(crop - background, axis=2) >= float(colour_threshold))
    if len(xs) < int(min_pixels):
        return None
    pixels = np.column_stack([xs, ys]).astype(float)
    centre = pixels.mean(axis=0)
    _, singular, axes = np.linalg.svd(pixels - centre, full_matrices=False)
    if len(singular) < 2 or singular[1] <= 1e-9:
        return None
    elongation = float(singular[0] / singular[1])
    if elongation < float(min_elongation):
        return None
    direction = axes[0]
    along = (pixels - centre) @ direction
    lo, hi = np.percentile(along, [5, 95])
    span = float(hi - lo)
    end_band = 0.2 * span
    low_count = int((along <= lo + end_band).sum())
    high_count = int((along >= hi - end_band).sum())
    if high_count > low_count:
        direction = -direction
        along = -along
        lo, hi = -hi, -lo
    centre = centre + direction * 0.5 * (lo + hi)
    angle = axis_error_deg(math.degrees(math.atan2(direction[1], direction[0])), 0.0)
    return {
        "angle_deg": angle,
        "elongation": elongation,
        "center": (float(centre[0] + x1), float(centre[1] + y1)),
        "direction": (float(direction[0]), float(direction[1])),
        "length_px": span,
        "pixels": int(len(xs)),
    }


def rpy_to_matrix(rpy_deg) -> np.ndarray:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` from ``[roll, pitch, yaw]`` in degrees."""
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R) -> list[float]:
    """Inverse of :func:`rpy_to_matrix` (degrees)."""
    R = np.asarray(R, float)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fold yaw into roll
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


class Kinematics:
    """FK, camera rays and grasp IK on the SO-101 model (no physics thread)."""

    def __init__(self, world=None, sim_cfg: dict | None = None, camera: str = "wrist_cam"):
        if world is None:
            from so101_sim.world import SimWorld

            world = SimWorld(sim_cfg, start=False)
        self.world = world
        self.model = world.model
        self._lock = threading.RLock()
        import mujoco

        m = self.model
        cid = m.camera(camera).id
        R_body_cam = np.zeros(9)
        mujoco.mju_quat2Mat(R_body_cam, m.cam_quat[cid])
        R_body_cam = R_body_cam.reshape(3, 3) @ _CV_FROM_MJ
        # camera body pose in the gripper frame (any rest pose will do)
        d = world.fk({})
        Rg, tg = self._body(d, "gripper")
        Rb, tb = self._body(d, m.body(m.cam_bodyid[cid]).name)
        R_g_body, t_g_body = Rg.T @ Rb, Rg.T @ (tb - tg)
        self.default_mount = (R_g_body @ R_body_cam, t_g_body + R_g_body @ m.cam_pos[cid])
        self._sensor = (m.cam_sensorsize[cid].copy(), m.cam_intrinsic[cid].copy(), m.cam_resolution[cid].copy())
        self._limits = {}
        for i, name in enumerate(IK_JOINTS):  # the first four of MOTOR_NAMES
            lo, hi = (world.jmap.to_cal(name, v) for v in world._jrange[i])
            self._limits[name] = (min(lo, hi), max(lo, hi))
        lo, hi = (world.jmap.to_cal("wrist_roll", v) for v in world._jrange[4])
        self._limits["wrist_roll"] = (min(lo, hi), max(lo, hi))
        self._dofs = np.array([m.joint(name).dofadr[0] for name in IK5_JOINTS], int)
        self._bottle = self._bottle_silhouette()
        # MuJoCo Jacobians are per radian. Public poses are calibrated degrees.
        self._rad_per_cal = np.array(
            [world.jmap.to_rad(name, 1.0) - world.jmap.to_rad(name, 0.0) for name in IK5_JOINTS]
        )
        self._clear_geoms = [
            g for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and m.body(m.geom_bodyid[g]).name in _CLEARANCE_BODIES
        ]

    # ------------------------------------------------------------------
    # frames
    # ------------------------------------------------------------------
    @staticmethod
    def _body(d, name: str):
        b = d.body(name)
        return b.xmat.reshape(3, 3).copy(), b.xpos.copy()

    def gripper_pose(self, joints: dict):
        """``(R, t)`` of the ``gripper`` body in the world (base) frame."""
        with self._lock:
            return self._body(self.world.fk(dict(joints)), "gripper")

    def camera_pose(self, joints: dict, mount=None):
        """``(R, t)`` world-from-camera (OpenCV axes) at ``joints``.

        ``mount`` is ``(R, t)`` of the camera in the gripper frame, a config
        dict ``{pos, rpy}``, or ``None`` for the twin's ``wrist_cam``.
        """
        Rm, tm = self.mount(mount)
        Rg, tg = self.gripper_pose(joints)
        return Rg @ Rm, tg + Rg @ tm

    def mount(self, mount=None):
        """Normalise a mount spec to ``(R, t)`` (see :meth:`camera_pose`)."""
        if mount is None:
            return self.default_mount
        if isinstance(mount, dict):
            if mount.get("pos") is None or mount.get("rpy") is None:
                return self.default_mount
            return rpy_to_matrix(mount["rpy"]), np.asarray(mount["pos"], float)
        R, t = mount
        return np.asarray(R, float), np.asarray(t, float).reshape(3)

    def default_intrinsics(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """3x3 camera matrix of the twin's ``wrist_cam`` at ``width`` x ``height``."""
        sensor, intr, res = self._sensor
        W, H = int(width or res[0]), int(height or res[1])
        fx, fy = intr[0] / sensor[0] * W, intr[1] / sensor[1] * H
        cx, cy = W / 2.0 + intr[2] / sensor[0] * W, H / 2.0 + intr[3] / sensor[1] * H
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def intrinsics(self, spec: dict | None, width: int, height: int):
        """``(K, dist)`` from a ``cameras.arm.intrinsics`` dict (or the twin's).

        A calibration taken at another resolution is scaled to the frame.
        """
        if not spec or spec.get("fx") is None:
            return self.default_intrinsics(width, height), None
        sx = width / float(spec.get("width") or width)
        sy = height / float(spec.get("height") or height)
        K = np.array([[spec["fx"] * sx, 0.0, spec["cx"] * sx],
                      [0.0, spec["fy"] * sy, spec["cy"] * sy],
                      [0.0, 0.0, 1.0]])
        dist = spec.get("dist")
        return K, (np.asarray(dist, float) if dist is not None and any(dist) else None)

    def project(self, points, joints: dict, K=None, mount=None) -> np.ndarray:
        """World points -> pixels (no distortion); for tests and calibration checks."""
        K = self.default_intrinsics() if K is None else np.asarray(K, float)
        R, t = self.camera_pose(joints, mount)
        pc = (np.atleast_2d(points) - t) @ R
        uv = pc[:, :2] / pc[:, 2:3]
        return uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

    # ------------------------------------------------------------------
    # localization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized(pixels, K, dist=None) -> np.ndarray:
        """Pixels -> undistorted normalized image coordinates ((N, 2))."""
        px = np.asarray(pixels, float).reshape(-1, 2)
        if dist is not None:
            import cv2

            return cv2.undistortPoints(px.reshape(-1, 1, 2), K, np.asarray(dist, float)).reshape(-1, 2)
        return (px - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]

    def rays(self, pixels, joints: dict, K=None, dist=None, mount=None):
        """Camera centre and unit world directions through ``pixels`` ((N, 2))."""
        K = self.default_intrinsics() if K is None else np.asarray(K, float)
        xy = self._normalized(pixels, K, dist)
        R, t = self.camera_pose(joints, mount)
        d = np.column_stack([xy, np.ones(len(xy))]) @ R.T
        return t, d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def _surface(shape: str, params, table_z: float) -> np.ndarray:
        """Sample points whose projection outlines the object."""
        if shape == "sphere":
            cx, cy, r = params
            return np.array([cx, cy, table_z + r]) + r * _SPHERE
        cx, cy, R, H = params
        ring = np.column_stack([R * _RING, np.zeros(len(_RING))]) + [cx, cy, table_z]
        return np.vstack([ring + [0.0, 0.0, f * H] for f in np.linspace(0.0, 1.0, 9)])

    def _bottle_silhouette(self):
        """Visual-mesh samples of the resting bottle, in the body frame.

        Returns ``(points, height of the body origin above the table)``, or
        ``(None, 0)`` when this model has no bottle. The points are a fixed
        stride of the mesh, so the fit below does not depend on iteration order.
        """
        import mujoco

        m = self.model
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_visual")
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        if gid < 0 or bid < 0 or int(m.geom_dataid[gid]) < 0:
            return None, 0.0
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[gid])
        rot = rot.reshape(3, 3)
        mesh = int(m.geom_dataid[gid])
        adr = int(m.mesh_vertadr[mesh])
        count = int(m.mesh_vertnum[mesh])
        verts = np.asarray(m.mesh_vert[adr:adr + count], float)[::8]
        return verts @ rot.T + np.asarray(m.geom_pos[gid], float), float(m.body_pos[bid][2]) - TABLE_Z

    def _fit_bottle_centre(self, seed_xy, target, R, t, table_z, focal):
        """Bottle centre that puts the visual mesh on the box, or ``None``.

        ``target`` is the box's left, right, and bottom in normalized image
        coordinates. A residual above 4 px means the mesh does not explain
        this box, and the caller keeps the cylinder step.
        """
        pts, above = self._bottle
        if pts is None:
            return None
        origin_z = float(table_z) + float(above)

        def edges(xy):
            pc = (pts + (float(xy[0]), float(xy[1]), origin_z) - t) @ R
            pc = pc[pc[:, 2] > 1e-3]
            if len(pc) < 8:
                return None
            uv = pc[:, :2] / pc[:, 2:3]
            return np.array([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].max()])

        xy = np.asarray(seed_xy, float).copy()
        for _ in range(8):
            edge = edges(xy)
            if edge is None:
                return None
            residual = edge - np.asarray(target, float)
            jac = np.empty((3, 2))
            for j in range(2):
                step = np.zeros(2)
                step[j] = 1e-4
                moved = edges(xy + step)
                if moved is None:
                    return None
                jac[:, j] = (moved - edge) / 1e-4
            try:
                delta = -np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(2), jac.T @ residual)
            except np.linalg.LinAlgError:
                return None
            xy = xy + delta
            if np.abs(delta).max() < 1e-6:
                break
        edge = edges(xy)
        if edge is None or np.abs(edge - np.asarray(target, float)).max() * float(focal) > 4.0:
            return None
        return xy

    def locate_on_table(
        self,
        box,
        joints: dict,
        K=None,
        dist=None,
        mount=None,
        table_z: float = TABLE_Z,
        image_size: tuple[int, int] | None = None,
        shape: str = "upright",
    ) -> dict | None:
        """Object centre ``xyz`` (m), ``width`` and ``height`` from a box, or ``None``.

        ``box`` is ``(x1, y1, x2, y2)`` pixels in a frame of ``image_size``
        ``(w, h)`` taken at ``joints``. The box bottom's ray meets the table
        at the object's front edge. A bottle is the visual mesh resting on the
        table (the collision cylinder left the centre about 13 mm off, and the
        jaws closed beside it); a model with no bottle mesh keeps the cylinder
        step. A resting sphere (``shape="sphere"``), upright cylinder
        (``"upright"``), or thin table-plane footprint (``"flat"``) is fitted
        to the box. A box cut by the image's sides or bottom gives ``None``; a
        cut top edge is ignored only for upright objects.
        """
        x1, y1, x2, y2 = (float(v) for v in box)
        top_cut = False
        if image_size is not None:
            W, H = image_size
            if x1 <= 1 or x2 >= W - 1 or y2 >= H - 1:
                return None
            top_cut = y1 <= 1
        K = self.default_intrinsics(*(image_size or (None, None))) if K is None else np.asarray(K, float)
        if shape == "flat":
            if top_cut:
                return None
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            o, rays = self.rays(corners, joints, K, dist, mount)
            height = 0.012
            top_z = table_z + height
            if o[2] <= top_z or (rays[:, 2] >= -1e-6).any():
                return None
            points = o + rays * ((top_z - o[2]) / rays[:, 2])[:, None]
            side_a = 0.5 * (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3]))
            side_b = 0.5 * (np.linalg.norm(points[2] - points[1]) + np.linalg.norm(points[3] - points[0]))
            width = float(np.clip(min(side_a, side_b), 0.008, 0.08))
            _, centre_ray = self.rays([(0.5 * (x1 + x2), 0.5 * (y1 + y2))], joints, K, dist, mount)
            centre = o + centre_ray[0] * ((top_z - o[2]) / centre_ray[0, 2])
            centre[2] = table_z + 0.5 * height
            return {"xyz": centre, "width": width, "height": height, "shape": shape}
        uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        edges = [(x1, vc), (x2, vc), (uc, y1), (uc, y2)]
        o, d = self.rays(edges, joints, K, dist, mount)
        bottom = d[3]
        if o[2] <= table_z or bottom[2] >= -1e-6:
            return None
        R, t = self.camera_pose(joints, mount)
        front = o + bottom * ((table_z - o[2]) / bottom[2])
        half_w = 0.5 * (x2 - x1) / K[0, 0] * float((front - o) @ R[:, 2])
        horiz = np.array([bottom[0], bottom[1], 0.0])
        horiz /= max(np.linalg.norm(horiz), 1e-9)
        c0 = front + horiz * half_w
        if shape == "bottle":
            # The camera sees the mustard bottle's visual mesh, an oval, not
            # the r=0.02 collision cylinder. Matching that cylinder to the box
            # left the centre 13 mm off along the table and the jaws closed
            # beside the bottle. Fit the mesh. The cylinder step is the
            # fallback when this model has no bottle mesh, or the mesh cannot
            # explain the box. Either way one box and one pose give one centre.
            radius, height = 0.02, 0.085
            centre = front + horiz * radius
            observed = self._normalized(edges, K, dist)
            target = np.array([observed[0, 0], observed[1, 0], observed[3, 1]])

            def bottle_edges(xy):
                pc = (self._surface("upright", [xy[0], xy[1], radius, height], table_z) - t) @ R
                uv = pc[:, :2] / pc[:, 2:3]
                return np.array([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].max()])

            xy = centre[:2].copy()
            residual = bottle_edges(xy) - target
            jac = np.empty((3, 2))
            for j in range(2):
                step = np.zeros(2)
                step[j] = 1e-4
                jac[:, j] = (bottle_edges(xy + step) - bottle_edges(xy)) / 1e-4
            delta = -np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(2), jac.T @ residual)
            xy = xy + delta
            fitted = self._fit_bottle_centre(centre[:2], target, R, t, table_z, K[0, 0])
            if fitted is not None:
                xy = fitted
            return {
                "xyz": np.array([xy[0], xy[1], table_z + 0.5 * height]),
                "width": 2 * radius,
                "height": height,
                "shape": shape,
            }
        if shape == "sphere":
            params = np.array([c0[0], c0[1], half_w])
        else:
            params = np.array([c0[0], c0[1], half_w, 2 * half_w])
        n = self._normalized(edges, K, dist)
        obs = np.array([n[0, 0], n[1, 0], n[2, 1], n[3, 1]])
        lo = np.array([-np.inf, -np.inf, 0.004] + ([] if shape == "sphere" else [0.004]))
        hi = np.array([np.inf, np.inf, 0.15] + ([] if shape == "sphere" else [0.4]))

        def outline(p):
            pc = (self._surface(shape, p, table_z) - t) @ R
            if (pc[:, 2] <= 1e-3).any():
                return None
            xy = pc[:, :2] / pc[:, 2:3]
            top = xy[:, 1].min()
            if top_cut:  # the sides are what's left inside the image; the top lies above it
                xy = xy[xy[:, 1] >= obs[2]]
                if not len(xy):
                    return None
                top = obs[2] + max(0.0, top - obs[2])
            return np.array([xy[:, 0].min(), xy[:, 0].max(), top, xy[:, 1].max()]) - obs

        r = outline(params)
        if r is None:
            return None
        lam = 1e-3
        for _ in range(40):
            J = np.empty((len(r), len(params)))
            for j in range(len(params)):
                dp = np.zeros(len(params))
                dp[j] = 1e-4
                rj = outline(params + dp)
                if rj is None:
                    return None
                J[:, j] = (rj - r) / 1e-4
            step = -np.linalg.solve(J.T @ J + lam * np.eye(len(params)), J.T @ r)
            p_new = np.clip(params + step, lo, hi)
            r_new = outline(p_new)
            if r_new is not None and r_new @ r_new < r @ r:
                params, r, lam = p_new, r_new, lam * 0.3
                if np.abs(step).max() < 1e-6:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        if np.abs(r).max() * K[0, 0] > 3.0:  # the model can't explain the box
            return None
        if shape == "sphere":
            cx, cy, rad = params
            return {"xyz": np.array([cx, cy, table_z + rad]), "width": 2 * rad, "height": 2 * rad, "shape": shape}
        cx, cy, rad, h = params
        return {"xyz": np.array([cx, cy, table_z + 0.5 * h]), "width": 2 * rad, "height": h, "shape": shape}

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    def _pose(self, q, gripper: float) -> dict:
        return {**dict(zip(IK5_JOINTS, map(float, np.asarray(q, float)))), "gripper": float(gripper)}

    def _bounds5(self, limits: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self._limits[k][0] for k in IK5_JOINTS])
        hi = np.array([self._limits[k][1] for k in IK5_JOINTS])
        for i, k in enumerate(IK5_JOINTS):
            if limits and k in limits:
                lo[i], hi[i] = max(lo[i], float(limits[k][0])), min(hi[i], float(limits[k][1]))
        return lo, hi

    def _point_jacobian(self, d, point, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Translational/angular Jacobians per calibrated degree for ``point``."""
        import mujoco

        # world.fk() intentionally runs only mj_kinematics; Jacobians also
        # require the composite body/dof transforms produced by mj_comPos.
        mujoco.mj_comPos(self.model, d)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, d, jp, jr, np.asarray(point, float), self.model.body(body).id)
        scale = self._rad_per_cal
        return jp[:, self._dofs] * scale, jr[:, self._dofs] * scale

    def _grasp_state(self, q, width: float, gripper: float):
        pose = self._pose(q, gripper)
        p = self.world.grasp_point(pose, width)
        d = self.world.fk(pose)
        axis = d.body("gripper").xmat.reshape(3, 3)[:, 0].copy()
        jp, jw = self._point_jacobian(d, p, "gripper")
        # d(axis)/dq = omega/dq cross axis.
        ja = np.column_stack([np.cross(jw[:, i], axis) for i in range(5)])
        return pose, p, axis, jp, ja

    @staticmethod
    def _bounded_step(H, g, lower, upper) -> np.ndarray:
        """Deterministic active-set solve for a tiny box-constrained QP."""
        n = len(g)
        x = np.zeros(n)
        free = np.ones(n, bool)
        for _ in range(n + 1):
            if free.any():
                idx = np.flatnonzero(free)
                rhs = -g[idx] - H[np.ix_(idx, ~free)] @ x[~free]
                try:
                    x[idx] = np.linalg.solve(H[np.ix_(idx, idx)], rhs)
                except np.linalg.LinAlgError:
                    x[idx] = np.linalg.lstsq(H[np.ix_(idx, idx)], rhs, rcond=None)[0]
            bad_lo = free & (x < lower)
            bad_hi = free & (x > upper)
            if not (bad_lo | bad_hi).any():
                break
            # Activate the most violated bound, then re-solve the free variables.
            violation = np.maximum(lower - x, x - upper)
            j = int(np.argmax(np.where(bad_lo | bad_hi, violation, -np.inf)))
            x[j] = lower[j] if bad_lo[j] else upper[j]
            free[j] = False
        return np.clip(x, lower, upper)

    def _sqp(self, evaluate, seeds, lo, hi, iters=60):
        """Bounded damped Gauss-Newton SQP with trust region and line search."""
        best = None
        for initial in seeds:
            q = np.clip(np.asarray(initial, float), lo, hi)
            damping = _SQP_REG
            r, J, metrics = evaluate(q)
            cost = float(r @ r)
            for _ in range(iters):
                if metrics["converged"]:
                    break
                H = J.T @ J + damping * np.eye(5)
                g = J.T @ r
                lower = np.maximum(lo - q, -_SQP_TRUST_DEG)
                upper = np.minimum(hi - q, _SQP_TRUST_DEG)
                step = self._bounded_step(H, g, lower, upper)
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.125):
                    qn = np.clip(q + alpha * step, lo, hi)
                    rn, Jn, mn = evaluate(qn)
                    cn = float(rn @ rn)
                    if cn < cost - 1e-12:
                        q, r, J, metrics, cost = qn, rn, Jn, mn, cn
                        damping = max(damping * 0.4, 1e-9)
                        accepted = True
                        break
                if not accepted:
                    damping *= 10.0
                    if damping > 1e3 or np.linalg.norm(step, ord=np.inf) < 1e-5:
                        break
            cand = (not metrics["converged"], cost, q.copy(), metrics)
            if best is None or cand[:2] < best[:2]:
                best = cand
        return best[2], best[3]

    def _grasp_eval(self, q, target, width, gripper, object_axis, reference):
        _pose, p, jaw, jp, ja = self._grasp_state(q, width, gripper)
        rows = [p - target, [_AXIS_WEIGHT * jaw[2]]]
        jac = [jp, _AXIS_WEIGHT * ja[2:3]]
        yaw_err = 0.0
        if object_axis is not None:
            axis = np.asarray(object_axis, float)
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            yaw_err = float(jaw @ axis)
            rows.append([_AXIS_WEIGHT * yaw_err])
            jac.append(_AXIS_WEIGHT * (axis @ ja)[None, :])
        # Null-space preference: continuity and distance from joint stops.
        rows.append(1e-5 * (q - reference))
        jac.append(1e-5 * np.eye(5))
        r = np.concatenate([np.ravel(v) for v in rows])
        J = np.vstack(jac)
        pos_err = float(np.linalg.norm(p - target))
        metrics = {
            "pos_err": pos_err,
            "tilt": float(jaw[2]),
            "yaw_err": yaw_err,
            "converged": pos_err < 3e-4 and abs(jaw[2]) < 0.005
                         and (object_axis is None or abs(yaw_err) < 0.01),
        }
        return r, J, metrics

    def _elbow_pair(self, target, roll) -> list[list[float]]:
        """The two closed-form elbow branches for ``target``, as joint lists."""
        target = np.asarray(target, float)
        d0 = self.world.fk({})
        shoulder = d0.body("upper_arm").xpos
        elbow = d0.body("lower_arm").xpos
        wrist = d0.body("wrist").xpos
        l1 = float(np.linalg.norm(elbow - shoulder))
        l2 = float(np.linalg.norm(wrist - elbow))
        radial = float(np.linalg.norm(target[:2] - shoulder[:2]))
        dz = float(target[2] - shoulder[2])
        D = np.clip(math.hypot(radial, dz), abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
        interior = math.degrees(math.acos(np.clip((l1 * l1 + l2 * l2 - D * D) / (2 * l1 * l2), -1, 1)))
        gamma = math.degrees(math.acos(np.clip((l1 * l1 + D * D - l2 * l2) / (2 * l1 * D), -1, 1)))
        line = math.degrees(math.atan2(radial, dz))
        pan = -math.degrees(math.atan2(target[1] - shoulder[1], target[0] - shoulder[0]))
        out = []
        for elbow_sign in (1.0, -1.0):
            lift = line - elbow_sign * gamma
            flex = elbow_sign * (180.0 - interior)
            wrist_flex = 180.0 - lift - flex
            out.append([pan, lift, flex, wrist_flex, roll])
        return out

    def _overhead_elbow(self, target, roll, lo, hi) -> np.ndarray:
        """The one elbow branch that cocks the wrist so the hand comes from above.

        Larger ``wrist_flex`` is that branch. The other elbow, and the
        hand-tuned seeds, are not started: the same target then always
        finishes in the same pose.
        """
        pair = [np.clip(np.asarray(q, float), lo, hi) for q in self._elbow_pair(target, roll)]
        return max(pair, key=lambda q: (float(q[3]), -abs(float(q[2]))))

    def _geometric_seeds(self, target, roll, seed=None):
        """SO-101 elbow branches, initialized by planar triangle geometry.

        The exact calibrated offsets are model-specific, so these closed-form
        candidates are only seeds; MuJoCo FK and SQP produce the final answer.
        """
        target = np.asarray(target, float)
        d0 = self.world.fk({})
        shoulder = d0.body("upper_arm").xpos[:2]
        pan = -math.degrees(math.atan2(target[1] - shoulder[1], target[0] - shoulder[0]))
        out = []
        if seed:
            out.append([seed[k] for k in IK5_JOINTS])
        out.extend(self._elbow_pair(target, roll))
        out.extend([[pan, *s, roll] for s in _SEEDS])
        return out

    def lowest_point(self, joints: dict) -> float:
        """Lowest z (m) of the arm's collision geometry from the lower arm out."""
        with self._lock:
            d = self.world.fk(dict(joints))
            m = self.model
            low = math.inf
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            for g in self._clear_geoms:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                corners = (c + signs * h) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                low = min(low, float(corners[:, 2].min()))
            return low

    def hand_clearance(self, joints: dict, obstacle) -> float:
        """Closest approach (m) of the arm from the lower arm out to an upright
        cylinder ``obstacle`` = ``(x, y, radius, top_z)`` standing on the table;
        negative when they overlap."""
        x, y, radius, top_z = (float(v) for v in obstacle)
        with self._lock:
            d = self.world.fk(dict(joints))
            m = self.model
            points = []
            for g in self._clear_geoms:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                points.append((c + _BOX_GRID * h) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        points = np.vstack(points)
        radial = np.hypot(points[:, 0] - x, points[:, 1] - y) - radius
        above = points[:, 2] - top_z
        outside = np.where((radial > 0) & (above > 0), np.hypot(radial, above), np.maximum(radial, above))
        return float(outside.min())

    def ik(self, target, width: float = 0.04, gripper: float = OPEN_GRIPPER, seed: dict | None = None,
           wrist_roll: float | None = None, iters: int = 80,
           limits: dict | None = None, object_axis=None, one_branch: bool = False) -> tuple[dict, float, float]:
        """Best top-down pose for grasp point ``target``: ``(pose, pos_err_m, axis_z)``.

        ``wrist_roll`` stays fixed (it spins the jaws without changing the
        arm's reach). It defaults to the seed pose's roll. ``limits``
        (``{joint: (lo, hi)}``) narrows the model's joint range to the arm's.
        ``one_branch`` starts from the overhead elbow only, or from ``seed``
        once a pose on that branch already exists.
        """
        target = np.asarray(target, float)
        roll = float((seed or {}).get("wrist_roll", 0.0) if wrist_roll is None else wrist_roll)
        lo, hi = self._bounds5(limits)
        if object_axis is None:
            # Backwards-compatible fixed-roll solve.
            lo[4] = hi[4] = float(np.clip(roll, lo[4], hi[4]))
        roll0 = float(np.clip(roll, lo[4], hi[4]))
        if one_branch:
            if seed is not None:
                seeds = [[float(seed[k]) for k in IK5_JOINTS]]
            else:
                seeds = [self._overhead_elbow(target, roll0, lo, hi)]
        else:
            roll_candidates = [roll0]
            if object_axis is not None:
                roll_candidates += [
                    float(np.clip(roll0 + d, lo[4], hi[4]))
                    for d in (-180.0, -90.0, 90.0, 180.0)
                ]
            seeds = []
            for candidate in dict.fromkeys(round(v, 8) for v in roll_candidates):
                seeds.extend(self._geometric_seeds(target, candidate, seed if candidate == roll0 else None))
        reference = np.array(
            [float((seed or {}).get(k, 0.0)) for k in IK5_JOINTS], float
        )
        reference[4] = roll0
        with self._lock:
            q, metrics = self._sqp(
                lambda x: self._grasp_eval(x, target, width, gripper, object_axis, reference),
                seeds, lo, hi, iters,
            )
        return self._pose(q, gripper), metrics["pos_err"], metrics["tilt"]

    def solve_grasp(
        self,
        p,
        above_m: float = 0.06,
        width: float = 0.04,
        table_z: float = TABLE_Z,
        clearance_m: float = 0.003,
        tol_m: float = 0.003,
        max_tilt: float = 0.3,
        max_raise_m: float = 0.015,
        gripper: float = OPEN_GRIPPER,
        wrist_roll: float = 0.0,
        limits: dict | None = None,
        object_axis=None,
        seed: dict | None = None,
        obstacle=None,
        obstacle_clearance_m: float = 0.01,
    ) -> tuple[dict, dict] | None:
        """``(pregrasp, grasp)`` joint poses for an object centred at ``p``, or ``None``.

        The grasp puts ``p`` on the midpoint of the two inner faces with the
        jaws still open (``gripper``, default the full open angle) and the
        closing axis horizontal (within ``max_tilt``). The jaw stays that open
        through the approach and the descent; it closes only after the hand
        is there. If the jaw tips would touch the table it grasps up to
        ``max_raise_m`` higher. The pre-grasp is the same grasp ``above_m``
        straight above. With ``object_axis`` all five
        joints are solved together so the jaws close perpendicular to that
        table-plane axis; otherwise ``wrist_roll`` is fixed. ``None`` means
        out of reach (within ``limits``, see :meth:`ik`).

        ``obstacle`` is the object's own volume, an upright cylinder
        ``(x, y, radius, top_z)`` (see :meth:`hand_clearance`). A tall object
        is then grasped as low as the hand fits around it, up to
        ``_MIN_HOLD_M`` below its top, and the pre-grasp is raised until the
        whole hand is ``obstacle_clearance_m`` above it - so neither jaw nor
        the moving jaw's root can push it over on the way in.
        """
        p = np.asarray(p, float)

        def ok(pose, err, tilt):
            return err <= tol_m and abs(tilt) <= max_tilt and self.lowest_point(pose) >= table_z + clearance_m

        def clear(pose, margin):
            return obstacle is None or self.hand_clearance(pose, obstacle) >= margin

        # A standing object has one elbow branch. Searching the other one,
        # or the hand-tuned seeds, is what made the same bottle come down
        # on a different angle.
        one_branch = obstacle is not None
        if obstacle is not None:
            max_raise_m = max(max_raise_m, float(obstacle[3]) - _MIN_HOLD_M - p[2])
        grasp = None
        solve_seed = seed
        for raise_m in np.arange(0.0, max_raise_m + 1e-9, 0.003):
            pose, err, tilt = self.ik(
                p + [0.0, 0.0, raise_m], width, gripper, solve_seed, wrist_roll,
                limits=limits, object_axis=object_axis, one_branch=one_branch,
            )
            if err <= tol_m and abs(tilt) <= max_tilt:
                solve_seed = pose
                if ok(pose, err, tilt) and clear(pose, 0.0):
                    grasp = pose
                    break
        if grasp is None:
            return None
        top = self.world.grasp_point(grasp, width)
        rise = 0.0 if obstacle is None else max(0.0, float(obstacle[3]) - top[2])
        for extra in np.arange(0.0, rise + 1e-9, 0.01) if rise else [0.0]:
            pre, err, tilt = self.ik(
                top + [0.0, 0.0, above_m + extra], width, gripper, grasp,
                grasp["wrist_roll"], limits=limits, object_axis=object_axis,
                one_branch=one_branch,
            )
            if ok(pre, err, tilt) and clear(pre, obstacle_clearance_m):
                return pre, grasp
        return None

    def grasp_beside(
        self,
        grasp: dict,
        p,
        gap_m: float,
        width: float = 0.04,
        table_z: float = TABLE_Z,
        clearance_m: float = 0.003,
        tol_m: float = 0.003,
        max_tilt: float = 0.3,
        limits: dict | None = None,
    ) -> dict | None:
        """``grasp`` moved along the closing axis until the fixed finger's inner
        face is ``gap_m`` from the object at ``p``, or ``None`` when it already
        is that close or the moved pose does not solve.

        The hand comes down centred, where a lagging joint cannot put a finger
        on the object, and then moves sideways at grasp height. The moving jaw
        then pushes the object only ``gap_m`` into the fixed finger. From the
        centre it has to shove the object across half the open jaw, and an
        object that neither slides nor rolls drags the arm over and tilts the
        fingers instead.
        """
        p = np.asarray(p, float)
        with self._lock:
            b = self.world.fk(grasp).body("gripper")
            R = b.xmat.reshape(3, 3).copy()
            gap = float((R.T @ (p - b.xpos))[0]) - 0.5 * width - _FIXED_PAD_X
            point = self.world.grasp_point(grasp, width)
        if gap <= gap_m:
            return None
        pose, err, tilt = self.ik(
            point + R[:, 0] * (gap - gap_m), width, float(grasp["gripper"]), grasp,
            limits=limits, one_branch=True,
        )
        if err > tol_m or abs(tilt) > max_tilt or self.lowest_point(pose) < table_z + clearance_m:
            return None
        return pose

    # ------------------------------------------------------------------
    # viewing an oriented object
    # ------------------------------------------------------------------
    def table_axis(self, center_px, direction_px, length_px: float, joints: dict, K=None, dist=None,
                   mount=None, z: float = TABLE_Z) -> tuple[np.ndarray, np.ndarray] | None:
        """World ``(centre, unit_axis)`` of an image segment on the plane ``z``.

        ``center_px``/``direction_px``/``length_px`` come from
        :func:`principal_axis_in_box`. ``None`` if a ray misses the plane.
        """
        c = np.asarray(center_px, float)
        u = np.asarray(direction_px, float)
        half = 0.5 * max(float(length_px), 4.0)
        o, d = self.rays([c, c - u * half, c + u * half], joints, K, dist, mount)
        if (d[:, 2] >= -1e-6).any():
            return None
        pts = o + d * ((z - o[2]) / d[:, 2])[:, None]
        axis = pts[2] - pts[1]
        axis[2] = 0.0
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None
        return pts[0], axis / n

    def waypoints(self, start: dict, end: dict, width: float = 0.04, step_m: float = 0.015,
                  limits: dict | None = None) -> list[dict]:
        """Poses that move the grasp point along the straight line from
        ``start``'s to ``end``'s, ``end`` included.

        Servos interpolate each joint on its own, so one move between two
        poses only roughly follows the line; short steps keep the jaws on it.
        """
        with self._lock:
            a, b = self.world.grasp_point(start, width), self.world.grasp_point(end, width)
            n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step_m))
            out, seed = [], start
            for i in range(1, n):
                pose, err, _ = self.ik(a + (b - a) * (i / n), width, end["gripper"], seed, limits=limits)
                if err > 0.003:
                    return [end]
                out.append(pose)
                seed = pose
            return [*out, end]


def joint_line(start: dict, end: dict, max_step_deg: float = 8.0) -> list[dict]:
    """``end`` split into poses along the straight joint-space line from ``start``.

    A single goal lets every servo run its own profile, so the joint with the
    longest travel arrives last and the hand leaves the line between the two
    poses; short steps hold it on the line the planner checked.
    """
    keys = [k for k in end if k in start]
    n = max(1, math.ceil(max((abs(end[k] - start[k]) for k in keys), default=0.0) / max_step_deg))
    return [{**end, **{k: start[k] + (end[k] - start[k]) * i / n for k in keys}} for i in range(1, n + 1)]


def solve_mount(gripper_poses, board_poses):
    """Eye-in-hand calibration: the camera's mount on the gripper.

    ``gripper_poses`` are ``(R, t)`` gripper-in-base from FK, ``board_poses``
    the matching ``(R, t)`` board-in-camera from ``cv2.solvePnP``, for a
    board that stays put on the table. Returns ``(R, t, spread_m)``: the
    mount (camera in gripper, as :meth:`Kinematics.camera_pose` takes it)
    and how far the board's base-frame position wanders across the views -
    a few mm for a good calibration.
    """
    import cv2

    if len(gripper_poses) < 3 or len(gripper_poses) != len(board_poses):
        raise ValueError("need at least 3 matching gripper/board poses")
    Rg = [np.asarray(R, float) for R, _ in gripper_poses]
    tg = [np.asarray(t, float).reshape(3, 1) for _, t in gripper_poses]
    Rb = [np.asarray(R, float) for R, _ in board_poses]
    tb = [np.asarray(t, float).reshape(3, 1) for _, t in board_poses]
    R, t = cv2.calibrateHandEye(Rg, tg, Rb, tb, method=cv2.CALIB_HAND_EYE_PARK)
    t = t.reshape(3)
    origins = np.array([Rg[i] @ (R @ tb[i].reshape(3) + t) + tg[i].reshape(3) for i in range(len(Rg))])
    spread = float(np.linalg.norm(origins - origins.mean(axis=0), axis=1).max())
    return R, t, spread


_shared: dict = {}
_shared_lock = threading.Lock()


def shared_kinematics(gripper_closed_cal: float | None = None) -> Kinematics:
    """One :class:`Kinematics` per process and gripper floor (loading takes ~0.3 s).

    ``gripper_closed_cal`` is the stack's calibrated gripper value with the
    jaws touching (its ``GRIPPER_CLOSED_FLOOR``); it sets where the open
    moving jaw sits, and so the grasp point between the jaws.
    """
    with _shared_lock:
        if gripper_closed_cal not in _shared:
            cfg = None if gripper_closed_cal is None else {"gripper_map": {"closed_cal": float(gripper_closed_cal)}}
            _shared[gripper_closed_cal] = Kinematics(sim_cfg=cfg)
        return _shared[gripper_closed_cal]
