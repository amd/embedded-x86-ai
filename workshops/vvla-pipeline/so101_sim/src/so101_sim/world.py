# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The physics world: one MuJoCo model + data stepped in real time.

``SimWorld`` owns the ``MjModel``/``MjData`` pair and a background thread
that steps physics at ``physics_hz`` (500 Hz) locked to the wall clock, so
behaviors that ``time.sleep`` between commands see the arm move exactly as
fast as the real one. Every step runs the servo profiles
(:class:`~so101_sim.servo.ServoBank`), the current model and the servos'
overload protection.

All access to ``data`` goes through ``world.lock``; the render thread only
holds it long enough to copy the scene (``mjv_updateScene``), never while
the GPU draws.

Physics is CPU-only (MuJoCo's solver is single-threaded and ~150k steps/s
on this machine, i.e. <0.5 % of one core at 500 Hz); the iGPU is reserved
for rendering and the LLM.
"""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time

import numpy as np

from so101_sim import MOTOR_NAMES, gl
from so101_sim.config import MODELS_DIR, load_sim_config, scene_path
from so101_sim.mapping import JointMap
from so101_sim.servo import CurrentModel, OverloadProtection, ServoBank

logger = logging.getLogger(__name__)

# Grasp zone in the gripper body frame (see so101.xml): the fixed jaw's tip
# pad (fixed_jaw_box5, x=-0.0108, half-thickness 1 mm) faces +x, and the
# moving jaw's tip pads sweep onto it at z~-0.09 as the gripper closes.
_FIXED_PAD_X = -0.0098
_GRASP_DEPTH_Z = -0.086

# Collision geoms of the fixed finger, by geom name or (unnamed) mesh name.
# The palm (fixed_jaw_box1), the block above the finger (fixed_jaw_box2), and
# the wrist box are left out: an object resting on them is not held.
_FIXED_FINGER = frozenset({
    "fixed_jaw_box3", "fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6", "fixed_jaw_box7",
    "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3",
    "wrist_roll_follower_so101_gripper_part0_v1",
})

TABLE_Z = -0.005  # tabletop surface height in scene_vvla.xml


def _yaw_deg(quat) -> float:
    """Z rotation of a MuJoCo ``(w, x, y, z)`` quaternion, in degrees."""
    w, x, y, z = (float(v) for v in quat)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _pose_text(qpos7) -> str:
    x, y, z = (float(v) for v in qpos7[:3])
    return f"xyz ({x:.3f}, {y:.3f}, {z:.3f}) yaw {_yaw_deg(qpos7[3:7]):.1f} deg"


STANDS = ("stand_a", "stand_b")
_STAND_HALF_HEIGHT = 0.15


class SimWorld:
    """MuJoCo model + data, a real-time physics thread, and twin helpers."""

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        start: bool = True,
        rest_pose: dict | None = None,
        root=None,
    ):
        self.cfg = load_sim_config(cfg)
        gl.configure(self.cfg.get("gl", "auto"), self.cfg.get("egl_device", 0))
        import mujoco

        self._mj = mujoco
        self.scene_path = scene_path(self.cfg, root)
        if self.scene_path.parent == MODELS_DIR:
            from so101_sim.assets import require

            require()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = 1.0 / float(self.cfg["physics_hz"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.lock = threading.RLock()

        self.jmap = JointMap(self.cfg["joint_map"], self.cfg["gripper_map"])
        m = self.model
        self._qadr = np.array([m.joint(n).qposadr[0] for n in MOTOR_NAMES])
        self._dadr = np.array([m.joint(n).dofadr[0] for n in MOTOR_NAMES])
        self._act = np.array([m.actuator(n).id for n in MOTOR_NAMES])
        kp = m.actuator_gainprm[self._act, 0].copy()
        kv = -m.actuator_biasprm[self._act, 2].copy()
        self._gain_on = m.actuator_gainprm[self._act].copy()
        self._bias_on = m.actuator_biasprm[self._act].copy()
        self._jrange = m.jnt_range[[m.joint(n).id for n in MOTOR_NAMES]].copy()
        self.force_full = m.actuator_forcerange[self._act].copy()  # Torque_Limit 1000
        self.servo = ServoBank(self.jmap, self.cfg["servo"], kv / kp)
        self.current = CurrentModel(self.cfg["current_model"])
        self.protection = OverloadProtection(len(MOTOR_NAMES))

        pose = rest_pose or self.cfg.get("rest_pose")
        if not pose:
            from vla_pipeline.robot.arm_interface import REST_POSE

            pose = REST_POSE
        self.rest_pose = dict(pose)
        self._free = {}  # body name -> (qposadr, dofadr)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])
                self._free[body] = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        self._scratch = mujoco.MjData(m)  # FK without touching live state
        self._fixed_finger = self._collision_geoms("gripper", _FIXED_FINGER)
        self._moving_finger = self._collision_geoms("moving_jaw_so101_v1")

        self._running = False
        self._thread: threading.Thread | None = None
        self._viewer = None
        self._closed = False
        self.reset()
        if start:
            self.start()

    # ------------------------------------------------------------------
    # state reset
    # ------------------------------------------------------------------
    def reset(self, pose: dict | None = None) -> None:
        """``mj_resetData``, then hold the arm at ``pose`` (the rest pose).

        ``mj_resetData`` is MuJoCo's reset: ``qpos`` returns to ``qpos0``,
        mocap returns to the compiled body poses, and every other state and
        control value is cleared. Derived fields are stale until ``mj_forward``.
        The arm joints are then set to ``pose``, and ``ctrl`` is set to that
        pose so the next step holds it instead of driving toward the zeros
        ``mj_resetData`` left in ``ctrl``.
        """
        mj = self._mj
        pose = {**self.rest_pose, **(pose or {})}
        with self.lock:
            # Captured before mj_resetData so the log shows what the reset
            # threw away: a grasp turns the toothbrush and lowers Torque_Limit.
            left = {
                name: self.data.qpos[qadr : qadr + 7].copy()
                for name, (qadr, _dadr) in self._free.items()
            }
            bus = getattr(self, "bus", None)
            limit_before = None if bus is None else bus._torque_limit.get("gripper")
            tripped = bool(self.protection.tripped[MOTOR_NAMES.index("gripper")])
            mj.mj_resetData(self.model, self.data)
            for i, name in enumerate(MOTOR_NAMES):
                q = self.jmap.to_rad(name, pose[name])
                lo, hi = self._jrange[i]
                self.data.qpos[self._qadr[i]] = min(max(q, lo), hi)
            mj.mj_forward(self.model, self.data)
            q = self.data.qpos[self._qadr].copy()
            self.servo.reset(q)
            self.data.ctrl[self._act] = q
            self.current.reset()
            self.protection.reset()
            self._apply_torque_caps()
            # mj_resetData does not touch the servo registers. Torque_Limit is
            # RAM; a grasp lowers the gripper's, and a power-on copies it back
            # from Max_Torque_Limit.
            if bus is not None:
                bus.reload_torque_limits()
            limit_after = None if bus is None else bus._torque_limit.get("gripper")
            restored = {
                name: self.model.qpos0[qadr : qadr + 7].copy()
                for name, (qadr, _dadr) in self._free.items()
            }
            self._sync_epoch()
        for name in left:
            logger.info(
                "reset: %s left %s, restored %s",
                name, _pose_text(left[name]), _pose_text(restored[name]),
            )
        logger.info(
            "reset: gripper Torque_Limit %s (was %s); %s",
            "n/a" if limit_after is None else limit_after,
            "n/a" if limit_before is None else limit_before,
            "trip cleared" if tripped else "no trip",
        )

    # ------------------------------------------------------------------
    # physics thread
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._sync_epoch()
        self._thread = threading.Thread(
            target=self._loop, name="so101-physics", daemon=True
        )
        self._thread.start()
        if self.cfg.get("viewer"):
            self._open_viewer()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def _sync_epoch(self) -> None:
        self._wall0 = time.perf_counter()
        self._sim0 = float(self.data.time)

    def _step_once(self) -> None:
        mj = self._mj
        ctrl = self.servo.update(self.dt)
        self.data.ctrl[self._act] = ctrl
        mj.mj_step(self.model, self.data)
        tau = (
            self.data.actuator_force[self._act] + self.data.qfrc_damper[self._dadr]
        )
        tau = np.where(self.servo.torque, tau, 0.0)
        self.current.update(tau, self.dt, moving=np.abs(self.servo.v_ref) > 1e-4)
        if self.protection.update(self.current.load_frac(), self.dt):
            self._apply_torque_caps()

    def _apply_torque_caps(self) -> None:
        self.model.actuator_forcerange[self._act] = (
            self.force_full * self.protection.cap()[:, None]
        )

    def advance(self, seconds: float) -> None:
        """Step ``seconds`` of physics synchronously (thread must be stopped)."""
        if self._running:
            raise RuntimeError("advance() needs the physics thread stopped")
        n = max(int(round(seconds / self.dt)), 1)
        with self.lock:
            for _ in range(n):
                self._step_once()

    def _loop(self) -> None:
        rtf = float(self.cfg.get("realtime_factor", 1.0))
        max_lag = float(self.cfg.get("max_catchup_s", 0.05))
        next_view = 0.0
        while self._running:
            target = self._sim0 + (time.perf_counter() - self._wall0) * rtf
            lag = target - self.data.time
            n = int(lag / self.dt)
            if lag > max_lag:  # stalled (GC, debugger, heavy load): drop time
                with self.lock:
                    self._sync_epoch()
                n = 1
            if n > 0:
                with self.lock:
                    for _ in range(n):
                        self._step_once()
            if self._viewer is not None and time.perf_counter() >= next_view:
                next_view = time.perf_counter() + 1.0 / 60.0
                try:
                    with self.lock:
                        self._viewer.sync()
                except Exception:
                    self._viewer = None
            time.sleep(0.001)

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
        except Exception as e:
            logger.warning("MuJoCo viewer unavailable (%s) - running headless", e)
            self._viewer = None

    # ------------------------------------------------------------------
    # servo-level access (used by SimBus)
    # ------------------------------------------------------------------
    def joint_rad(self) -> np.ndarray:
        with self.lock:
            return self.data.qpos[self._qadr].copy()

    def joint_vel(self) -> np.ndarray:
        with self.lock:
            return self.data.qvel[self._dadr].copy()

    def joint_cal(self) -> dict[str, float]:
        q = self.joint_rad()
        return {n: self.jmap.to_cal(n, q[i]) for i, n in enumerate(MOTOR_NAMES)}

    def set_goal_cal(self, joints: dict[str, float]) -> None:
        with self.lock:
            for name, cal in joints.items():
                if name not in MOTOR_NAMES:
                    continue
                i = MOTOR_NAMES.index(name)
                lo, hi = self._jrange[i]
                self.servo.goal[i] = min(max(self.jmap.to_rad(name, cal), lo), hi)
                if self.protection.clear(i):
                    self._apply_torque_caps()

    def set_torque_limit(self, name: str, frac: float) -> None:
        """Torque_Limit: cap one servo's output at ``frac`` of stall torque."""
        i = MOTOR_NAMES.index(name)
        with self.lock:
            self.protection.limit[i] = min(max(float(frac), 0.0), 1.0)
            self._apply_torque_caps()

    def set_protection(
        self, name: str, threshold=None, protective=None, time_s=None
    ) -> None:
        """Overload_Torque / Protective_Torque / Protection_Time, as fractions and seconds."""
        i = MOTOR_NAMES.index(name)
        p = self.protection
        with self.lock:
            if threshold is not None:
                p.threshold[i] = threshold
            if protective is not None:
                p.protective[i] = protective
            if time_s is not None:
                p.time_s[i] = time_s
            self._apply_torque_caps()

    def goal_cal(self) -> dict[str, float]:
        with self.lock:
            g = self.servo.goal.copy()
        return {n: self.jmap.to_cal(n, g[i]) for i, n in enumerate(MOTOR_NAMES)}

    def set_torque(self, name: str, enabled: bool) -> None:
        """Torque_Enable: off makes the joint limp (actuator gains zeroed)."""
        i = MOTOR_NAMES.index(name)
        a = self._act[i]
        with self.lock:
            if enabled:
                self.model.actuator_gainprm[a] = self._gain_on[i]
                self.model.actuator_biasprm[a] = self._bias_on[i]
                if not self.servo.torque[i]:
                    self.servo.hold(i, self.data.qpos[self._qadr[i]])
            else:
                self.model.actuator_gainprm[a, 0] = 0.0
                self.model.actuator_biasprm[a, 1:3] = 0.0
            self.servo.torque[i] = bool(enabled)

    # ------------------------------------------------------------------
    # scene helpers (tests, notebook, calibration)
    # ------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def objects(self) -> list[str]:
        return list(self._free)

    def body_pos(self, name: str) -> np.ndarray:
        with self.lock:
            return self.data.body(name).xpos.copy()

    def body_quat(self, name: str) -> np.ndarray:
        with self.lock:
            return self.data.body(name).xquat.copy()

    def place(self, name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport a free prop body and zero its velocity."""
        qadr, dadr = self._free[name]
        with self.lock:
            self.data.qpos[qadr : qadr + 3] = pos
            self.data.qpos[qadr + 3 : qadr + 7] = quat
            self.data.qvel[dadr : dadr + 6] = 0.0
            self._mj.mj_forward(self.model, self.data)

    def hide(self, name: str) -> None:
        """Move an object far off the table (out of every camera)."""
        k = self.objects.index(name)
        self.place(name, (-1.0 - 0.2 * k, 1.5, 0.05))

    def set_stand(self, stand: str, xy, top_z: float) -> None:
        """Raise a mocap pedestal so its top surface sits at ``top_z``."""
        mid = self.model.body(stand).mocapid[0]
        with self.lock:
            self.data.mocap_pos[mid] = (xy[0], xy[1], top_z - _STAND_HALF_HEIGHT)

    def park_stand(self, stand: str) -> None:
        mid = self.model.body(stand).mocapid[0]
        with self.lock:
            x, y = self.model.body_pos[self.model.body(stand).id][:2]
            self.data.mocap_pos[mid] = (x, y, TABLE_Z - 0.01 - _STAND_HALF_HEIGHT)

    def _stand_mocap_ids(self) -> dict[str, int]:
        out = {}
        for stand in STANDS:
            try:
                out[stand] = int(self.model.body(stand).mocapid[0])
            except KeyError:  # a scene without pedestals
                continue
        return out

    def snapshot_scene(self) -> dict:
        """Every prop's free-joint ``qpos`` and every pedestal's position, JSON-ready.

        The arm is not part of the snapshot; :meth:`apply_scene` puts the props
        and pedestals back without moving it.
        """
        with self.lock:
            bodies = {
                name: [float(v) for v in self.data.qpos[qadr : qadr + 7]]
                for name, (qadr, _dadr) in self._free.items()
            }
            mocap = {
                stand: [float(v) for v in self.data.mocap_pos[mid]]
                for stand, mid in self._stand_mocap_ids().items()
            }
        return {"bodies": bodies, "mocap": mocap}

    def apply_scene(self, scene: dict) -> None:
        """Put props and pedestals where :meth:`snapshot_scene` recorded them, at rest."""
        mocap_ids = self._stand_mocap_ids()
        with self.lock:
            for name, qpos in (scene.get("bodies") or {}).items():
                qadr, dadr = self._free[name]
                self.data.qpos[qadr : qadr + 7] = qpos
                self.data.qvel[dadr : dadr + 6] = 0.0
            for stand, pos in (scene.get("mocap") or {}).items():
                self.data.mocap_pos[mocap_ids[stand]] = pos
            self._mj.mj_forward(self.model, self.data)

    def fk(self, pose_cal: dict) -> "object":
        """Forward kinematics of ``pose_cal`` (rest-pose defaults) on scratch data."""
        mj = self._mj
        pose = {**self.rest_pose, **pose_cal}
        d = self._scratch
        with self.lock:
            d.qpos[:] = self.data.qpos
        for i, name in enumerate(MOTOR_NAMES):
            d.qpos[self._qadr[i]] = self.jmap.to_rad(name, pose[name])
        mj.mj_kinematics(self.model, d)
        return d

    def grasp_point(self, pose_cal: dict, width: float = 0.032) -> np.ndarray:
        """World point between the jaws of ``pose_cal`` for an object of ``width``.

        Centred between the fixed pad and the moving pad at the pose's gripper
        opening (or resting against the fixed pad if the jaws are nearly
        closed), so putting an object here makes the configured pose close on
        it - what the operator does on the rig when tuning ``block_pose``.
        """
        d = self.fk(pose_cal)
        b = d.body("gripper")
        R = b.xmat.reshape(3, 3)
        moving = R.T @ (d.geom("moving_jaw_box3").xpos - b.xpos)
        mid = 0.5 * (_FIXED_PAD_X + moving[0] - 0.001)
        x = max(mid, _FIXED_PAD_X + 0.5 * width + 0.002)
        return b.xpos + R @ np.array([x, 0.0, _GRASP_DEPTH_Z])

    def jaw_axis(self, pose_cal: dict) -> np.ndarray:
        """World direction the moving jaw closes along (fixed pad -> moving pad)."""
        return self.fk(pose_cal).body("gripper").xmat.reshape(3, 3)[:, 0].copy()

    def stage_object(
        self,
        name: str,
        pose_cal: dict,
        width: float | None = None,
        half_height: float | None = None,
        stand: str = "stand_a",
    ) -> np.ndarray:
        """Put ``name`` between the jaws of ``pose_cal``.

        The object is supported the way an operator would set it up: on the
        fixed jaw when the jaws close vertically (a hand-over), else on the
        table or a mocap pedestal raised to the right height.
        """
        geom = self.model.body(name).geomadr[0]
        size = self.model.geom_size[geom]
        mj = self._mj
        gtype = int(self.model.geom_type[geom])
        sphere = gtype == int(mj.mjtGeom.mjGEOM_SPHERE)
        capsule = gtype == int(mj.mjtGeom.mjGEOM_CAPSULE)
        cylinder = gtype == int(mj.mjtGeom.mjGEOM_CYLINDER)
        if width is None:
            width = 2.0 * size[0]
        if half_height is None:
            if sphere or capsule:
                half_height = size[0]
            elif cylinder:
                half_height = size[1]
            else:
                half_height = size[2]
        axis = self.jaw_axis(pose_cal)
        p = self.grasp_point(pose_cal, width)
        vertical = axis[2] > 0.6
        if vertical:
            # jaws close vertically: the object rests on the fixed jaw
            p = self.grasp_point({**pose_cal, "gripper": self.rest_pose["gripper"]}, width)
            push = axis
        else:
            push = np.array([axis[0], axis[1], 0.0])
            push /= max(np.linalg.norm(push), 1e-9)
        # flat on its support, faces square to the jaws (yaw only)
        yaw = math.atan2(axis[1], axis[0]) if abs(axis[2]) < 0.95 else 0.0
        quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        if capsule:
            tilt = np.zeros(4)
            mj.mju_axisAngle2Quat(tilt, np.array([0.0, 1.0, 0.0]), math.pi / 2)
            out = np.zeros(4)
            mj.mju_mulQuat(out, quat, tilt)
            quat = out
        arm_bodies = {"gripper", "moving_jaw_so101_v1"}
        for _ in range(20):
            if not vertical:
                bottom = p[2] - half_height
                if bottom > TABLE_Z + 0.002:
                    self.set_stand(stand, p[:2], bottom)
                else:
                    p[2] = TABLE_Z + half_height
            self.place(name, p, quat)
            depth = max(
                (pen for body, _g, pen in self.contact_report(name) if body in arm_bodies),
                default=0.0,
            )
            if depth <= 0.0005:
                break
            p = p + push * 0.002  # slide away from the fixed jaw, toward the open side
        return p

    def contact_report(self, body: str) -> list[tuple[str, str, float]]:
        """``(other_body, other_geom, penetration_m)`` for each contact on ``body``."""
        mj = self._mj
        m = self.model
        bid = m.body(body).id
        out = []
        with self.lock:
            mj.mj_forward(m, self.data)
            for k in range(self.data.ncon):
                c = self.data.contact[k]
                for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if m.geom_bodyid[mine] == bid:
                        gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
                        bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])
                        out.append((bname, gname, round(-float(c.dist), 4)))
        return out

    def _collision_geoms(self, body: str, names=None) -> frozenset[int]:
        """Colliding geom ids on ``body``, only those in ``names`` when given."""
        mj, m = self._mj, self.model
        bid = m.body(body).id
        out = set()
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if names is not None:
                label = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g)
                if label is None and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                    label = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g]))
                if label not in names:
                    continue
            out.add(g)
        return frozenset(out)

    def is_grasped(self, body: str) -> bool:
        """Both fingers touch ``body``: the fixed finger and the moving jaw.
        Contact with the palm or the wrist does not count."""
        m = self.model
        bid = m.body(body).id
        fixed = moving = False
        with self.lock:
            for k in range(self.data.ncon):
                c = self.data.contact[k]
                if m.geom_bodyid[c.geom1] == bid:
                    other = c.geom2
                elif m.geom_bodyid[c.geom2] == bid:
                    other = c.geom1
                else:
                    continue
                fixed = fixed or other in self._fixed_finger
                moving = moving or other in self._moving_finger
        return fixed and moving

    def grip_report(self) -> str:
        """Which finger pads touch what, and where each touched object's
        centre sits in the jaw frame (``gripperframe``, mm)."""
        mj, m = self._mj, self.model
        fingers = self._fixed_finger | self._moving_finger
        pads: dict[str, set[str]] = {}
        with self.lock:
            for k in range(self.data.ncon):
                c = self.data.contact[k]
                for finger, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if finger not in fingers or other in fingers:
                        continue
                    body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])
                    label = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, finger)
                    if label is None:
                        label = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, int(m.geom_dataid[finger]))
                    pads.setdefault(body, set()).add(label or f"geom{finger}")
            site = self.data.site("gripperframe")
            rot, centre = site.xmat.reshape(3, 3), site.xpos.copy()
            where = {b: 1000.0 * rot.T @ (self.data.body(b).xpos - centre) for b in pads if b != "table"}
        if not pads:
            return "nothing on the fingers"
        parts = []
        for body in sorted(pads):
            text = f"{body} on {'+'.join(sorted(pads[body]))}"
            if body in where:
                x, y, z = where[body]
                text += f", centre ({x:+.1f}, {y:+.1f}, {z:+.1f}) mm in the jaw frame"
            parts.append(text)
        return "; ".join(parts)

    def in_gripper(self, body: str) -> np.ndarray:
        """``body``'s position in the gripper body frame (m). It stays put
        while the object is held; any change is the object slipping."""
        with self.lock:
            g = self.data.body("gripper")
            return g.xmat.reshape(3, 3).T @ (self.data.body(body).xpos - g.xpos)

    def snapshot(self) -> dict:
        """Compact state for logs / video HUD (calibrated units)."""
        with self.lock:
            q = self.data.qpos[self._qadr].copy()
            g = self.servo.goal.copy()
            t = float(self.data.time)
        amps = self.current.amps()
        return {
            "t": t,
            "joints": {n: self.jmap.to_cal(n, q[i]) for i, n in enumerate(MOTOR_NAMES)},
            "goal": {n: self.jmap.to_cal(n, g[i]) for i, n in enumerate(MOTOR_NAMES)},
            "current_a": {n: float(amps[i]) for i, n in enumerate(MOTOR_NAMES)},
        }

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------
# one world per process: the in-process arm and cameras must share it
# ----------------------------------------------------------------------
_shared: SimWorld | None = None
_shared_lock = threading.Lock()


def shared_world(cfg: dict | None = None, **kwargs) -> SimWorld:
    """The process-wide :class:`SimWorld` (created on first call)."""
    global _shared
    with _shared_lock:
        if _shared is None or _shared._closed:
            _shared = SimWorld(cfg, **kwargs)
        elif kwargs.get("rest_pose") and _shared.rest_pose != kwargs["rest_pose"]:
            _shared.rest_pose = dict(kwargs["rest_pose"])
        return _shared


def close_shared_world() -> None:
    """Tear down the shared renderer (if any) and world."""
    global _shared
    import sys

    render = sys.modules.get("so101_sim.render")
    if render is not None:
        render.close_shared_renderer()
    with _shared_lock:
        if _shared is not None:
            _shared.close()
            _shared = None


atexit.register(close_shared_world)
