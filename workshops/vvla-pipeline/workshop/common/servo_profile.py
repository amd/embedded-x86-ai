# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Per-move STS3215 speed and acceleration from the arm's gravity and inertia (PROVIDED).

A direct-goal move is one ``Goal_Position`` write; the servo's onboard
trapezoid profile then runs at the ``Goal_Velocity`` and ``Acceleration`` in
its registers. :class:`ServoProfiler` picks those two registers for each joint
of each move so the servo stays inside its torque budget:

- cruise: gravity + friction + damping x speed <= ``cruise_frac`` of stall
  torque. The twin's joint damping is the servo's back-EMF, so this is the
  servo's torque-speed line.
- ramp: cruise torque + inertia x acceleration <= ``peak_frac`` of stall
  torque, a peak that lasts only the ramp, far shorter than the overload
  protection time.

Gravity and inertia come from the arm model at every pose along the move's
joint line, with the heaviest table prop in the jaws, so the arm runs fast
when folded and slows only where its own weight needs the torque. The policy
(the two fractions and the payload) comes from ``config/servo_profiles.json``,
which ``scripts/derive_servo_profiles.py`` writes; :func:`profile_for` is the
one formula both use. The twin runs the same registers, so the predicted
duration (:attr:`MovePlan.eta_s`) holds for the twin and the real arm alike.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

from so101_sim import MOTOR_NAMES

PROFILE_FILE = Path(__file__).resolve().parents[2] / "config" / "servo_profiles.json"
# The slowest profile this code writes: a joint whose budget is used up by
# gravity still creeps toward its goal instead of standing still.
MIN_SPEED_RAW = 50
MIN_ACCEL_RAW = 1
MAX_ACCEL_RAW = 254  # the register's largest value; 0 would mean "no ramp"
PROPS = ("sports_ball", "bottle", "book", "toothbrush")  # what the jaws may carry


def heaviest_prop(model) -> tuple[str, float]:
    """``(name, kg)`` of the heaviest table prop in the arm model: the payload a profile allows for."""
    masses = {n: float(model.body(n).mass[0]) for n in PROPS}
    name = max(masses, key=masses.get)
    return name, masses[name]


def profile_for(gravity_nm: float, inertia: float, damping: float, friction_nm: float, *,
                rad_per_step: float, stall_nm: float, cruise_frac: float, peak_frac: float,
                max_speed_raw: float, accel_unit_steps: float) -> tuple[int, int]:
    """``(Goal_Velocity, Acceleration)`` raw register values for one joint and load.

    ``rad_per_step`` is the joint's radians per servo step
    (``JointMap.steps_to_rad``): the jaws' linkage makes the gripper's differ.
    """
    speed = max((cruise_frac * stall_nm - gravity_nm - friction_nm) / damping, 0.0)
    speed = min(speed, max_speed_raw * rad_per_step)
    speed_raw = int(min(max(round(speed / rad_per_step), MIN_SPEED_RAW), max_speed_raw))
    cruise_nm = gravity_nm + friction_nm + damping * speed_raw * rad_per_step
    accel = max((peak_frac * stall_nm - cruise_nm) / inertia, 0.0)
    accel_raw = int(min(max(round(accel / (accel_unit_steps * rad_per_step)), MIN_ACCEL_RAW), MAX_ACCEL_RAW))
    return speed_raw, accel_raw


def trapezoid_s(distance_rad: float, speed_rad_s: float, accel_rad_s2: float) -> float:
    """Duration of a rest-to-rest trapezoid (triangle when it never reaches cruise)."""
    if distance_rad <= 0.0:
        return 0.0
    if distance_rad >= speed_rad_s * speed_rad_s / accel_rad_s2:
        return distance_rad / speed_rad_s + speed_rad_s / accel_rad_s2
    return 2.0 * math.sqrt(distance_rad / accel_rad_s2)


@dataclass(frozen=True)
class MovePlan:
    speed_raw: dict      # joint -> Goal_Velocity
    accel_raw: dict      # joint -> Acceleration
    eta_s: float         # the longest joint's profile duration


class ServoProfiler:
    """Plans each move's per-joint profile on a private copy of the arm model."""

    def __init__(self, profile_file: Path = PROFILE_FILE):
        import mujoco

        from so101_sim.config import DEFAULT_SCENE, load_sim_config
        from so101_sim.mapping import JointMap

        table = json.loads(Path(profile_file).read_text())
        self.cruise_frac = float(table["policy"]["cruise_frac_of_stall"])
        self.peak_frac = float(table["policy"]["peak_frac_of_stall"])
        self.static = {j: (v["speed_raw"], v["accel_raw"]) for j, v in table["joints"].items()}
        sim = load_sim_config()
        self.stall_nm = float(sim["current_model"]["stall_torque_nm"])
        self.max_speed_raw = float(sim["servo"]["max_speed_raw"])
        self.accel_unit = float(sim["servo"]["accel_unit_steps"])
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        self.rad_per_step = {j: self._jmap.steps_to_rad(j, 1.0) for j in MOTOR_NAMES}
        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        self._model.body_mass[self._model.jnt_bodyid[self._model.joint("wrist_roll").id]] += heaviest_prop(
            self._model
        )[1]
        self._data = mujoco.MjData(self._model)
        self._qadr = [int(self._model.joint(n).qposadr[0]) for n in MOTOR_NAMES]
        self._dadr = [int(self._model.joint(n).dofadr[0]) for n in MOTOR_NAMES]
        self._full = None
        self._lock = threading.Lock()

    def _loads(self, poses: list[dict]) -> tuple[list[float], list[float]]:
        """Per joint, the largest gravity torque and inertia over ``poses``."""
        import numpy as np

        m, d, mj = self._model, self._data, self._mj
        if self._full is None:
            self._full = np.zeros((m.nv, m.nv))
        gravity = [0.0] * len(MOTOR_NAMES)
        inertia = [0.0] * len(MOTOR_NAMES)
        for pose in poses:
            d.qpos[:] = m.qpos0
            d.qvel[:] = 0.0
            for i, name in enumerate(MOTOR_NAMES):
                if name in pose:
                    d.qpos[self._qadr[i]] = self._jmap.to_rad(name, float(pose[name]))
            mj.mj_forward(m, d)
            mj.mj_fullM(m, d, self._full)
            for i, dof in enumerate(self._dadr):
                gravity[i] = max(gravity[i], abs(float(d.qfrc_bias[dof])))
                inertia[i] = max(inertia[i], float(self._full[dof, dof]))
        return gravity, inertia

    def plan(self, start: dict, goal: dict, lockstep: bool = False) -> MovePlan:
        """Registers for moving the joints in ``goal`` from ``start`` (calibrated degrees).

        ``lockstep`` slows the faster joints so every joint finishes with the
        slowest one; their straight joint-space line is the path the planner
        checked.
        """
        from so101_sim.kinematics import joint_line

        joints = [j for j in MOTOR_NAMES if j in goal]
        whole = {**start, **goal}
        with self._lock:
            gravity, inertia = self._loads([start, *joint_line(start, whole, max_step_deg=10.0)])
            m = self._model
            speed_raw, accel_raw, eta = {}, {}, {}
            for j in joints:
                i = MOTOR_NAMES.index(j)
                dof = self._dadr[i]
                speed_raw[j], accel_raw[j] = profile_for(
                    gravity[i], inertia[i], float(m.dof_damping[dof]), float(m.dof_frictionloss[dof]),
                    rad_per_step=self.rad_per_step[j], stall_nm=self.stall_nm, cruise_frac=self.cruise_frac,
                    peak_frac=self.peak_frac, max_speed_raw=self.max_speed_raw, accel_unit_steps=self.accel_unit,
                )
                eta[j] = trapezoid_s(*self._rad(j, start, goal, speed_raw[j], accel_raw[j]))
        if lockstep and eta:
            longest = max(eta.values())
            for j in joints:
                if longest > 0.0 and eta[j] > 0.0:
                    scale = eta[j] / longest
                    speed_raw[j] = max(int(speed_raw[j] * scale), MIN_SPEED_RAW)
                    accel_raw[j] = max(int(accel_raw[j] * scale * scale), MIN_ACCEL_RAW)
                    eta[j] = trapezoid_s(*self._rad(j, start, goal, speed_raw[j], accel_raw[j]))
        return MovePlan(speed_raw, accel_raw, max(eta.values(), default=0.0))

    def _rad(self, joint: str, start: dict, goal: dict, speed_raw: int, accel_raw: int):
        distance = abs(self._jmap.to_rad(joint, float(goal[joint])) - self._jmap.to_rad(joint, float(start.get(joint, goal[joint]))))
        step = self.rad_per_step[joint]
        return distance, speed_raw * step, accel_raw * self.accel_unit * step


_shared: ServoProfiler | None = None
_shared_lock = threading.Lock()


def shared_profiler() -> ServoProfiler:
    """One profiler per process (loading the model takes a few hundred ms)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ServoProfiler()
        return _shared
