# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The twin's arm, exposed through the same API layers as the real one.

``SimBus``       - Feetech ``MotorsBus`` look-alike: ``read`` / ``write`` /
                   ``sync_read`` / ``sync_write`` on STS3215 register names,
                   normalized (calibrated deg, 0-100 gripper) or raw counts.
``SimFollower``  - LeRobot ``SO101Follower`` look-alike: ``connect`` (runs the
                   same ``configure()`` register writes), ``send_action`` with
                   the ``max_relative_target`` clamp, ``get_observation``,
                   ``disconnect``.
``SimArm``       - a small convenience facade (calibrated dicts in/out) for
                   notebooks and the ROS node.

Because the host stacks' transport (``LeRobotArm``) talks to a
``SO101Follower``, the sim adapter swaps in :class:`SimFollower`. Every retry,
effort-register probe and motion-profile write it does on the rig runs
unchanged here.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from so101_sim import MOTOR_NAMES

logger = logging.getLogger(__name__)

_SPEED_REGS = ("Goal_Velocity", "Goal_Speed", "Maximum_Speed_Limit")
_ACCEL_REGS = ("Acceleration", "Maximum_Acceleration")
# Accepted and stored, but no physical effect in the twin. Max_Torque_Limit
# is EEPROM: the servo copies it into Torque_Limit only at power-on.
_PASSIVE_REGS = {
    "Operating_Mode": 0,
    "P_Coefficient": 32,
    "I_Coefficient": 0,
    "D_Coefficient": 32,
    "Return_Delay_Time": 0,
    "Protection_Current": 500,
    "Max_Torque_Limit": 1000,
    "Lock": 0,
    "Min_Position_Limit": 0,
    "Max_Position_Limit": 4095,
    "Homing_Offset": 0,
}
# Overload protection: register -> (factory value, SimWorld.set_protection
# keyword, scale to a fraction of stall torque or to seconds).
_PROTECTION_REGS = {
    "Overload_Torque": (80, "threshold", 0.01),
    "Protective_Torque": (20, "protective", 0.01),
    "Protection_Time": (200, "time_s", 0.01),
}
_MOTOR_IDS = {m: i + 1 for i, m in enumerate(MOTOR_NAMES)}


class SimBus:
    """Feetech bus emulation over a :class:`~so101_sim.world.SimWorld`.

    Creating one is a power-on: ``Torque_Limit`` loads from
    ``Max_Torque_Limit`` and the protection registers take their factory values.
    """

    def __init__(self, world):
        self.world = world
        self.motors = dict(_MOTOR_IDS)
        self._regs = {
            m: {**_PASSIVE_REGS, **{k: v[0] for k, v in _PROTECTION_REGS.items()}}
            for m in MOTOR_NAMES
        }
        self._torque_limit = {}
        world.bus = self
        for m in MOTOR_NAMES:
            self.write("Torque_Limit", m, self._regs[m]["Max_Torque_Limit"])
            for reg in _PROTECTION_REGS:
                self.write(reg, m, self._regs[m][reg])

    def reload_torque_limits(self) -> None:
        """Power-on load: ``Torque_Limit`` RAM copies ``Max_Torque_Limit``."""
        for m in MOTOR_NAMES:
            self.write("Torque_Limit", m, self._regs[m]["Max_Torque_Limit"])

    # -- helpers ------------------------------------------------------------
    def _motors(self, motors) -> list[str]:
        if motors is None:
            return list(MOTOR_NAMES)
        if isinstance(motors, str):
            return [motors]
        return list(motors)

    def _idx(self, motor: str) -> int:
        try:
            return MOTOR_NAMES.index(motor)
        except ValueError:
            raise KeyError(f"unknown motor {motor!r}") from None

    # -- writes -------------------------------------------------------------
    def write(self, data_name: str, motor: str, value, normalize: bool = True, num_retry: int = 0):
        w = self.world
        i = self._idx(motor)
        if data_name == "Goal_Position":
            cal = float(value) if normalize else w.jmap.raw_to_cal(motor, value)
            w.set_goal_cal({motor: cal})
        elif data_name in _SPEED_REGS:
            with w.lock:
                w.servo.set_speed(i, float(value))
        elif data_name in _ACCEL_REGS:
            with w.lock:
                w.servo.set_accel(i, float(value))
        elif data_name == "Torque_Enable":
            w.set_torque(motor, bool(value))
        elif data_name == "Torque_Limit":
            self._torque_limit[motor] = int(min(max(int(value), 0), 1000))
            w.set_torque_limit(motor, self._torque_limit[motor] / 1000.0)
        elif data_name in _PROTECTION_REGS:
            _, key, scale = _PROTECTION_REGS[data_name]
            self._regs[motor][data_name] = int(value)
            w.set_protection(motor, **{key: int(value) * scale})
        elif data_name in _PASSIVE_REGS:
            self._regs[motor][data_name] = int(value)
        else:
            raise KeyError(f"register {data_name!r} not in the STS3215 control table")

    def sync_write(self, data_name: str, values, normalize: bool = True, num_retry: int = 0):
        if not isinstance(values, dict):
            values = {m: values for m in MOTOR_NAMES}
        for motor, v in values.items():
            self.write(data_name, motor, v, normalize=normalize)

    # -- reads --------------------------------------------------------------
    def read(self, data_name: str, motor: str, normalize: bool = True, num_retry: int = 0):
        return self.sync_read(data_name, [motor], normalize=normalize)[motor]

    def sync_read(self, data_name: str, motors=None, normalize: bool = True, num_retry: int = 0) -> dict:
        w = self.world
        names = self._motors(motors)
        idx = [self._idx(m) for m in names]
        jm = w.jmap
        if data_name == "Present_Position":
            q = w.joint_rad()
            out = {}
            for m, i in zip(names, idx):
                cal = jm.to_cal(m, q[i])
                if normalize:
                    out[m] = jm.quantize(m, cal) if w.cfg.get("quantize_readback") else cal
                else:
                    out[m] = jm.cal_to_raw(m, cal)
            return out
        if data_name == "Goal_Position":
            g = w.goal_cal()
            return {m: (g[m] if normalize else jm.cal_to_raw(m, g[m])) for m in names}
        if data_name == "Present_Velocity":
            v = w.joint_vel()
            return {
                m: int(round(v[i] / jm.steps_to_rad(m, 1.0))) for m, i in zip(names, idx)
            }
        if data_name == "Present_Current":
            raw = w.current.present_current()
            return {m: int(raw[i]) for m, i in zip(names, idx)}
        if data_name == "Present_Load":
            raw = w.current.present_load()
            return {m: int(raw[i]) for m, i in zip(names, idx)}
        if data_name in _SPEED_REGS:
            return {m: int(w.servo.speed_raw[i]) for m, i in zip(names, idx)}
        if data_name in _ACCEL_REGS:
            return {m: int(w.servo.accel_raw[i]) for m, i in zip(names, idx)}
        if data_name == "Torque_Enable":
            return {m: int(w.servo.torque[i]) for m, i in zip(names, idx)}
        if data_name == "Torque_Limit":
            return {m: self._torque_limit[m] for m in names}
        if data_name == "Status":
            with w.lock:
                tripped = w.protection.tripped.copy()
            return {m: 0x20 if tripped[i] else 0 for m, i in zip(names, idx)}
        if data_name == "Moving":
            with w.lock:
                moving = np.abs(w.servo.v_ref) > 1e-4
            return {m: int(moving[i]) for m, i in zip(names, idx)}
        if data_name == "Present_Voltage":
            return {m: 74 for m in names}  # 7.4 V supply
        if data_name == "Present_Temperature":
            return {m: 32 for m in names}
        if data_name in _PASSIVE_REGS or data_name in _PROTECTION_REGS:
            return {m: self._regs[m][data_name] for m in names}
        raise KeyError(f"register {data_name!r} not in the STS3215 control table")

    def configure_motors(self, return_delay_time: int = 0, maximum_acceleration: int = 254, acceleration: int = 254) -> None:
        for m in MOTOR_NAMES:
            self.write("Return_Delay_Time", m, return_delay_time)
            self.write("Maximum_Acceleration", m, maximum_acceleration)
            self.write("Acceleration", m, acceleration)

    def disable_torque(self, motors=None, num_retry: int = 0) -> None:
        for m in self._motors(motors):
            self.write("Torque_Enable", m, 0)

    def enable_torque(self, motors=None, num_retry: int = 0) -> None:
        for m in self._motors(motors):
            self.write("Torque_Enable", m, 1)

    @property
    def is_connected(self) -> bool:
        return True


def _ensure_safe_goal_position(goal_present: dict, max_relative_target) -> dict:
    """LeRobot's relative-target clamp: limit each goal to present +/- max."""
    out = {}
    for key, (goal, present) in goal_present.items():
        lim = max_relative_target[key] if isinstance(max_relative_target, dict) else max_relative_target
        diff = float(goal) - float(present)
        out[key] = float(present) + max(min(diff, float(lim)), -float(lim))
    return out


class SimFollower:
    """LeRobot ``SO101Follower`` stand-in bound to the twin."""

    name = "so101_follower"

    def __init__(self, world, max_relative_target=None, robot_id: str = "sim"):
        self.world = world
        self.bus = SimBus(world)
        self.config = SimpleNamespace(
            id=robot_id,
            port="sim://so101",
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=True,
            cameras={},
        )
        self.is_connected = False
        self.is_calibrated = True

    def connect(self, calibrate: bool = True) -> None:
        self.configure()
        self.bus.enable_torque()
        self.is_connected = True

    def configure(self) -> None:
        """The register writes ``SO101Follower.configure()`` makes on the rig."""
        self.bus.configure_motors()
        for m in MOTOR_NAMES:
            self.bus.write("Operating_Mode", m, 0)
            self.bus.write("P_Coefficient", m, 16)
            self.bus.write("I_Coefficient", m, 0)
            self.bus.write("D_Coefficient", m, 32)
            if m == "gripper":
                # EEPROM, so Torque_Limit stays at its power-on value until
                # the grip writes it.
                self.bus.write("Max_Torque_Limit", m, 500)
                self.bus.write("Protection_Current", m, 250)
                self.bus.write("Overload_Torque", m, 25)

    def send_action(self, action: dict) -> dict:
        goal = {k[: -len(".pos")]: float(v) for k, v in action.items() if k.endswith(".pos")}
        mrt = self.config.max_relative_target
        if mrt is not None:
            present = self.bus.sync_read("Present_Position", list(goal))
            goal = _ensure_safe_goal_position(
                {k: (g, present[k]) for k, g in goal.items()}, mrt
            )
        self.bus.sync_write("Goal_Position", goal)
        return {f"{m}.pos": v for m, v in goal.items()}

    def get_observation(self) -> dict:
        pos = self.bus.sync_read("Present_Position")
        return {f"{m}.pos": v for m, v in pos.items()}

    def disconnect(self) -> None:
        if self.config.disable_torque_on_disconnect:
            self.bus.disable_torque()
        self.is_connected = False


class SimArm:
    """Calibrated-units convenience facade (notebooks, the sim ROS node)."""

    def __init__(self, world=None, max_relative_target=None, speed_raw=1000, accel_raw=50):
        if world is None:
            from so101_sim.world import shared_world

            world = shared_world()
        self.world = world
        self.follower = SimFollower(world, max_relative_target=max_relative_target)
        self.follower.connect()
        self.bus = self.follower.bus
        if speed_raw is not None:
            self.bus.sync_write("Goal_Velocity", int(speed_raw), normalize=False)
        if accel_raw is not None:
            self.bus.sync_write("Acceleration", int(accel_raw), normalize=False)

    def send_joints(self, joints: dict) -> None:
        self.follower.send_action({f"{k}.pos": v for k, v in joints.items()})

    def get_joints(self) -> dict:
        return {k[:-4]: v for k, v in self.follower.get_observation().items()}

    def get_gripper_load(self) -> float:
        return float(self.bus.read("Present_Current", "gripper", normalize=False))

    def close(self) -> None:
        self.follower.disconnect()
