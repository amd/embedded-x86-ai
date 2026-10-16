# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Feetech STS3215 emulation: onboard motion profile + current sensing.

A real STS3215 does not jump to a new ``Goal_Position``: its firmware runs a
trapezoidal profile limited by ``Goal_Velocity`` (steps/s, 0 = max) and
``Acceleration`` (x100 steps/s^2, 0 = unlimited), and its PID tracks that
moving setpoint. :class:`ServoBank` reproduces the profile online (the goal
may change at any tick, as the 60 Hz server loop does) and hands MuJoCo's
position actuator the setpoint plus a velocity feed-forward
``ctrl = q_ref + (kv/kp) * qd_ref`` so the actuator's damping acts on the
tracking error rather than braking the whole move.

:class:`CurrentModel` turns actuator torque into ``Present_Current`` (6.5 mA
LSB) and ``Present_Load`` (0.1 % of stall) from the 7.4 V STS3215 datasheet:
6 mA idle, 150 mA running free, 650 mA at the rated 0.49 N·m, 2.5 A at the
1.91 N·m stall. Back-EMF is already in the joint damping, so it is
subtracted out of the actuator force before conversion.

:class:`OverloadProtection` is the firmware's burnout guard: a load at or
above ``Overload_Torque`` % for ``Protection_Time`` drops the output to
``Protective_Torque`` % until the next position command.
"""

from __future__ import annotations

import math

import numpy as np

from so101_sim import MOTOR_NAMES
from so101_sim.mapping import JointMap

_UNLIMITED_ACCEL = 1.0e4  # rad/s^2 - "Acceleration 0" means no ramp


class ServoBank:
    """Online trapezoidal setpoint generator for the six servos (radians)."""

    def __init__(self, jmap: JointMap, servo_cfg: dict, kv_over_kp: np.ndarray):
        self.names = MOTOR_NAMES
        n = len(self.names)
        self._accel_unit = float(servo_cfg.get("accel_unit_steps", 100.0))
        self._max_speed_raw = float(servo_cfg.get("max_speed_raw", 3073))
        self._rad_per_step = np.array(
            [jmap.steps_to_rad(m, 1.0) for m in self.names], dtype=float
        )
        self.speed_raw = np.full(n, float(servo_cfg.get("speed_raw", 0)))
        self.accel_raw = np.full(n, float(servo_cfg.get("accel_raw", 254)))
        self.goal = np.zeros(n)
        self.q_ref = np.zeros(n)
        self.v_ref = np.zeros(n)
        self.torque = np.ones(n, dtype=bool)
        self._ff = np.asarray(kv_over_kp, dtype=float)
        self._vmax = np.zeros(n)
        self._amax = np.zeros(n)
        self._refresh_limits()

    def _refresh_limits(self) -> None:
        speed = np.where(self.speed_raw <= 0, self._max_speed_raw, self.speed_raw)
        speed = np.minimum(speed, self._max_speed_raw)
        self._vmax = speed * self._rad_per_step
        self._amax = np.where(
            self.accel_raw <= 0,
            _UNLIMITED_ACCEL,
            self.accel_raw * self._accel_unit * self._rad_per_step,
        )

    def set_speed(self, idx: int, raw: float) -> None:
        self.speed_raw[idx] = max(float(raw), 0.0)
        self._refresh_limits()

    def set_accel(self, idx: int, raw: float) -> None:
        self.accel_raw[idx] = max(float(raw), 0.0)
        self._refresh_limits()

    def reset(self, q: np.ndarray) -> None:
        """Snap setpoints/goals to ``q`` with zero velocity (power-on)."""
        self.q_ref[:] = q
        self.goal[:] = q
        self.v_ref[:] = 0.0

    def hold(self, idx: int, q: float) -> None:
        """Re-seat one servo's profile at ``q`` (torque re-enable)."""
        self.q_ref[idx] = q
        self.goal[idx] = q
        self.v_ref[idx] = 0.0

    def update(self, dt: float) -> np.ndarray:
        """Advance every profile by ``dt``; returns the actuator ``ctrl`` vector."""
        e = self.goal - self.q_ref
        ae = np.abs(e)
        s = np.sign(e)
        v_des = s * np.minimum(self._vmax, np.sqrt(2.0 * self._amax * ae))
        step = self._amax * dt
        v_new = self.v_ref + np.clip(v_des - self.v_ref, -step, step)
        q_new = self.q_ref + 0.5 * (self.v_ref + v_new) * dt
        arrived = (e * (self.goal - q_new) <= 0.0) | (ae < 1e-6)
        q_new = np.where(arrived, self.goal, q_new)
        v_new = np.where(arrived, 0.0, v_new)
        self.q_ref = q_new
        self.v_ref = v_new
        return q_new + self._ff * v_new


class CurrentModel:
    """Actuator torque -> Present_Current / Present_Load register values."""

    def __init__(self, cm_cfg: dict, n: int = len(MOTOR_NAMES)):
        self.idle_a = float(cm_cfg.get("idle_a", 0.006))
        self.moving_a = float(cm_cfg.get("moving_a", 0.15))
        self.rated_nm = float(cm_cfg.get("rated_nm", 0.49))
        self.rated_a = float(cm_cfg.get("rated_a", 0.65))
        self.stall_a = float(cm_cfg.get("stall_a", 2.5))
        self.stall_torque = float(cm_cfg.get("stall_torque_nm", 1.91))
        self.ma_per_unit = float(cm_cfg.get("ma_per_unit", 6.5))
        self.noise_units = float(cm_cfg.get("noise_units", 3.0))
        self.lpf_hz = float(cm_cfg.get("lpf_hz", 20.0))
        self._rng = np.random.default_rng(cm_cfg.get("seed", 0))
        self._tau = np.zeros(n)  # low-passed |net motor torque|
        self._moving = np.zeros(n, dtype=bool)

    def reset(self) -> None:
        self._tau[:] = 0.0
        self._moving[:] = False

    def update(self, tau_motor: np.ndarray, dt: float, moving=None) -> None:
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.lpf_hz * dt)
        self._tau += alpha * (np.abs(tau_motor) - self._tau)
        if moving is not None:
            self._moving[:] = moving

    def amps(self) -> np.ndarray:
        base = np.where(self._moving, self.moving_a, self.idle_a)
        tau = np.minimum(self._tau, self.stall_torque)
        low = base + tau / self.rated_nm * (self.rated_a - base)
        high = self.rated_a + (tau - self.rated_nm) / (
            self.stall_torque - self.rated_nm
        ) * (self.stall_a - self.rated_a)
        return np.where(tau <= self.rated_nm, low, high)

    def load_frac(self) -> np.ndarray:
        """Low-passed |torque| as a fraction of stall (``Present_Load`` / 1000)."""
        return np.clip(self._tau / self.stall_torque, 0.0, 1.0)

    def present_current(self) -> np.ndarray:
        """Raw ``Present_Current`` counts (6.5 mA LSB) with readback noise."""
        raw = self.amps() * 1000.0 / self.ma_per_unit
        if self.noise_units > 0:
            raw = raw + self._rng.normal(0.0, self.noise_units, raw.shape)
        return np.clip(np.round(raw), 0, 1023).astype(int)

    def present_load(self) -> np.ndarray:
        """Raw ``Present_Load`` (0.1 % of stall torque, 0..1000)."""
        return np.round(self.load_frac() * 1000.0).astype(int)


class OverloadProtection:
    """Per-servo overload guard; ``cap()`` is the output fraction of stall torque.

    ``limit`` is ``Torque_Limit`` / 1000. While the load stays at or above
    ``threshold`` (``Overload_Torque`` / 100) for ``time_s``
    (``Protection_Time``), the servo trips and its output is capped at
    ``protective`` (``Protective_Torque`` / 100) until :meth:`clear`, which a
    ``Goal_Position`` write calls.
    """

    def __init__(self, n: int = len(MOTOR_NAMES)):
        self.limit = np.ones(n)
        self.threshold = np.full(n, 0.80)
        self.protective = np.full(n, 0.20)
        self.time_s = np.full(n, 2.0)
        self.tripped = np.zeros(n, dtype=bool)
        self._over_s = np.zeros(n)

    def reset(self) -> None:
        self.tripped[:] = False
        self._over_s[:] = 0.0

    def update(self, load_frac: np.ndarray, dt: float) -> bool:
        """Advance the timers; True when a servo trips on this step."""
        over = load_frac >= self.threshold
        self._over_s = np.where(over, self._over_s + dt, 0.0)
        trip = over & (self._over_s >= self.time_s) & ~self.tripped
        self.tripped |= trip
        return bool(trip.any())

    def clear(self, idx: int) -> bool:
        """Drop servo ``idx``'s protection; True when it had tripped.

        A goal stream must not keep an overloaded servo from tripping, so the
        timer only restarts after a trip.
        """
        was = bool(self.tripped[idx])
        if was:
            self.tripped[idx] = False
            self._over_s[idx] = 0.0
        return was

    def cap(self) -> np.ndarray:
        return np.where(
            self.tripped, np.minimum(self.limit, self.protective), self.limit
        )
