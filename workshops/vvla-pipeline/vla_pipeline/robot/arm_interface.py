# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""SO-101 arm access layer: one ``ArmClient`` API, three transports.

Behaviors never touch hardware directly - they talk to an :class:`ArmClient`:

- :class:`DirectArm`   - owns the Feetech serial bus via LeRobot's
  ``SO101Follower``. Used for component tests and, owned by a thread of this
  process (:class:`BusOwnerArm`), for ``robot.use_ros2: false``.
- :class:`DryRunArm`   - no hardware; integrates commands so behaviors can be
  tested end-to-end on any machine.
- ``Ros2ArmClient`` (in :mod:`vla_pipeline.robot.robot_node`) - drives the
  ros2_control controllers on top of the ROS 2 node that owns the bus.

The digital twin's :class:`~vla_pipeline.robot.sim_arm.SimArmClient` is
:class:`DirectArm` with a :class:`so101_sim.SimFollower` in place of the
serial bus.

Joint convention (degrees, LeRobot SO-101 calibration space):
``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper``.
Every command passes :func:`clamp_joints`, the software safety backstop.

The workshop keeps its own copy of this layer (``workshop/common/motion.py``,
where :class:`DirectArm` is called ``LeRobotArm``); ``tests/sim/test_copies.py`` fails when the two drift apart.

Independent test
----------------
    python -m vla_pipeline.robot.arm_interface --dry-run
    python -m vla_pipeline.robot.arm_interface --port /dev/ttyACM0 --id my_awesome_follower_arm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from so101_sim import MOTOR_NAMES, MOTOR_SHORT

logger = logging.getLogger(__name__)

GRIPPER_CLOSED_FLOOR = 38.0  # the jaws' physical closed point (calibration space)

# Gripper Torque_Limit (0-1000 of stall torque) for closing and opening: the
# Max_Torque_Limit LeRobot's configure writes. That register is EEPROM and
# reaches the live Torque_Limit only at power-on, so transports write this at
# connect and grips restore it before every open.
GRIPPER_TORQUE_LIMIT = 500

# Safe rest pose (degrees).
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -96.5,
    "elbow_flex": 95.9,
    "wrist_flex": 22.5,
    "wrist_roll": 86.8,
    "gripper": GRIPPER_CLOSED_FLOOR,
}

def joints_text(joints: dict) -> str:
    """``pan 0.0 lift -96.5 ...`` for log lines, in motor order."""
    return " ".join(
        f"{MOTOR_SHORT[m]} {float(joints[m]):.1f}" for m in MOTOR_NAMES if m in joints
    )

# Software joint limits (degrees) - a hard backstop every command is clamped to.
# These are the mechanical ranges pulled in by 25%, so the arm stops well
# short of the desks and monitors around the workshop rigs. Three bounds are
# relaxed just enough (rest + 2 deg) to keep REST_POSE reachable
# (shoulder_lift low, elbow_flex and wrist_roll high). The gripper span is
# the rig's calibrated jaw range, not a scaled angle.
JOINT_LIMITS = {
    "shoulder_pan": (-86.0, 86.0),
    "shoulder_lift": (-98.5, 82.5),
    "elbow_flex": (-75.0, 98.0),
    "wrist_flex": (-82.5, 67.5),
    "wrist_roll": (-75.0, 89.0),  # NOT range-swept by LeRobot - keep tight
    "gripper": (GRIPPER_CLOSED_FLOOR, 95.0),
}

# Wide envelope for IK grasps and live mimic. The desk cap above is too tight
# for a standing approach. Gripper stays on the calibrated span via JOINT_LIMITS.
_PIPELINE_BACKSTOP = {
    "shoulder_pan": (-115.0, 115.0),
    "shoulder_lift": (-120.0, 110.0),
    "elbow_flex": (-100.0, 120.0),
    "wrist_flex": (-110.0, 90.0),
    "wrist_roll": (-100.0, 100.0),
}

# IK grasps and the carry are sent in this envelope, and standing grasps are
# solved in it. The workshop's 67.5° wrist cap deletes the only approach that
# stays outside a bottle and tilts every pre-grasp; the pipeline backstop is
# the range those fit. Gripper stays on the calibrated workshop span. Other
# behaviors keep JOINT_LIMITS.
GRASP_LIMITS = {**JOINT_LIMITS, **_PIPELINE_BACKSTOP}

_LIVE_LIMIT_SCALE = 1.25


def _extended_limits() -> dict:
    out = {}
    for name, (lo, hi) in JOINT_LIMITS.items():
        if name == "gripper":  # calibrated jaw span, NOT a scaled angle:
            out[name] = (lo, hi)  # extending it would jam the jaws
            continue
        cap_lo, cap_hi = _PIPELINE_BACKSTOP.get(name, (-180.0, 180.0))
        out[name] = (
            max(lo * _LIVE_LIMIT_SCALE, cap_lo),
            min(hi * _LIVE_LIMIT_SCALE, cap_hi),
        )
    return out


# Live gesture-mimic envelope: JOINT_LIMITS extended by 25%, capped at the
# pipeline backstop, gripper kept as-is. Active ONLY while gesture mimic
# drives a REAL arm: run_mimic() switches it in for the session and restores
# the base limits afterwards; dry-run/synthetic mimic and every scripted
# behavior (dance/wave/grip/rest) stay on the base JOINT_LIMITS.
LIVE_MIMIC_LIMITS = _extended_limits()

# The envelope clamp_joints() applies when not handed an explicit one. A real
# arm server node also switches its process over (see vla_pipeline/robot/robot_node.py) so
# extended live-mimic commands are not re-clamped in transit - scripted
# behaviors are clamped to the BASE limits client-side before publish, so
# only live mimic ever uses the extra travel.
_active_limits = JOINT_LIMITS


def set_active_limits(limits: dict) -> dict:
    """Swap the default clamp envelope; returns the previous one (restore it)."""
    global _active_limits
    prev = _active_limits
    _active_limits = limits
    return prev


def clamp_joints(joints: dict, limits: dict | None = None) -> dict:
    """Clamp every joint to ``limits`` (default: the active envelope, normally
    :data:`JOINT_LIMITS`; unknown names pass ±180)."""
    lims = _active_limits if limits is None else limits
    out = {}
    for name, value in joints.items():
        lo, hi = lims.get(name, (-180.0, 180.0))
        out[name] = float(min(max(value, lo), hi))
    return out


# ----------------------------------------------------------------------------
# Servo motion state
# ----------------------------------------------------------------------------

# STS3215 Status register (address 65): bit 5 is the overload protection.
OVERLOAD_BIT = 0x20


class MotionTimeout(RuntimeError):
    """The arm's owner (the servos, or the controllers) never reported a goal finished within its budget."""


def settled(state: dict, joints: list) -> bool:
    """True when no servo in ``joints`` reports its profile still running."""
    return not any(state["moving"].get(j, False) for j in joints)


# ----------------------------------------------------------------------------
# ArmClient ABC + transports
# ----------------------------------------------------------------------------


class ArmClient(ABC):
    """Transport-agnostic arm API used by every behavior."""

    # Set True by transports whose relative-target clamp is disabled
    # (``max_relative_target: null`` - the workshop config). In that mode a
    # command is ONE absolute goal write and the servo's onboard profile
    # produces the motion, so go_to() must NOT stream interpolated setpoints:
    # re-writing a goal every tick restarts that profile (the 02_robot
    # lesson) and the move turns twitchy. It delegates to move_to() instead -
    # the same behavior as the real pipeline's arm interface.
    _direct_goal: bool = False

    # Limits for the command currently being sent. None keeps JOINT_LIMITS.
    # A standing grasp sets this to GRASP_LIMITS for the move, so send_joints
    # does not pull a 90° wrist back to the workshop's 67.5° cap.
    _send_limits: dict | None = None

    # True when commands (may) reach real hardware. Gesture mimic uses this to
    # decide whether the 25%-extended LIVE_MIMIC_LIMITS envelope applies;
    # simulated arms stay on the base JOINT_LIMITS.
    is_live: bool = False

    # Last Torque_Limit sent to the gripper; displays scale the grasp level by it.
    gripper_torque_limit: int = GRIPPER_TORQUE_LIMIT

    @abstractmethod
    def send_joints(self, joints: dict) -> None:
        """Command absolute joint targets (degrees); partial dicts allowed."""

    @abstractmethod
    def get_joints(self) -> dict:
        """Read current joint positions (degrees)."""

    def get_gripper_load(self):
        """Gripper servo effort/current, or ``None`` when not readable."""
        return None

    def set_gripper_torque_limit(self, value: int) -> None:  # noqa: B027 - optional override
        """Cap the gripper at ``value`` (0-1000) of stall torque: the servo's
        RAM ``Torque_Limit``. A no-op on transports without one."""

    def grip_report(self):
        """What touches the fingers, where the transport can tell (the twin)."""
        return None

    def go_to(
        self, target: dict, duration_s: float = 2.0, fps: float = 30.0, stop_check=None,
        limits: dict | None = None, max_step_deg: float = 0.0,
    ) -> None:
        """Move to ``target`` (partial dicts allowed).

        ``stop_check`` (callable → bool) aborts mid-way - the safety stop
        polls it every tick, so long moves react immediately.

        Streaming transports get a host-interpolated trajectory over
        ``duration_s``. ``max_step_deg`` (0 = off) caps each tick so the
        command never jumps farther than that from the measured position.
        Direct-goal transports (``_direct_goal``) delegate to :meth:`move_to`:
        the goal is sent once and the servo profiles the move, so
        ``duration_s``/``fps``/``max_step_deg`` no longer apply.
        """
        if self._direct_goal:
            return self.move_to(target, stop_check=stop_check, limits=limits)
        previous = self._send_limits
        self._send_limits = limits
        try:
            start = self.get_joints()
            target = clamp_joints({**start, **target}, limits)
            steps = max(int(duration_s * fps), 1)
            deltas = {k: target[k] - start[k] for k in target}
            interval = 1.0 / fps
            for i in range(1, steps + 1):
                if stop_check and stop_check():
                    return
                a = i / steps
                desired = {k: start[k] + a * deltas[k] for k in target}
                if max_step_deg > 0:
                    actual = self.get_joints()
                    pose = {}
                    for k, goal in desired.items():
                        cur = actual.get(k, goal)
                        delta = goal - cur
                        if abs(delta) > max_step_deg:
                            goal = cur + max_step_deg * (1.0 if delta > 0 else -1.0)
                        pose[k] = goal
                else:
                    pose = desired
                self.send_joints(pose)
                time.sleep(interval)
        finally:
            self._send_limits = previous

    def move_to(
        self, target: dict, tolerance_deg: float = 2.5, stop_check=None, limits: dict | None = None,
    ) -> str:
        """Send an absolute goal ONCE and wait until the servos report it finished.

        The servo's onboard profile produces the motion. The move is over when
        every joint in ``target`` reports its profile done (the STS3215
        ``Moving`` register), which :meth:`_await_motion` waits for once,
        bounded by the profile's predicted duration. Returns ``"arrived"``
        (body joints within ``tolerance_deg``), ``"blocked"`` (the servos
        finished away from the goal: a limit, a contact, an overload) or
        ``"stopped"`` (``stop_check`` fired; the arm halts where it is). The
        gripper's open-fraction readback need not converge, so a gripper-only
        move arrives when its servo finishes. Raises :class:`MotionTimeout`
        when the servos never report the goal finished.
        """
        previous = self._send_limits
        self._send_limits = limits
        try:
            target = clamp_joints(target, limits)
            sent_at = time.time()
            self.send_joints(target)
            how, state = self._await_motion(list(target), sent_at, stop_check)
            joints = state["joints"]
            body = [k for k in target if k != "gripper"]
            if how == "stopped":
                self.send_joints(joints)  # halt here, don't finish the move
            elif body and max(abs(joints.get(k, target[k]) - target[k]) for k in body) > tolerance_deg:
                how = "blocked"
            else:
                how = "arrived"
            worst = max(body or list(target), key=lambda k: abs(joints.get(k, target[k]) - target[k]))
            overload = [k for k, v in state.get("overload", {}).items() if v]
            logger.log(
                logging.WARNING if how == "blocked" else logging.INFO,
                "move_to %s in %.2f s (servo profile %.2f s): %.1f deg left on %s%s",
                how, time.time() - sent_at, float(state.get("eta_s", 0.0)),
                abs(joints.get(worst, target[worst]) - target[worst]), worst,
                "" if how == "arrived" else (
                    f" | at {joints_text(joints)} | goal {joints_text(target)}"
                    f" | overload {', '.join(overload) or 'none'}"
                ),
            )
            return how
        finally:
            self._send_limits = previous

    def _await_motion(self, joints: list, sent_at: float, stop_check) -> tuple[str, dict]:
        """``("settled" | "stopped", state)`` once the servos in ``joints`` finish the goal
        sent at ``sent_at``. ``state`` carries ``joints``, ``moving``, ``overload`` and
        ``eta_s``. Each transport waits on its own bus owner; there is no default."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot report when a move finishes: drive the arm through "
            "the arm server node (robot.use_ros2: true), which owns the bus and reports it"
        )

    def state_time(self) -> float:
        """The arm owner's clock (seconds) at its newest state."""
        raise NotImplementedError(f"{type(self).__name__} has no owner reporting the arm's state")

    def wait_state(self, t: float, stop_check=None) -> float | None:
        """Block until the arm's owner reports a state at or after ``t`` on its
        clock (:meth:`state_time`). Returns that state's time, or None once
        ``stop_check`` fires. Loops that sample the arm pace themselves on it,
        so they never run ahead of the states they judge."""
        raise NotImplementedError(f"{type(self).__name__} has no owner reporting the arm's state")

    def go_to_rest(self, duration_s: float = 3.0, stop_check=None) -> None:
        """Bring the whole arm to :data:`REST_POSE`."""
        self.go_to(REST_POSE, duration_s=duration_s, stop_check=stop_check)

    def close(self) -> None:  # noqa: B027 - optional override
        pass


class DryRunArm(ArmClient):
    """No-hardware arm: commands are integrated so behaviors run anywhere."""

    is_live = False  # simulated - mimic keeps the base JOINT_LIMITS

    last_eta_s = 0.0  # commands take effect at once

    def __init__(self, verbose: bool = False):
        self._joints = dict(REST_POSE)
        self._verbose = verbose
        self.goal = dict(REST_POSE)
        self._clock = 0.0  # commands take effect at once, so any time is already reached

    def send_joints(self, joints: dict) -> None:
        self._joints.update(clamp_joints(joints, self._send_limits))
        self.goal = dict(self._joints)
        if self._verbose:
            pretty = " ".join(f"{k}={v:6.1f}" for k, v in self._joints.items())
            print(f"\r[dry-run] {pretty}", end="", flush=True)

    def get_joints(self) -> dict:
        return dict(self._joints)

    def read_motion_state(self) -> dict:
        """No servos: nothing is ever still moving, nothing ever overloads."""
        return {"moving": dict.fromkeys(MOTOR_NAMES, False), "overload": dict.fromkeys(MOTOR_NAMES, False)}

    def _await_motion(self, joints: list, sent_at: float, stop_check) -> tuple[str, dict]:
        return "settled", {"joints": dict(self._joints), **self.read_motion_state(), "eta_s": 0.0}

    def state_time(self) -> float:
        return self._clock

    def wait_state(self, t: float, stop_check=None) -> float | None:
        if stop_check is not None and stop_check():
            return None
        self._clock = max(self._clock, t)
        return self._clock


# Transient Feetech errors ("There is no status packet!") show up when several
# joints accelerate at once. LeRobot's send_action/get_observation retry count
# defaults to 0, so the transport retries the whole call.
BUS_RETRIES = 3
BUS_RETRY_DELAY_S = 0.004

# Direct-goal mode (max_relative_target is null): one Goal_Position write, and
# the servo profiles the move at the Goal_Velocity and Acceleration that
# vla_pipeline.robot.servo_profile plans for it. Both registers are SRAM (addresses 46 and
# 41), so a write per move does not wear the servo's EEPROM.
_SPEED_REGISTERS = ("Goal_Velocity",)
_ACCEL_REGISTERS = ("Acceleration",)
# The server node's control loop rate (robot.control_fps), read by the server and its clients.
DEFAULT_CONTROL_HZ = 60.0


class DirectArm(ArmClient):
    """Direct Feetech serial transport via LeRobot ``SO101Follower``.

    Only ONE process may own the bus - on the full system that is the ROS 2
    server node (:mod:`vla_pipeline.robot.robot_node`), which wraps this class. Effort readback
    is best-effort: ``None`` on firmware that can't provide it.
    """

    is_live = True  # real servos - live mimic may use LIVE_MIMIC_LIMITS

    _EFFORT_REGISTERS = ("Present_Current", "Present_Load")

    def __init__(self, port: str, robot_id: str, max_relative_target=8.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=port, id=robot_id, cameras={}, max_relative_target=max_relative_target
        )
        if hasattr(config, "disable_torque_on_disconnect"):
            config.disable_torque_on_disconnect = False
        self._robot = SO101Follower(config)
        self._robot.connect()
        logger.info("SO101Follower connected on %s (id=%s)", port, robot_id)
        self._init_transport(max_relative_target)

    def _init_transport(self, max_relative_target) -> None:
        """State shared by this transport and the twin's (``SimArmClient``)."""
        self._effort_register = False  # False = not probed yet
        self.goal: dict = {}
        self.last_eta_s = 0.0
        # Direct-goal mode (config: max_relative_target: null): LeRobot's
        # per-tick slew clamp is OFF, a command is ONE absolute Goal_Position
        # write, and the servo's onboard controller produces the motion.
        self._direct_goal = max_relative_target is None
        if self._direct_goal:
            from vla_pipeline.robot.servo_profile import shared_profiler

            self._profiler = shared_profiler()
            static = self._profiler.static
            self._sync_write_guarded(_SPEED_REGISTERS, {m: static[m][0] for m in MOTOR_NAMES}, "speed")
            self._sync_write_guarded(_ACCEL_REGISTERS, {m: static[m][1] for m in MOTOR_NAMES}, "acceleration")
        self.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)

    def _bus_retry(self, op, what: str):
        """Run a bus transaction, retrying briefly on a dropped status packet."""
        last_err: Exception | None = None
        for attempt in range(1, BUS_RETRIES + 1):
            try:
                return op()
            except Exception as e:
                last_err = e
                if attempt < BUS_RETRIES:
                    logger.debug(
                        "%s failed (attempt %d/%d): %s - retrying",
                        what, attempt, BUS_RETRIES, e,
                    )
                    time.sleep(BUS_RETRY_DELAY_S)
        raise last_err  # type: ignore[misc]

    def _sync_write_guarded(self, candidates: tuple, values: dict, what: str) -> bool:
        """sync_write per-joint raw values, trying each register name."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return False
        payload = {m: max(int(v), 1) for m, v in values.items()}
        for name in candidates:
            for _attempt in range(BUS_RETRIES):
                try:
                    bus.sync_write(name, payload, normalize=False)
                    return True
                except Exception:
                    time.sleep(BUS_RETRY_DELAY_S)
        logger.warning("Could not set per-joint %s - coordination may be off.", what)
        return False

    def _plan_start(self) -> dict:
        """Where the servos' next profile starts: their measured position."""
        return self.get_joints()

    def send_joints(self, joints: dict) -> None:
        joints = clamp_joints(joints, self._send_limits)
        if self._direct_goal:
            # From where the servos are now: another caller may have moved them since our last read.
            plan = self._profiler.plan(self._plan_start(), joints)
            self._sync_write_guarded(_SPEED_REGISTERS, plan.speed_raw, "speed")
            self._sync_write_guarded(_ACCEL_REGISTERS, plan.accel_raw, "acceleration")
            self.last_eta_s = plan.eta_s
        action = {f"{k}.pos": v for k, v in joints.items()}
        self._bus_retry(lambda: self._robot.send_action(action), "send_action")
        self.goal = {**self.goal, **joints}

    def get_joints(self) -> dict:
        obs = self._bus_retry(self._robot.get_observation, "get_observation")
        return {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES if f"{m}.pos" in obs}

    def read_motion_state(self) -> dict:
        """Each servo's ``Moving`` flag and the overload bit of its ``Status``.

        Not yet checked on the real arm: whether the STS3215 sets ``Moving``
        from the ``Goal_Position`` write until its profile ends (as the twin
        does) or from its measured speed. If it is the speed, a wait must also
        see the flag rise before it accepts it falling.
        """
        bus = self._robot.bus
        moving = self._bus_retry(lambda: bus.sync_read("Moving", normalize=False), "read Moving")
        status = self._bus_retry(lambda: bus.sync_read("Status", normalize=False), "read Status")
        return {
            "moving": {m: bool(int(v)) for m, v in moving.items()},
            "overload": {m: bool(int(v) & OVERLOAD_BIT) for m, v in status.items()},
        }

    def get_gripper_load(self):
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return None
        if self._effort_register is False:  # probe once
            self._effort_register = None
            for reg in self._EFFORT_REGISTERS:
                try:
                    bus.read(reg, "gripper", normalize=False)
                    self._effort_register = reg
                    break
                except Exception:
                    continue
        if self._effort_register is None:
            return None
        try:
            return abs(
                float(bus.read(self._effort_register, "gripper", normalize=False))
            )
        except Exception:
            return None

    def set_gripper_torque_limit(self, value: int) -> None:
        """Write the gripper's ``Torque_Limit`` (address 48, RAM). LeRobot's
        configure writes ``Max_Torque_Limit`` (16), which is EEPROM and only
        reaches 48 at power-on."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return
        v = int(min(max(int(value), 0), 1000))
        for _ in range(3):
            try:
                bus.write("Torque_Limit", "gripper", v, normalize=False)
                self.gripper_torque_limit = v
                return
            except Exception as e:
                err = e
                time.sleep(0.004)
        logger.warning("could not set the gripper Torque_Limit to %d: %s", v, err)

    def close(self) -> None:
        try:
            # Torque stays on through disconnect, so the servos' own profile
            # carries a direct goal to rest and the arm holds it instead of going limp.
            if self._direct_goal:
                self.send_joints(REST_POSE)
            else:
                self.go_to_rest()
        finally:
            self._robot.disconnect()


# How often the bus owner reads the gripper's current (an extra transaction).
_EFFORT_EVERY_N_TICKS = 3
# A goal is written on the owner's next tick and its servos report on the ones
# after; this bound only catches an owner that has stopped reporting.
OWNER_REPLY_S = 2.0


class BusOwnerArm(ArmClient):
    """The bus arm owned by this process (``robot.use_ros2: false``), as the
    server node owns it on the rig: one thread writes each new goal and reads
    the servos every control tick, and moves wait on what it reads - their
    ``Moving`` and overload flags."""

    def __init__(self, bus_arm: ArmClient, control_fps: float = DEFAULT_CONTROL_HZ):
        self._bus = bus_arm
        self.is_live = bus_arm.is_live
        self._direct_goal = getattr(bus_arm, "_direct_goal", False)
        self._interval = 1.0 / float(control_fps)
        self._cond = threading.Condition()
        self._goal: dict | None = None  # the next goal to write
        self._torque: int | None = None  # the next gripper Torque_Limit to write
        self._state: dict | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bus-owner")
        self._thread.start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._state is not None, OWNER_REPLY_S):
                self._running = False
                raise MotionTimeout(f"the bus owner read no state within {OWNER_REPLY_S:.1f} s")

    def _loop(self) -> None:
        load, cmd_time, eta_s, tick, errors = None, 0.0, 0.0, 0, 0
        while self._running:
            started = time.perf_counter()
            with self._cond:
                goal, self._goal = self._goal, None
                torque, self._torque = self._torque, None
            try:
                if torque is not None:
                    self._bus.set_gripper_torque_limit(torque)
                    self.gripper_torque_limit = torque
                if goal is not None:
                    self._bus.send_joints(goal)
                    cmd_time, eta_s = time.time(), float(getattr(self._bus, "last_eta_s", 0.0))
                joints = self._bus.get_joints()
                motion = self._bus.read_motion_state()
                tick += 1
                if tick % _EFFORT_EVERY_N_TICKS == 0:
                    load = self._bus.get_gripper_load()
                with self._cond:
                    self._state = {
                        "t": time.monotonic(), "cmd_time": cmd_time, "eta_s": eta_s, "joints": joints,
                        "load": load, **motion,
                    }
                    self._cond.notify_all()
                errors = 0
            except Exception as e:  # a transient bus error must not end the loop
                errors += 1
                if errors % 30 == 1:
                    logger.warning("bus tick failed (x%d), continuing: %s", errors, e)
            time.sleep(max(self._interval - (time.perf_counter() - started), 0.0))

    def send_joints(self, joints: dict) -> None:
        joints = clamp_joints(joints, self._send_limits)
        with self._cond:
            self._goal = {**(self._goal or {}), **joints}

    def get_joints(self) -> dict:
        with self._cond:
            return dict(self._state["joints"])

    def get_gripper_load(self):
        with self._cond:
            return self._state["load"]

    def read_motion_state(self) -> dict:
        with self._cond:
            return {"moving": dict(self._state["moving"]), "overload": dict(self._state["overload"])}

    def set_gripper_torque_limit(self, value: int) -> None:
        with self._cond:
            self._torque = int(min(max(int(value), 0), 1000))

    def _await_motion(self, joints: list, sent_at: float, stop_check) -> tuple[str, dict]:
        """The owner's first report that it wrote the goal sent at ``sent_at``,
        then its report that the servos in ``joints`` finished it."""

        def report() -> dict:
            return {**self._state, "joints": dict(self._state["joints"])}

        with self._cond:
            written = self._cond.wait_for(
                lambda: self._state["cmd_time"] >= sent_at or bool(stop_check and stop_check()), OWNER_REPLY_S,
            )
            if not written:
                raise MotionTimeout(f"the bus owner did not write the goal within {OWNER_REPLY_S:.1f} s")
            budget = float(self._state["eta_s"]) + OWNER_REPLY_S
            done = self._cond.wait_for(
                lambda: bool(stop_check and stop_check()) or settled(self._state, joints), budget,
            )
            if stop_check and stop_check():
                return "stopped", report()
            if not done:
                raise MotionTimeout(f"servos {joints} still moving {budget:.2f} s after the goal: {self._state}")
            return "settled", report()

    def state_time(self) -> float:
        with self._cond:
            return self._state["t"]

    def wait_state(self, t: float, stop_check=None) -> float | None:
        budget = max(t - time.monotonic(), 0.0) + OWNER_REPLY_S
        with self._cond:
            ok = self._cond.wait_for(lambda: self._state["t"] >= t or bool(stop_check and stop_check()), budget)
            if stop_check and stop_check():
                return None
            if not ok:
                raise MotionTimeout(f"the bus owner read no state within {budget:.2f} s")
            return self._state["t"]

    def grip_report(self):
        return self._bus.grip_report()

    def close(self) -> None:
        self._running = False
        self._thread.join()
        self._bus.close()


def bus_owner(arm: ArmClient, cfg: dict) -> ArmClient:
    """``arm`` with an owner in this process: a real bus arm gets one
    (:class:`BusOwnerArm`); the twin's physics and the dry-run arm are their own."""
    if type(arm) is not DirectArm:
        return arm
    return BusOwnerArm(arm, float((cfg.get("robot") or {}).get("control_fps", DEFAULT_CONTROL_HZ)))


def make_bus_arm(cfg: dict) -> ArmClient:
    """The bus-owning transport: the real Feetech bus, or the digital twin
    when ``robot.backend: sim``."""
    r = cfg["robot"]
    if r.get("backend", "real") == "sim":
        from vla_pipeline.robot.sim_arm import SimArmClient

        return SimArmClient(cfg)
    return DirectArm(r["motor_port"], r["robot_id"], r.get("max_relative_target", 8.0))


def make_arm(cfg: dict) -> ArmClient:
    """Build the ArmClient described by ``cfg['robot']``."""
    r = cfg["robot"]
    if not r.get("enable_motors", True):
        return DryRunArm()
    if r.get("use_ros2", False):
        from vla_pipeline.robot.robot_node import Ros2ArmClient

        return Ros2ArmClient(cfg)
    return bus_owner(make_bus_arm(cfg), cfg)


# =============================================================================
# Calibration
# =============================================================================
# LeRobot stores per-robot calibration at
#   ~/.cache/huggingface/lerobot/calibration/robots/<robot_type>/<robot_id>.json
# Calibration is INTERACTIVE - it prompts you to move the arm and press ENTER -
# so it can only run in a foreground terminal, never in the backgrounded server
# node. The launcher detects a missing file and runs calibrate_arm() first.


def calibration_path(robot_id: str) -> Path | None:
    """Return the existing calibration file for ``robot_id``, or None."""
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
    hits = sorted(base.glob(f"*/{robot_id}.json"))
    return hits[0] if hits else None


def calibration_exists(robot_id: str) -> bool:
    """Return True if a calibration file exists for robot_id."""
    return calibration_path(robot_id) is not None


# Feetech STS3215 encoder resolution (12-bit). LeRobot normalizes degrees as
# ``(raw - mid) * 360 / (resolution - 1)`` with ``mid = (range_min+range_max)/2``.
_STS3215_RESOLUTION = 4096
_DEG_PER_COUNT = 360.0 / (_STS3215_RESOLUTION - 1)

# The gripper uses 0..100 normalization; body joints run in DEGREES mode.
# See lerobot so_follower + motors_bus._normalize.
_GRIPPER_NAME = "gripper"


def load_calibration(robot_id: str) -> dict[str, dict] | None:
    """Load and parse the LeRobot calibration JSON for ``robot_id``.

    Returns a dict ``{motor_name: {id, drive_mode, homing_offset, range_min,
    range_max}}`` or ``None`` if no calibration file exists / it can't be read.
    """
    path = calibration_path(robot_id)
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read calibration %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Unexpected calibration format in %s", path)
        return None
    return data


def calibration_limits_and_rest(
    robot_id: str,
    full_range_eps: int = 3500,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], dict[str, dict]]:
    """Derive per-joint software limits and a rest pose from the calibration file.

    The SO-101 follower runs body joints in DEGREES mode: the calibrated zero is
    the **midpoint** of the recorded raw range, and the usable travel each way is
    ``±(range_max - range_min)/2`` converted to degrees. So:

        rest[joint]  = 0.0                         (midpoint maps to 0°)
        limits[joint] = (-half_span_deg, +half_span_deg)

    The gripper uses 0..100 normalization, so its rest is the 50 midpoint and its
    limits are (0, 100).

    A joint that was **not range-swept** during calibration (notably
    ``wrist_roll`` - LeRobot deliberately skips it) keeps the near-full encoder
    range, giving ≈±180°; we detect that (span > ``full_range_eps`` counts) and
    flag it via the returned ``meta`` so callers can warn that its rest/limits
    come from the default range, not a real sweep.

    Returns ``(limits, rest, meta)`` where ``meta[joint]`` is the raw calibration
    entry augmented with ``span_counts`` and ``swept`` (bool).
    """
    cal = load_calibration(robot_id)
    limits: dict[str, tuple[float, float]] = {}
    rest: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if not cal:
        return limits, rest, meta

    for name, entry in cal.items():
        try:
            rmin = float(entry["range_min"])
            rmax = float(entry["range_max"])
        except (KeyError, TypeError, ValueError):
            continue
        span_counts = abs(rmax - rmin)
        swept = span_counts <= full_range_eps  # full encoder range => not swept
        if name == _GRIPPER_NAME:
            # 0..100 normalization: midpoint is 50, range is the full [0,100].
            limits[name] = (0.0, 100.0)
            rest[name] = 50.0
        else:
            half_span_deg = 0.5 * span_counts * _DEG_PER_COUNT
            limits[name] = (-half_span_deg, half_span_deg)
            rest[name] = 0.0  # DEGREES mode: calibrated midpoint == 0°
        meta[name] = {**entry, "span_counts": span_counts, "swept": swept}
    return limits, rest, meta


def calibrate_arm(cfg: dict, force: bool = False) -> None:
    """Run LeRobot's interactive SO-101 calibration (foreground / TTY only).

    Connecting already triggers calibration when the file is missing or
    mismatched; ``force`` re-runs it even when a valid file exists. Either way
    this must run in a real terminal because it calls ``input()`` to step you
    through moving the arm.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Calibration needs an interactive terminal (it asks you to move the "
            "arm and press ENTER) but stdin is not a TTY. Run it in the "
            "foreground:\n    python -m vla_pipeline.robot.arm_interface --calibrate"
        )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    r = cfg["robot"]
    existing = calibration_path(r["robot_id"])
    if existing and not force:
        logger.info(
            "Calibration already exists at %s - nothing to do "
            "(use --calibrate --force to redo).",
            existing,
        )
        return

    config = SO101FollowerConfig(
        port=r["motor_port"],
        id=r["robot_id"],
        cameras={},
        max_relative_target=r.get("max_relative_target", 8.0),
    )
    robot = SO101Follower(config)
    print(f"\nCalibrating SO-101 '{r['robot_id']}' on {r['motor_port']}.")
    print("Follow the prompts: move the arm as instructed and press ENTER.\n")
    robot.connect()  # auto-calibrates when the file is missing/mismatched
    if force:
        robot.calibrate()  # explicit recalibration even if a file existed
    robot.disconnect()
    saved = calibration_path(r["robot_id"])
    print(f"\nCalibration complete: {saved}")


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: component-test the arm, show calibration, or run interactive calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 arm interface - component test / calibration"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="no hardware, just print commands"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run interactive calibration (foreground) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --calibrate: recalibrate even if a file exists",
    )
    parser.add_argument(
        "--config", default=None, help="pipeline.yaml (for --calibrate)"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="print calibration-derived limits/rest for --id and exit",
    )
    args = parser.parse_args()

    if args.show_calibration:
        limits, rest, meta = calibration_limits_and_rest(args.id)
        if not limits:
            print(f"No calibration found for '{args.id}'.")
            return
        print(f"Calibration-derived limits/rest for '{args.id}':")
        for m in MOTOR_NAMES:
            if m in limits:
                lo, hi = limits[m]
                swept = meta[m].get("swept", True)
                tag = "" if swept else "  (NOT range-swept - default range)"
                print(
                    f"  {m:<14} limits=({lo:7.1f}, {hi:7.1f})  rest={rest[m]:6.1f}{tag}"
                )
        return

    if args.calibrate:
        from vla_pipeline.utils.config import load_config

        calibrate_arm(load_config(args.config), force=args.force)
        return

    arm: ArmClient = DryRunArm() if args.dry_run else DirectArm(args.port, args.id)
    try:
        print("Current joints:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Wiggling gripper (open → close → open)...")
        arm.go_to({"gripper": 60.0}, duration_s=1.5)
        arm.go_to({"gripper": 5.0}, duration_s=1.5)
        arm.go_to({"gripper": 30.0}, duration_s=1.0)
        print("\nReturning to rest pose...")
        arm.go_to_rest()
        print("Arm interface test PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
