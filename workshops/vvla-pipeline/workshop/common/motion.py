# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Motion math and arm plumbing (PROVIDED).

Joint conventions, safety clamps, the One Euro filter, the camera-frame →
joint mapper, the dry-run arm, and the LeRobot serial transport. None of it
is AMD- or ROS-specific, so none of it is a TODO - but the *conventions*
matter everywhere you wire things up:

- Joints are **degrees** in LeRobot SO-101 calibration space, keyed by
  :data:`MOTOR_NAMES`. Partial dicts are allowed everywhere.
- Every command must pass through :func:`clamp_joints` before it reaches a
  transport - this is the software safety backstop.
- ``gripper`` runs on the rig's calibrated 43-95 open span (see
  :data:`JOINT_LIMITS`); 43 is the jaws' physical closed point - commanding
  below it jams them into the mechanical stop and stalls the servo - and
  :data:`GRIPPER_CLOSED_FLOOR` pins that value.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_CLOSED_FLOOR = 38.0  # the jaws' physical closed point (calibration space)

# Safe rest pose (degrees) - same values the real pipeline parks at.
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -96.5,
    "elbow_flex": 95.9,
    "wrist_flex": 22.5,
    "wrist_roll": 86.8,
    "gripper": GRIPPER_CLOSED_FLOOR,
}

# Software joint limits (degrees) - a hard backstop every command is clamped to.
# These are the mechanical ranges pulled in by 25%, so the arm stops well
# short of the desks and monitors around the workshop rigs. Three bounds are
# relaxed just enough (rest + 2 deg) to keep REST_POSE reachable
# (shoulder_lift low, elbow_flex and wrist_roll high). The gripper span is
# the rig's calibrated jaw range - 43 = physically closed, 95 = fully open -
# not a scaled angle.
JOINT_LIMITS = {
    "shoulder_pan": (-86.0, 86.0),
    "shoulder_lift": (-98.5, 82.5),
    "elbow_flex": (-75.0, 98.0),
    "wrist_flex": (-82.5, 67.5),
    "wrist_roll": (-75.0, 89.0),  # NOT range-swept by LeRobot - keep tight
    "gripper": (38.0, 95.0),  # calibrated jaws: 43 closed, 95 open
}

# Hard backstop - the widest envelope the deployed system ever allows. Used
# only to cap the live-mimic extension below.
_PIPELINE_BACKSTOP = {
    "shoulder_pan": (-115.0, 115.0),
    "shoulder_lift": (-120.0, 110.0),
    "elbow_flex": (-100.0, 120.0),
    "wrist_flex": (-110.0, 90.0),
    "wrist_roll": (-100.0, 100.0),
}

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
# arm server node also switches its process over (see ros/arm_node.py) so
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

    # True when commands (may) reach real hardware. Gesture mimic uses this to
    # decide whether the 25%-extended LIVE_MIMIC_LIMITS envelope applies;
    # simulated arms stay on the base JOINT_LIMITS.
    is_live: bool = False

    @abstractmethod
    def send_joints(self, joints: dict) -> None:
        """Command absolute joint targets (degrees); partial dicts allowed."""

    @abstractmethod
    def get_joints(self) -> dict:
        """Read current joint positions (degrees)."""

    def get_gripper_load(self):
        """Gripper servo effort/current, or ``None`` when not readable."""
        return None

    def go_to(
        self, target: dict, duration_s: float = 2.0, fps: float = 30.0, stop_check=None
    ) -> None:
        """Move to ``target`` (partial dicts allowed).

        ``stop_check`` (callable → bool) aborts mid-way - the safety stop
        polls it every tick, so long moves react immediately.

        Streaming transports get a host-interpolated trajectory over
        ``duration_s``. Direct-goal transports (``_direct_goal``) delegate to
        :meth:`move_to`: the goal is sent once and the servo interpolates in
        firmware, so ``duration_s``/``fps`` no longer apply.
        """
        if self._direct_goal:
            return self.move_to(
                target, timeout_s=max(duration_s * 2.0, 2.0), stop_check=stop_check
            )
        start = self.get_joints()
        target = clamp_joints({**start, **target})
        steps = max(int(duration_s * fps), 1)
        deltas = {k: target[k] - start[k] for k in target}
        interval = 1.0 / fps
        for i in range(1, steps + 1):
            if stop_check and stop_check():
                return
            a = i / steps
            self.send_joints({k: start[k] + a * deltas[k] for k in target})
            time.sleep(interval)

    def move_to(
        self,
        target: dict,
        timeout_s: float = 6.0,
        tolerance_deg: float = 2.5,
        poll_hz: float = 20.0,
        progress_eps_deg: float = 1.0,
        settle_dwell_s: float = 0.35,
        gripper_settle_s: float = 1.0,
        stop_check=None,
    ) -> None:
        """Send an absolute goal ONCE and wait until the arm settles.

        Unlike streaming, this lets the servo's onboard controller produce
        the motion - smooth and continuous instead of host-stepped. "Settled"
        is judged on the body joints only ("within ``tolerance_deg``", or "no
        longer making progress" - a joint straining at a range limit never
        reads exactly on target). The gripper's open-fraction readback needn't converge,
        so a gripper-only move just gets ``gripper_settle_s`` to travel. ``stop_check`` aborts and halts the arm where it is.
        """
        target = clamp_joints(target)
        self.send_joints(target)
        wait = [k for k in target if k != "gripper"]
        if not wait:  # gripper-only move
            time.sleep(gripper_settle_s)
            return

        def _dist() -> float:
            cur = self.get_joints()
            return max(
                (abs(cur.get(k, target[k]) - target[k]) for k in wait), default=0.0
            )

        deadline = time.time() + timeout_s
        interval = 1.0 / poll_hz
        ref_dist, ref_time, started = _dist(), time.time(), False
        while time.time() < deadline:
            if stop_check and stop_check():
                try:  # halt here, don't finish the move
                    self.send_joints(self.get_joints())
                except Exception:
                    pass
                return
            time.sleep(interval)
            dist = _dist()
            if dist <= tolerance_deg:  # arrived
                return
            if ref_dist - dist > progress_eps_deg:
                started = True  # still closing in - keep waiting
                ref_dist, ref_time = dist, time.time()
            elif time.time() - ref_time >= settle_dwell_s:
                if started:  # moved, then stopped: settled
                    return
                self.send_joints(target)  # never moved: command may have
                ref_dist, ref_time = (
                    dist,
                    time.time(),
                )  # dropped - resend once per window

    def go_to_rest(self, duration_s: float = 3.0, stop_check=None) -> None:
        """Bring the whole arm to :data:`REST_POSE`."""
        self.go_to(REST_POSE, duration_s=duration_s, stop_check=stop_check)

    def close(self) -> None:  # noqa: B027 - optional override
        pass


class DryRunArm(ArmClient):
    """No-hardware arm: commands are integrated so behaviors run anywhere."""

    is_live = False  # simulated - mimic keeps the base JOINT_LIMITS

    def __init__(self, verbose: bool = False):
        self._joints = dict(REST_POSE)
        self._verbose = verbose

    def send_joints(self, joints: dict) -> None:
        self._joints.update(clamp_joints(joints))
        if self._verbose:
            pretty = " ".join(f"{k}={v:6.1f}" for k, v in self._joints.items())
            print(f"\r[dry-run] {pretty}", end="", flush=True)

    def get_joints(self) -> dict:
        return dict(self._joints)


class LeRobotArm(ArmClient):
    """Direct Feetech serial transport via LeRobot ``SO101Follower``.

    Only ONE process may own the bus - on the full system that is the ROS 2
    server node (``ros/arm_node.py``), which wraps this class. Effort readback
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
        self._effort_register = False  # False = not probed yet
        logger.info("SO101Follower connected on %s (id=%s)", port, robot_id)

        # Direct-goal mode (config: max_relative_target: null - the pipeline's
        # setting, and how gesture_mimic drives the arm): LeRobot's per-tick
        # slew clamp is OFF, a command is ONE absolute Goal_Position write, and
        # the servo's onboard controller produces the motion. Give the servos a
        # sane speed/acceleration profile so that single write is smooth rather
        # than a full-speed slam. Register names vary across Feetech firmware /
        # LeRobot versions, so the writes are guarded best-effort.
        self._direct_goal = max_relative_target is None
        if self._direct_goal:
            self._set_motion_profile(speed=1000, acceleration=50)

    def _set_motion_profile(self, speed=None, acceleration=None) -> None:
        """Best-effort servo speed/accel setup for direct-goal mode."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return

        def _write(candidates, value, what):
            for name in candidates:
                try:
                    bus.sync_write(
                        name, {m: int(value) for m in MOTOR_NAMES}, normalize=False
                    )
                    logger.info("servo %s set to %d (register '%s')", what, value, name)
                    return
                except Exception:
                    continue
            logger.warning(
                "could not set servo %s - using the firmware "
                "default; moves may be faster/abrupter",
                what,
            )

        if acceleration is not None:
            _write(
                ("Maximum_Acceleration", "Acceleration"), acceleration, "acceleration"
            )
        if speed is not None:
            _write(
                ("Goal_Velocity", "Maximum_Speed_Limit", "Goal_Speed"), speed, "speed"
            )

    def send_joints(self, joints: dict) -> None:
        joints = clamp_joints(joints)
        self._robot.send_action({f"{k}.pos": v for k, v in joints.items()})

    def get_joints(self) -> dict:
        obs = self._robot.get_observation()
        return {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES if f"{m}.pos" in obs}

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

    def close(self) -> None:
        try:
            if getattr(self, "_direct_goal", False):
                # One write; the servo profiles its own way to rest (re-writing
                # the goal restarts that profile). Torque stays on through
                # disconnect, so the arm holds the pose instead of going limp.
                self.send_joints(REST_POSE)
                time.sleep(1.0)
            else:
                self.go_to_rest()
        finally:
            self._robot.disconnect()


def make_arm_backend(cfg: dict, dry_run: bool = False) -> ArmClient:
    """DryRunArm, or the real LeRobot serial arm per ``robot:`` config."""
    r = cfg.get("robot", {})
    if dry_run or not r.get("enable_motors", True):
        return DryRunArm()
    return LeRobotArm(r["motor_port"], r["robot_id"], r.get("max_relative_target", 8.0))


# ----------------------------------------------------------------------------
# One Euro smoothing
# ----------------------------------------------------------------------------


class _LowPass:
    def __init__(self):
        self._y = None

    def __call__(self, x: float, alpha: float) -> float:
        self._y = x if self._y is None else alpha * x + (1.0 - alpha) * self._y
        return self._y


class OneEuro:
    """One Euro filter: smooth at rest, low-lag in motion."""

    def __init__(
        self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0
    ):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x, self._dx = _LowPass(), _LowPass()
        self._t = None
        self._last = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t=None) -> float:
        t = time.monotonic() if t is None else t
        if self._t is None:
            self._t, self._last = t, x
            self._x(x, 1.0)
            self._dx(0.0, 1.0)
            return x
        dt = max(t - self._t, 1e-6)
        self._t = t
        dx = (x - self._last) / dt
        self._last = x
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)


# ----------------------------------------------------------------------------
# Camera-frame → joint mapper (gesture mimic)
# ----------------------------------------------------------------------------


def _axis(value: float, center: float, half: float, deadzone: float = 0.0) -> float:
    """Normalize a raw signal into [-1, 1] around ``center`` (± ``half``)."""
    a = max(-1.0, min(1.0, (value - center) / max(half, 1e-6)))
    return 0.0 if abs(a) < deadzone else a


class MimicMapper:
    """Map camera-frame signals straight onto SO-101 joints.

    The image is a control surface: wrist x/y (YOLO-pose) → pan and up/down
    (shoulder_lift and elbow_flex stay a locked pair on the up/fwd axes),
    wrist bend (hand direction vs elbow→wrist forearm) → wrist_flex, hand size
    (MediaPipe) → forward/back, thumb→index line angle → wrist_roll, thumb↔index distance
    → gripper. Gains and anchors come from ``mimic:`` in the config (the real
    rig's values).
    """

    def __init__(self, p: dict):
        self.p = p
        sm = p.get("smoothing", {}) or {}
        kw = dict(
            min_cutoff=float(sm.get("min_cutoff", 1.5)),
            beta=float(sm.get("beta", 0.03)),
            d_cutoff=float(sm.get("d_cutoff", 1.0)),
        )
        self._f = {k: OneEuro(**kw) for k in ("lr", "up", "fwd", "roll", "wf")}
        # Raw axes HELD across frames: a momentarily missing signal keeps its
        # previous value, so when MediaPipe loses the hand the arm keeps
        # following the YOLO wrist for x/y while depth/roll freeze - instead of
        # snapping back to neutral on every dropout.
        self._raw = {"lr": 0.0, "up": 0.0, "fwd": 0.0, "roll": 0.0, "wf": 0.0}
        self._size_samples: list = []
        self._size_neutral = float(p.get("hand_size_neutral", 0.0)) or None
        # Roll continuity: previous unwrapped source angle, the locked-in
        # ti↔reference offset (so cue switches don't jump), and the
        # auto-calibrated center (first reliable angle becomes roll 0).
        self._roll_prev = None
        self._roll_ref_offset = None
        self._roll_center = None
        self._last_gripper = float(p.get("grip_open_pos", 90.0))

    # -- axis extraction ---------------------------------------------------

    def _depth_axis(self, hand_size):
        """hand size → forward axis in [-1,1]; None = no reading (hold last)."""
        if hand_size is None or hand_size <= 0:
            return None
        p = self.p
        if self._size_neutral is None:
            self._size_samples.append(hand_size)
            if len(self._size_samples) >= int(p.get("depth_baseline_frames", 8)):
                s = sorted(self._size_samples)
                self._size_neutral = s[len(s) // 2]
            return 0.0  # still building the baseline → no forward command yet
        rel = (hand_size - self._size_neutral) / self._size_neutral
        return max(-1.0, min(1.0, rel / float(p.get("depth_rel_span", 0.25))))

    def _wrist_flex_axis(self, wrist_xy, elbow_xy, hand_dir_deg):
        """Human wrist BEND (hand vs forearm, image plane) → axis in [-1,1].

        ``hand_dir_deg`` is the hand's pointing direction - the MediaPipe
        wrist→middle-knuckle angle from vertical (``HandObservation.roll``,
        arriving here as ``roll_fallback``): 0 = hand straight up, +90 =
        screen-right. The bend is that direction measured against the
        forearm (elbow→wrist, pose kpts): hand in line with the forearm =
        0, bent one way = +, the other = -. Without an elbow this frame the
        forearm is assumed vertical, so the hand's own tilt still gives
        control. ``None`` = no hand seen (hold the previous value).
        """
        if hand_dir_deg is None:
            return None
        hand_ang = 90.0 - float(hand_dir_deg)  # 0=right, 90=up
        if wrist_xy is not None and elbow_xy is not None:
            dx = float(wrist_xy[0]) - float(elbow_xy[0])
            dy = float(elbow_xy[1]) - float(wrist_xy[1])  # +up (image y grows down)
            forearm_ang = math.degrees(math.atan2(dy, dx))
        else:
            forearm_ang = 90.0  # assume vertical forearm
        flex = (hand_ang - forearm_ang + 180.0) % 360.0 - 180.0
        half = max(float(self.p.get("wrist_half_deg", 45.0)), 1e-6)
        return max(-1.0, min(1.0, flex / half))

    @staticmethod
    def _unwrap_deg(prev: float, angle: float) -> float:
        """Shift ``angle`` by whole turns to be nearest ``prev`` (±180 seam)."""
        return prev + (angle - prev + 180.0) % 360.0 - 180.0

    def _roll_axis(self, roll_ti, roll_ref, ti_weight, roll_fallback):
        """Pick + unwrap a roll cue → axis in [-1,1]; None = no cue (hold last).

        Priority: the directed thumb→index line when reliable; else the steady
        knuckle-line reference kept on the ti scale via a locked-in offset (so
        switching cues doesn't jump); else the wrist→knuckle fallback. The
        chosen angle is unwrapped against the previous frame so the command
        only moves on a real rotation, and is measured relative to an
        auto-calibrated center: however you first hold your hand is roll 0.
        """
        min_w = float(self.p.get("roll_ti_min_weight", 0.4))

        # Maintain the ti↔reference offset whenever BOTH are reliably seen.
        if roll_ti is not None and roll_ref is not None and ti_weight >= min_w:
            self._roll_ref_offset = roll_ti - roll_ref

        cue = None
        if roll_ti is not None and ti_weight >= min_w:
            cue = roll_ti
        elif roll_ref is not None and self._roll_ref_offset is not None:
            cue = roll_ref + self._roll_ref_offset  # reference on the ti scale
        elif roll_ref is not None:
            cue = roll_ref
        elif roll_fallback is not None:
            cue = roll_fallback
        if cue is None:
            return None

        # Unwrap for continuity so a smooth rotation stays smooth across ±180.
        if self._roll_prev is not None:
            cue = self._unwrap_deg(self._roll_prev, cue)
        self._roll_prev = cue

        if self._roll_center is None:
            self._roll_center = cue
        delta = cue - self._roll_center
        return max(-1.0, min(1.0, delta / float(self.p.get("roll_half_deg", 75.0))))

    def _gripper(self, pinch, override):
        if override is not None:
            return float(override)
        if pinch is None:
            return self._last_gripper  # hold when the hand is unseen
        p = self.p
        lo, hi = float(p.get("grip_min", 0.10)), float(p.get("grip_max", 1.40))
        norm = max(0.0, min(1.0, (pinch - lo) / max(hi - lo, 1e-6)))
        close_frac = float(p.get("grip_close_frac", 0.10))
        if norm <= close_frac:
            g = GRIPPER_CLOSED_FLOOR
        else:
            span = max(1.0 - close_frac, 1e-6)
            g = GRIPPER_CLOSED_FLOOR + (norm - close_frac) / span * (
                float(p.get("grip_open_pos", 90.0)) - GRIPPER_CLOSED_FLOOR
            )
        self._last_gripper = g
        return g

    # -- the mapping ---------------------------------------------------------

    def step(
        self,
        wrist_xy,
        frame_w: int,
        frame_h: int,
        *,
        elbow_xy=None,
        hand_size=None,
        roll_ti=None,
        roll_ref=None,
        roll_ti_weight=0.0,
        roll_fallback=None,
        pinch=None,
        gripper_override=None,
        t=None,
    ) -> dict:
        """One control tick: raw signals in, a full clamped joint dict out.

        Any signal may be ``None`` (not detected this frame) - its axis HOLDS
        its previous value rather than snapping back to neutral. wrist_flex
        follows YOUR wrist bend: the hand's pointing direction
        (``roll_fallback`` = MediaPipe wrist→knuckle angle) measured against
        the forearm (``elbow_xy``, from YOLO-pose via ``imaging.pick_arm``);
        with no elbow in frame the forearm is assumed vertical. ``t`` is an
        optional timestamp (seconds) for deterministic testing.
        """
        p = self.p
        t = time.monotonic() if t is None else t

        if wrist_xy is not None:
            xf, yf = wrist_xy[0] / max(frame_w, 1), wrist_xy[1] / max(frame_h, 1)
            lr = _axis(
                xf,
                float(p.get("x_center", 0.5)),
                float(p.get("x_half", 0.32)),
                float(p.get("deadzone", 0.05)),
            )
            if p.get("mirror", True):
                lr = -lr
            self._raw["lr"] = lr
            self._raw["up"] = -_axis(
                yf,
                float(p.get("y_center", 0.5)),
                float(p.get("y_half", 0.28)),
                float(p.get("deadzone", 0.05)),
            )
        fwd = self._depth_axis(hand_size)
        if fwd is not None:
            self._raw["fwd"] = fwd
        roll = self._roll_axis(roll_ti, roll_ref, roll_ti_weight, roll_fallback)
        if roll is not None:
            self._raw["roll"] = roll
        wf = self._wrist_flex_axis(wrist_xy, elbow_xy, roll_fallback)
        if wf is not None:
            self._raw["wf"] = wf

        lr = self._f["lr"](self._raw["lr"], t)
        up = self._f["up"](self._raw["up"], t)
        fwd = self._f["fwd"](self._raw["fwd"], t)
        roll = self._f["roll"](self._raw["roll"], t)
        wf = self._f["wf"](self._raw["wf"], t)

        # fwd < 0 = hand moving AWAY from the camera. shoulder_lift may use a
        # separate gain on that side (sl_fwd_away_gain, defaults to
        # sl_fwd_gain) - joint limits still apply via clamp_joints below.
        sl_fwd = float(p.get("sl_fwd_gain", 110.0))
        if fwd < 0.0:
            sl_fwd = float(p.get("sl_fwd_away_gain", sl_fwd))

        joints = {
            "shoulder_pan": float(p.get("shoulder_pan0", 0.0))
            + float(p.get("pan_gain", 150.0)) * lr,
            # shoulder_lift + elbow_flex are a LOCKED PAIR: both driven by the
            # same fwd/up axes, deliberately - see ef_/sl_ gains in the config.
            "shoulder_lift": float(p.get("shoulder_lift0", -5.0))
            + sl_fwd * fwd
            + float(p.get("sl_up_gain", 58.0)) * up,
            "elbow_flex": float(p.get("elbow_flex0", 10.0))
            + float(p.get("ef_fwd_gain", -75.0)) * fwd
            + float(p.get("ef_up_gain", 78.0)) * up,
            # wrist_flex follows the human wrist BEND (hand vs forearm) -
            # wf_up_gain is its gain (flip its sign to invert).
            "wrist_flex": float(p.get("wrist_flex0", -10.0))
            + float(p.get("wf_up_gain", -135.0)) * wf,
            "wrist_roll": float(p.get("wrist_roll0", 0.0))
            + float(p.get("roll_gain", 128.0)) * roll,
            "gripper": self._gripper(pinch, gripper_override),
        }
        return clamp_joints(joints)

    def reset(self) -> None:
        """Forget smoothing/baseline state (call when the person leaves view).

        The gripper opening is deliberately KEPT so a held object isn't
        dropped on a dropout - it only moves on a new thumb↔index reading.
        """
        for f in self._f.values():
            f.reset()
        self._raw = {"lr": 0.0, "up": 0.0, "fwd": 0.0, "roll": 0.0, "wf": 0.0}
        self._roll_prev = None
        self._roll_ref_offset = None
        self._roll_center = None
        self._size_samples.clear()
        if float(self.p.get("hand_size_neutral", 0.0)) == 0.0:
            self._size_neutral = None


def mapper_from_config(cfg: dict) -> MimicMapper:
    """Build a :class:`MimicMapper` from the ``mimic:`` section of ``cfg``."""
    return MimicMapper(cfg.get("mimic", {}) or {})


# ----------------------------------------------------------------------------
# Dance / wave keyframes (static choreography assets)
# ----------------------------------------------------------------------------

DANCE_MOVES = {
    "sway": [
        {
            "shoulder_pan": -45.0,
            "shoulder_lift": -60.0,
            "elbow_flex": 60.0,
            "wrist_roll": -40.0,
        },
        {
            "shoulder_pan": 45.0,
            "shoulder_lift": -60.0,
            "elbow_flex": 60.0,
            "wrist_roll": 40.0,
        },
    ],
    "pump": [
        {"shoulder_lift": -90.0, "elbow_flex": 95.0, "wrist_flex": 60.0},
        {"shoulder_lift": -20.0, "elbow_flex": 20.0, "wrist_flex": -20.0},
    ],
    "disco_point": [
        {
            "shoulder_pan": -60.0,
            "shoulder_lift": -30.0,
            "elbow_flex": 10.0,
            "wrist_flex": 0.0,
        },
        {
            "shoulder_pan": 60.0,
            "shoulder_lift": -85.0,
            "elbow_flex": 80.0,
            "wrist_flex": 50.0,
        },
    ],
    "wrist_spin": [
        {"shoulder_lift": -55.0, "elbow_flex": 70.0, "wrist_roll": -90.0},
        {"shoulder_lift": -55.0, "elbow_flex": 70.0, "wrist_roll": 90.0},
    ],
    "gripper_chomp": [
        {"gripper": 80.0, "wrist_flex": 30.0},
        {"gripper": 5.0, "wrist_flex": 50.0},
    ],
}

WAVE_UP = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -10.0,
    "elbow_flex": 80.0,
    "wrist_flex": -30.0,
    "wrist_roll": 0.0,
    "gripper": 60.0,
}
WAVE_LEFT = {"wrist_roll": -45.0, "wrist_flex": -10.0}
WAVE_RIGHT = {"wrist_roll": 45.0, "wrist_flex": -10.0}
