# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Camera-frame to joint mapping for gesture mimic.

:class:`MimicMapper` maps camera-frame signals straight onto SO-101 joints,
smoothed by a One Euro filter per axis. :class:`CameraMimicMapper` accepts
the keyword names :mod:`vla_pipeline.behaviors.gesture_mimic` passes. The
workshop keeps its own copy of the mapper (``workshop/common/motion.py``);
``tests/sim/test_copies.py`` fails when the two drift apart.
"""

from __future__ import annotations

import math
import time

from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, clamp_joints


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
        """Gripper goal from the thumb↔index distance; ``None`` without a reading."""
        if override is not None:
            return float(override)
        if pinch is None:
            return None
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
        """One control tick: raw signals in, a clamped joint dict out.

        Any signal may be ``None`` (not detected this frame) - its axis HOLDS
        its previous value rather than snapping back to neutral. The gripper
        is in the dict only when this frame has a ``pinch`` reading (or a
        ``gripper_override``): without one it is not commanded, so the jaws
        keep their current goal and a held object stays held until MediaPipe
        reports the pinch. wrist_flex
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
        }
        gripper = self._gripper(pinch, gripper_override)
        if gripper is not None:  # no pinch reading: the jaws keep their current goal
            joints["gripper"] = gripper
        return clamp_joints(joints)

    def reset(self) -> None:
        """Forget smoothing/baseline state (call when the person leaves view).

        The mapper keeps no gripper state; the gripper is commanded only on a
        pinch reading.
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



class CameraMimicMapper(MimicMapper):
    """:class:`MimicMapper` with the pipeline step() keyword names."""

    def step(self, wrist_xy, frame_w, frame_h, **kwargs):
        if "hand_roll_deg" in kwargs:
            kwargs.setdefault("roll_fallback", kwargs.pop("hand_roll_deg"))
        if "roll_ti_deg" in kwargs:
            kwargs["roll_ti"] = kwargs.pop("roll_ti_deg")
        if "roll_ref_deg" in kwargs:
            kwargs["roll_ref"] = kwargs.pop("roll_ref_deg")
        return super().step(wrist_xy, frame_w, frame_h, **kwargs)


def mapper_from_config(cfg: dict) -> CameraMimicMapper:
    """Build a mapper from ``behaviors.gesture_mimic`` or a top-level ``mimic``."""
    behaviors = cfg.get("behaviors") or {}
    section = behaviors.get("gesture_mimic") or cfg.get("mimic") or {}
    return CameraMimicMapper(section)
