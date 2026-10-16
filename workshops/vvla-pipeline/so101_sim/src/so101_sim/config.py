# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Defaults for the digital twin, merged with a ``sim:`` config section.

Both host stacks (``config/pipeline.yaml`` and ``workshop/config/workshop.yaml``)
carry a ``sim:`` block; anything omitted there falls back to :data:`DEFAULTS`.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_SCENE = MODELS_DIR / "scene_vvla.xml"

# LeRobot calibration "homing" pose (the L shape) is the MJCF zero, so body
# joints map 1:1 (deg -> rad, sign +1, offset 0). See calibration/Image.jpg.
_IDENTITY = {"sign": 1.0, "offset_deg": 0.0}

DEFAULTS: dict = {
    "scene": None,  # None -> bundled models/scene_vvla.xml
    "physics_hz": 500,  # mj_step rate (sets model timestep)
    "realtime_factor": 1.0,
    "max_catchup_s": 0.05,  # fall further behind than this -> drop sim time
    "gl": "auto",  # auto | egl | osmesa | glfw
    "egl_device": 0,
    "require_hw_gl": False,  # True: refuse llvmpipe (software) rendering
    "viewer": False,  # MuJoCo passive viewer window (needs a display)
    "servo": {
        # Power-on profile. LeRobot's SO101Follower.configure() leaves
        # Goal_Velocity at 0 (= max speed) with Acceleration 254; direct-goal
        # transports then write their own profile (1000 / 50) like on the rig.
        "speed_raw": 0,
        "accel_raw": 254,
        "max_speed_raw": 3073,  # ~270 deg/s, the 12 V no-load speed (7.4 V: ~312 deg/s)
        "accel_unit_steps": 100.0,  # one Acceleration unit = 100 steps/s^2
    },
    "joint_map": {
        "shoulder_pan": dict(_IDENTITY),
        "shoulder_lift": dict(_IDENTITY),
        "elbow_flex": dict(_IDENTITY),
        "wrist_flex": dict(_IDENTITY),
        "wrist_roll": dict(_IDENTITY),
    },
    "gripper_map": {
        "closed_cal": None,  # filled from GRIPPER_CLOSED_FLOOR in load_sim_config
        "open_cal": 95.0,
        "closed_rad": -0.15,
        "open_rad": 1.6,
        "counts_span": 2271,  # raw encoder span of the 0-100 gripper range
    },
    "current_model": {
        # 7.4 V STS3215 datasheet; current is interpolated through
        # (0, rated_nm, stall_torque_nm) -> (idle or moving, rated_a, stall_a).
        "idle_a": 0.006,  # stopped, no load
        "moving_a": 0.15,  # running, no load
        "rated_nm": 0.49,  # rated torque, 5 kg·cm
        "rated_a": 0.65,
        "stall_a": 2.5,
        "stall_torque_nm": 1.91,  # 19.5 kg·cm, matches the actuator forcerange
        "ma_per_unit": 6.5,  # Present_Current LSB
        "noise_units": 3.0,  # 1-sigma readback noise
        "lpf_hz": 20.0,
        "seed": 0,
    },
    "quantize_readback": True,  # encoder resolution on Present_Position
    "rest_pose": None,  # None -> the host stack's REST_POSE (set by adapters)
    "cameras": {
        "arm": {"camera": "wrist_cam", "width": 640, "height": 480, "fps": 30},
        "operator": {"camera": "operator_cam", "width": 640, "height": 480, "fps": 30},
        "view": {
            "camera": "free",
            "width": 800,
            "height": 800,
            "fps": 30,
            "lookat": [0.17, 0.0, 0.18],  # high and far enough to keep the raised arm in frame
            "distance": 0.9,
            "azimuth": 150.0,
            "elevation": -24.0,
        },
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_sim_config(cfg: dict | None = None, **overrides) -> dict:
    """Merged sim settings from a full host config, a bare ``sim:`` dict, or None.

    Camera entries may be a bare camera name (``arm: wrist_cam``) as shorthand.
    """
    if not cfg:
        section = {}
    elif isinstance(cfg.get("sim"), dict):
        section = cfg["sim"]
    elif "robot" in cfg or "behaviors" in cfg:  # host config without a sim block
        section = {}
    else:
        section = cfg
    cams = section.get("cameras")
    if isinstance(cams, dict):
        section = dict(section)
        section["cameras"] = {
            k: ({"camera": v} if isinstance(v, str) else v) for k, v in cams.items()
        }
    out = _merge(DEFAULTS, section)
    out = _merge(out, overrides)
    root = Path(__file__).resolve().parents[3]
    sol = str(root / "workshop" / "solution")
    if sol not in sys.path:
        sys.path.append(sol)
    from common.motion import GRIPPER_CLOSED_FLOOR

    out["gripper_map"]["closed_cal"] = GRIPPER_CLOSED_FLOOR
    return out


def scene_path(sim_cfg: dict, root: Path | None = None) -> Path:
    """Absolute scene path; relative paths resolve against ``root`` then the package."""
    scene = sim_cfg.get("scene")
    if not scene:
        return DEFAULT_SCENE
    p = Path(scene)
    if p.is_absolute():
        return p
    for base in (root, Path.cwd(), MODELS_DIR.parent.parent, MODELS_DIR):
        if base is not None and (base / p).exists():
            return (base / p).resolve()
    if (MODELS_DIR / p.name).exists():
        return MODELS_DIR / p.name
    raise FileNotFoundError(f"sim scene not found: {scene}")
