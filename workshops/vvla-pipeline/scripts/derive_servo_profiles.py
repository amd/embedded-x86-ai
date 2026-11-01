# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Per-joint STS3215 motion profiles derived from the servo and the arm.

    python scripts/derive_servo_profiles.py            # print the table
    python scripts/derive_servo_profiles.py --write    # also write config/servo_profiles.json

Inputs, all read from their owners rather than restated here:

- the twin's arm model (``so101_sim`` scene): link masses and inertias, rotor
  armature, joint damping (the twin's back-EMF, so it is the servo's
  torque-speed line), and joint friction;
- the twin's servo settings (``so101_sim.config.load_sim_config``): stall
  torque, the fastest ``Goal_Velocity`` and the ``Acceleration`` unit;
- the twin's overload protection (``so101_sim.servo.OverloadProtection``):
  the ``Overload_Torque`` threshold and ``Protection_Time``;
- the workshop's joint limits (``common.motion.JOINT_LIMITS``) as the
  operating envelope, and the heaviest table prop as the payload in the jaws.

Policy (the only choices this script makes):

- cruise: gravity + friction + damping x speed stays at or below
  ``--cruise-frac`` of stall torque in the worst pose of the envelope, so a
  long move never sits above the overload threshold;
- acceleration: the cruise torque plus inertia x acceleration stays at or
  below ``--peak-frac`` of stall torque. That peak lasts only the ramp, far
  shorter than ``Protection_Time``.

The JSON holds the policy and the worst-pose table. At run time
``common.servo_profile.ServoProfiler`` applies the same formula
(``profile_for``) to the gravity and inertia along each move's own path; the
table is the profile a transport writes at connect, before its first move.

The bus tick is estimated from the Feetech protocol at 1 Mbps. It is not
measured; measure it on the real arm and record the figure here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workshop" / "solution"))
sys.path.insert(0, str(ROOT / "so101_sim" / "src"))

from common.motion import JOINT_LIMITS  # noqa: E402
from common.servo_profile import heaviest_prop, profile_for  # noqa: E402
from so101_sim import MOTOR_NAMES  # noqa: E402
from so101_sim.config import DEFAULT_SCENE, load_sim_config  # noqa: E402
from so101_sim.mapping import JointMap  # noqa: E402
from so101_sim.servo import OverloadProtection  # noqa: E402

# Feetech protocol at 1 Mbps: 10 bits per byte on the wire, plus each servo's
# Return_Delay_Time (0 after LeRobot's configure) and the host's turnaround.
BAUD = 1_000_000
BITS_PER_BYTE = 10
HOST_TURNAROUND_S = 0.0005


def _packet_s(n_bytes: int) -> float:
    return n_bytes * BITS_PER_BYTE / BAUD


def bus_tick_s(n_servos: int = 6) -> float:
    """One control tick: sync_write the goal, sync_read position, Moving and Status."""
    write = _packet_s(8 + n_servos * 3)  # header + per servo: id + 2 position bytes
    reads = 0.0
    for data_bytes in (2, 1, 1):  # Present_Position, Moving, Status
        request = _packet_s(8 + n_servos)
        replies = n_servos * _packet_s(6 + data_bytes)
        reads += request + replies + HOST_TURNAROUND_S
    return write + HOST_TURNAROUND_S + reads


def _envelope_stats(payload_kg: float, samples: int, seed: int) -> dict:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    wrist = model.jnt_bodyid[model.joint("wrist_roll").id]
    model.body_mass[wrist] += payload_kg
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in MOTOR_NAMES]
    dadr = [model.jnt_dofadr[model.joint(n).id] for n in MOTOR_NAMES]
    body = [n for n in MOTOR_NAMES if n != "gripper"]
    lo = np.radians([JOINT_LIMITS[n][0] for n in body])
    hi = np.radians([JOINT_LIMITS[n][1] for n in body])
    rng = np.random.default_rng(seed)
    full = np.zeros((model.nv, model.nv))
    gravity = np.zeros(len(MOTOR_NAMES))
    inertia = np.zeros(len(MOTOR_NAMES))
    for _ in range(samples):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[qadr[: len(body)]] = lo + (hi - lo) * rng.random(len(body))
        mujoco.mj_forward(model, data)
        gravity = np.maximum(gravity, np.abs(data.qfrc_bias[dadr]))
        mujoco.mj_fullM(model, data, full)
        inertia = np.maximum(inertia, full[dadr, dadr])
    return {
        "gravity_nm": gravity,
        "inertia_kgm2": inertia,
        "damping": model.dof_damping[dadr].copy(),
        "friction_nm": model.dof_frictionloss[dadr].copy(),
    }


def derive(cruise_frac: float, peak_frac: float, samples: int, seed: int) -> dict:
    sim = load_sim_config()
    stall = float(sim["current_model"]["stall_torque_nm"])
    max_speed_raw = float(sim["servo"]["max_speed_raw"])
    accel_unit = float(sim["servo"]["accel_unit_steps"])
    protection = OverloadProtection(len(MOTOR_NAMES))
    jmap = JointMap(sim["joint_map"], sim["gripper_map"])
    prop, payload = heaviest_prop(mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE)))
    stats = _envelope_stats(payload, samples, seed)
    joints = {}
    for i, name in enumerate(MOTOR_NAMES):
        g, m = float(stats["gravity_nm"][i]), float(stats["inertia_kgm2"][i])
        b, f = float(stats["damping"][i]), float(stats["friction_nm"][i])
        step = jmap.steps_to_rad(name, 1.0)
        speed_raw, accel_raw = profile_for(
            g, m, b, f, rad_per_step=step, stall_nm=stall, cruise_frac=cruise_frac, peak_frac=peak_frac,
            max_speed_raw=max_speed_raw, accel_unit_steps=accel_unit,
        )
        joints[name] = {
            "gravity_nm": round(g, 3),
            "inertia_kgm2": round(m, 5),
            "speed_raw": speed_raw,
            "accel_raw": accel_raw,
            "speed_deg_s": round(math.degrees(speed_raw * step), 1),
            "accel_deg_s2": round(math.degrees(accel_raw * accel_unit * step), 1),
        }
    tick = bus_tick_s()
    return {
        "policy": {"cruise_frac_of_stall": cruise_frac, "peak_frac_of_stall": peak_frac},
        "inputs": {
            "model": str(DEFAULT_SCENE.relative_to(ROOT)),
            "stall_torque_nm": stall,
            "overload_threshold_frac": float(protection.threshold[0]),
            "protection_time_s": float(protection.time_s[0]),
            "max_speed_raw": max_speed_raw,
            "accel_unit_steps": accel_unit,
            "payload": {"prop": prop, "kg": payload},
            "envelope": "common.motion.JOINT_LIMITS",
            "samples": samples,
            "seed": seed,
        },
        "bus_tick": {
            "estimated_s": round(tick, 5),
            "max_rate_hz": round(1.0 / tick, 1),
            "measured_s": None,
            "note": "Feetech protocol at 1 Mbps, not measured on the real arm",
        },
        "joints": joints,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cruise-frac", type=float, default=0.70)
    ap.add_argument("--peak-frac", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="write config/servo_profiles.json")
    args = ap.parse_args()
    table = derive(args.cruise_frac, args.peak_frac, args.samples, args.seed)
    inp = table["inputs"]
    print(
        f"stall {inp['stall_torque_nm']} N.m, overload above {inp['overload_threshold_frac']:.0%} "
        f"for {inp['protection_time_s']} s, payload {inp['payload']['prop']} {inp['payload']['kg']} kg"
    )
    print(f"policy: cruise <= {args.cruise_frac:.0%} of stall, ramp peak <= {args.peak_frac:.0%} of stall")
    print("worst pose of the envelope; common.servo_profile plans each move from its own path")
    print(f"{'joint':14s} {'gravity':>8s} {'inertia':>8s} {'speed':>7s} {'deg/s':>7s} {'accel':>6s} {'deg/s2':>8s}")
    for name, j in table["joints"].items():
        print(
            f"{name:14s} {j['gravity_nm']:8.3f} {j['inertia_kgm2']:8.4f} {j['speed_raw']:7d} "
            f"{j['speed_deg_s']:7.1f} {j['accel_raw']:6d} {j['accel_deg_s2']:8.1f}"
        )
    tick = table["bus_tick"]
    print(f"bus tick (estimated, not measured): {tick['estimated_s'] * 1000:.2f} ms, at most {tick['max_rate_hz']} Hz")
    if args.write:
        out = ROOT / "config" / "servo_profiles.json"
        out.write_text(json.dumps(table, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
