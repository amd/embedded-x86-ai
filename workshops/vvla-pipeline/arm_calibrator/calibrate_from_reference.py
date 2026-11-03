# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""Calibrate an SO-ARM101 follower arm from a reference calibration.

The SO-ARM101 uses Feetech STS3215 bus servos. LeRobot stores each joint's
calibration as ``homing_offset`` + ``range_min``/``range_max``, where the ranges
live in the *homed frame* (home = 2047 sits inside every range).

Transfer method
---------------
* ``range_min`` / ``range_max`` are COPIED DIRECTLY from the reference. Because
  they are in the homed frame, and every arm is homed so that ``present = 2047``
  at its home pose, that frame is aligned across arms -- so the ranges transfer
  as-is.
* ``homing_offset`` is MEASURED live from the target arm held at its home pose
  (it is per-arm and cannot be copied):

      raw_at_home = present + current_homing_offset
      homing_new  = raw_at_home - 2047

Only the ``homing_offset`` is per-arm; the ranges come from the reference.

Safety
------
* Nothing is written unless ``--commit`` is given (default is a dry run).
* Every joint is validated (``0 <= range_min < range_max <= 4095``) and the
  script ABORTS before writing if any joint is out of range.
* A joint whose home (2047) falls outside its range is flagged (an under-swept
  reference) -- that joint would not be able to reach home.
* Writes go to EEPROM with torque disabled and the EEPROM lock handled, then
  are read back and verified.

Usage
-----
    # Dry run (no writes) -- inspect the numbers first:
    sudo python3 calibrate_from_reference.py \\
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_<ID>-if00 \\
        --ref golden_reference.json --out robotN.json

    # Commit to the servos once the dry run looks right:
    sudo python3 calibrate_from_reference.py \\
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_<ID>-if00 \\
        --ref golden_reference.json --out robotN.json --commit

The target arm must be held at its home/middle pose while the script reads.
"""

import argparse
import json
import os
import sys
import time

import serial

# --- Feetech STS3215 control table: name -> (address, size_bytes) -----------
REG = {
    "Model_Number": (3, 2),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Torque_Enable": (40, 1),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}

INSTRUCTION_READ = 0x02
INSTRUCTION_WRITE = 0x03
HALF_TURN = 2047  # home reference (12-bit encoder midpoint)
ENCODER_MAX = 4095
SIGN_BIT = 11  # Feetech homing offset is sign-magnitude, bit 11

# SO-ARM101 joint order (servo id -> joint name).
JOINT_ORDER = [
    (1, "shoulder_pan"),
    (2, "shoulder_lift"),
    (3, "elbow_flex"),
    (4, "wrist_flex"),
    (5, "wrist_roll"),
    (6, "gripper"),
]


# --------------------------------------------------------------------------- #
# Feetech serial protocol helpers
# --------------------------------------------------------------------------- #
def _checksum(data):
    return (~sum(data)) & 0xFF


def read_register(ser, servo_id, address, size):
    """Read a register; return its little-endian value, or None on failure."""
    body = [servo_id, 4, INSTRUCTION_READ, address, size]
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
    ser.flush()
    resp = ser.read(6 + size)
    if (
        len(resp) < 6 + size
        or resp[0] != 0xFF
        or resp[1] != 0xFF
        or resp[2] != servo_id
    ):
        return None
    return int.from_bytes(resp[5 : 5 + size], "little", signed=False)


def write_register(ser, servo_id, address, value, size):
    """Write a register; return True if a status packet came back."""
    data = list(value.to_bytes(size, "little", signed=False))
    body = [servo_id, size + 3, INSTRUCTION_WRITE, address, *data]
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
    ser.flush()
    resp = ser.read(6)
    return len(resp) == 6 and resp[2] == servo_id


def decode_sign_magnitude(value, sign_bit=SIGN_BIT):
    mask = 1 << sign_bit
    return -(value & (mask - 1)) if value & mask else value


def encode_sign_magnitude(value, sign_bit=SIGN_BIT):
    mask = 1 << sign_bit
    magnitude = abs(value)
    if magnitude >= mask:
        raise ValueError(f"homing offset {value} out of encodable range")
    return (mask | magnitude) if value < 0 else value


def read_position_median(ser, servo_id, samples=5):
    """Median of a few Present_Position reads (rejects transient noise)."""
    vals = []
    for _ in range(samples):
        p = read_register(ser, servo_id, *REG["Present_Position"])
        if p is not None:
            vals.append(p)
        time.sleep(0.005)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def load_reference_ranges(path):
    """Load {joint: (range_min, range_max)} from a reference calibration JSON."""
    with open(path) as f:
        cal = json.load(f)
    return {name: (int(e["range_min"]), int(e["range_max"])) for name, e in cal.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Calibrate an SO-ARM101 arm from a reference calibration."
    )
    parser.add_argument(
        "--port", required=True, help="Serial port or /dev/serial/by-id/... path"
    )
    parser.add_argument(
        "--ref", required=True, help="Reference calibration JSON (source of ranges)"
    )
    parser.add_argument("--out", required=True, help="Output calibration JSON path")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write to the servos (default: dry run only)",
    )
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=float, default=0.1)
    args = parser.parse_args()

    ref_ranges = load_reference_ranges(args.ref)
    print(f"Reference ranges from {args.ref}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as exc:
        print(f"ERROR: could not open {args.port}: {exc}", file=sys.stderr)
        return 1

    mode = "COMMIT (writing to servos)" if args.commit else "DRY RUN (no writes)"
    print(f"Port {args.port} @ {args.baud} baud  [{mode}]\n")

    plan = {}
    errors = []
    with ser:
        # --- measure + compute, validating everything before any write ------
        for servo_id, name in JOINT_ORDER:
            if name not in ref_ranges:
                errors.append(f"joint {name}: not present in reference JSON")
                continue
            if read_register(ser, servo_id, *REG["Model_Number"]) is None:
                errors.append(f"joint {name} (id {servo_id}): no response")
                continue

            present = read_position_median(ser, servo_id)
            cur_off = read_register(ser, servo_id, *REG["Homing_Offset"])
            if present is None or cur_off is None:
                errors.append(f"joint {name} (id {servo_id}): read failed")
                continue
            cur_off = decode_sign_magnitude(cur_off)

            # Wrap: a joint whose current offset straddles 0/4095 sums outside the encoder.
            raw_at_home = (present + cur_off) % (ENCODER_MAX + 1)
            homing_new = raw_at_home - HALF_TURN
            range_min, range_max = ref_ranges[name]  # copied directly

            plan[servo_id] = {
                "name": name,
                "present": present,
                "raw_at_home": raw_at_home,
                "homing_new": homing_new,
                "range_min": range_min,
                "range_max": range_max,
            }

            if not (0 <= range_min < range_max <= ENCODER_MAX):
                errors.append(
                    f"joint {name}: reference range "
                    f"[{range_min}, {range_max}] outside [0, {ENCODER_MAX}]"
                )
            try:
                encode_sign_magnitude(homing_new)
            except ValueError as exc:
                errors.append(f"joint {name}: {exc}")
            if not (range_min <= HALF_TURN <= range_max):
                plan[servo_id]["home_warning"] = True

        # --- report ---------------------------------------------------------
        hdr = (
            f"{'id':>2} {'joint':<14} {'present':>7} {'homing':>7} "
            f"{'range_min':>9} {'range_max':>9} {'home?':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for servo_id, _ in JOINT_ORDER:
            p = plan.get(servo_id)
            if not p:
                continue
            flag = "OUT!" if p.get("home_warning") else "ok"
            print(
                f"{servo_id:>2} {p['name']:<14} {p['present']:>7} "
                f"{p['homing_new']:>7} {p['range_min']:>9} {p['range_max']:>9} "
                f"{flag:>6}"
            )

        warned = [p["name"] for p in plan.values() if p.get("home_warning")]
        if warned:
            print(
                f"\nWARNING: home (2047) is OUTSIDE the range for: "
                f"{', '.join(warned)} -- under-swept reference; those joints "
                f"cannot reach home.",
                file=sys.stderr,
            )

        if errors:
            print("\nVALIDATION FAILED -- no servo writes performed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2

        # --- write to servos (EEPROM) ---------------------------------------
        if args.commit:
            print("\nWriting to servos...")
            for servo_id, _ in JOINT_ORDER:
                p = plan[servo_id]
                write_register(
                    ser, servo_id, REG["Torque_Enable"][0], 0, 1
                )  # torque off
                write_register(ser, servo_id, REG["Lock"][0], 0, 1)  # unlock EEPROM
                write_register(
                    ser,
                    servo_id,
                    REG["Homing_Offset"][0],
                    encode_sign_magnitude(p["homing_new"]),
                    2,
                )
                write_register(
                    ser, servo_id, REG["Min_Position_Limit"][0], p["range_min"], 2
                )
                write_register(
                    ser, servo_id, REG["Max_Position_Limit"][0], p["range_max"], 2
                )
                write_register(ser, servo_id, REG["Lock"][0], 1, 1)  # lock EEPROM

            print("\nRead-back verification:")
            all_ok = True
            for servo_id, name in JOINT_ORDER:
                off = decode_sign_magnitude(
                    read_register(ser, servo_id, *REG["Homing_Offset"])
                )
                mn = read_register(ser, servo_id, *REG["Min_Position_Limit"])
                mx = read_register(ser, servo_id, *REG["Max_Position_Limit"])
                exp = plan[servo_id]
                ok = (
                    off == exp["homing_new"]
                    and mn == exp["range_min"]
                    and mx == exp["range_max"]
                )
                all_ok = all_ok and ok
                print(
                    f"  {name:<14} homing={off:>6} min={mn:>5} max={mx:>5}  "
                    f"[{'OK' if ok else 'MISMATCH'}]"
                )
            if not all_ok:
                print(
                    "\nERROR: read-back mismatch on one or more joints.",
                    file=sys.stderr,
                )
                return 3
        else:
            print("\n(dry run -- rerun with --commit to write to the servos)")

    # --- write output JSON --------------------------------------------------
    calibration = {
        name: {
            "id": servo_id,
            "drive_mode": 0,
            "homing_offset": plan[servo_id]["homing_new"],
            "range_min": plan[servo_id]["range_min"],
            "range_max": plan[servo_id]["range_max"],
        }
        for servo_id, name in JOINT_ORDER
        if servo_id in plan
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=4)
        f.write("\n")
    print(f"\nCalibration JSON written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
