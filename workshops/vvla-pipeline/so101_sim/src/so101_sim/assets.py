# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Fetch the SO-101 STL meshes the MJCF model references (install-time step).

The meshes are not committed to this repo. They come from MuJoCo Menagerie's
``robotstudio_so101`` model (Apache-2.0, see ``models/LICENSE``) at a pinned
commit, and every file is checked against its SHA-256 below.

    python -m so101_sim.assets            # download whatever is missing
    python -m so101_sim.assets --check    # exit 1 if anything is missing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from so101_sim.config import MODELS_DIR

ASSETS_DIR = MODELS_DIR / "assets"
MENAGERIE_COMMIT = "3abca2e302625cec9429c49806121ea2855cef81"
BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MENAGERIE_COMMIT}/robotstudio_so101/assets"
)

SHA256 = {
    "base_motor_holder_so101_v1.stl": "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424",
    "base_so101_v2.stl": "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
    "motor_holder_so101_base_v1.stl": "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b",
    "motor_holder_so101_wrist_v1.stl": "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011",
    "moving_jaw_so101_gripper_part0_v1.stl": "3a92e115d1934d4c45cff808c4eff2ea10361a137a7f89940033a1f8a821b88c",
    "moving_jaw_so101_gripper_part1_v1.stl": "d10f6c671d26f62988ec95efee9370a0657b5833b3c5f8aad59fd69a87ef2aee",
    "moving_jaw_so101_gripper_v1.stl": "e8a5dc040b98e27453a5b74f9087d1dbca14589cf9a17c4de7868d5ab98413cd",
    "moving_jaw_so101_v1.stl": "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d",
    "rotation_pitch_so101_v1.stl": "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34",
    "sts3215_03a_no_horn_v1.stl": "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887",
    "sts3215_03a_v1.stl": "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    "under_arm_so101_v1.stl": "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71",
    "upper_arm_so101_v1.stl": "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f",
    "waveshare_mounting_plate_so101_v2.stl": "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9",
    "wrist_roll_follower_so101_camera_mount.stl": "58c457b33a26d9c9bb337aa351173e7123f84719a93f486541adef6eba57ae21",
    "wrist_roll_follower_so101_gripper_part0_v1.stl": "370eb470a6ab48b08febcaeb5274d2bcc91246692532ce12c795fb82977def72",
    "wrist_roll_follower_so101_gripper_v1.stl": "65f32debcf265420c0160afe004f8befc5df4b3e578fd59ebaa62eade0371224",
    "wrist_roll_follower_so101_v1.stl": "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4",
    "wrist_roll_pitch_so101_v2.stl": "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing(dest: Path = ASSETS_DIR) -> list[str]:
    """Mesh files that are absent or don't match their pinned checksum."""
    return [
        name
        for name, digest in SHA256.items()
        if not (dest / name).is_file() or _sha256(dest / name) != digest
    ]


def fetch(dest: Path = ASSETS_DIR, timeout: float = 60.0) -> int:
    """Download every missing mesh into ``dest``; returns how many were fetched."""
    todo = missing(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in todo:
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=timeout) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != SHA256[name]:
            raise RuntimeError(f"{name}: checksum mismatch from {BASE_URL}")
        tmp = dest / f".{name}.part"
        tmp.write_bytes(data)
        tmp.replace(dest / name)
    return len(todo)


def require(dest: Path = ASSETS_DIR) -> None:
    """Raise a clear error when meshes are missing (the model can't load without them)."""
    gone = missing(dest)
    if gone:
        raise FileNotFoundError(
            f"{len(gone)} SO-101 mesh file(s) missing in {dest} - "
            "fetch them once with: python -m so101_sim.assets"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only report; exit 1 if missing")
    args = ap.parse_args()
    if args.check:
        gone = missing()
        print(f"SO-101 meshes: {len(SHA256) - len(gone)}/{len(SHA256)} present in {ASSETS_DIR}")
        return 1 if gone else 0
    n = fetch()
    print(f"SO-101 meshes: fetched {n}, all {len(SHA256)} present in {ASSETS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
