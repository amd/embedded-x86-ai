#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Versions bootstrap.sh restores after installing LeRobot.
# Sourced by bootstrap.sh. The tests in tests/sim/test_dep_pins.py call the
# same functions, so the decision lives here and nowhere else.

# LeRobot v0.4.4 is the last release that installs with numpy 1.x and
# huggingface-hub 0.x. v0.5.0 and later require numpy>=2 and
# huggingface-hub>=1, which break ONNX Runtime and transformers.
# There is no v0.5.2 tag.
lerobot_rev() {
  printf '%s\n' v0.4.4
}

# transformers requires huggingface-hub <1.0. LeRobot v0.4.4 requires
# >=0.34.2,<0.36.0. The install uses the intersection.
hub_spec() {
  printf '%s\n' 'huggingface_hub[cli]>=0.34.2,<0.36.0'
}

# ok: the installed version is inside hub_spec. repin: anything else,
# including 1.x, 2.x, 0.36.x, and a missing package.
hub_pin_action() {
  python3 - "$1" <<'PY'
import sys

ver = sys.argv[1]
parts = ver.split(".")
try:
    nums = [int(part.split("+", 1)[0]) for part in parts[:3]]
    if len(nums) != 3:
        raise ValueError
except ValueError:
    print("repin")
    raise SystemExit
major, minor, patch = nums
if (0, 34, 2) <= (major, minor, patch) < (0, 36, 0):
    print("ok")
else:
    print("repin")
PY
}
