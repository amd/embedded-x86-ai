#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Versions and packages bootstrap.sh installs for the SO-101 arm.
# Sourced by bootstrap.sh. The tests in tests/sim/test_dep_pins.py call the
# same functions, so the decision lives here and nowhere else.

# LeRobot v0.6.0 is the first release whose code accepts the ROCm torchvision
# wheel (0.26.0). v0.4.4 through v0.5.1 declare torchvision<0.26. v0.6.0 also
# declares numpy>=2 and huggingface-hub>=1. No release accepts the wheel and
# the numpy 1.x / hub <1 pins in requirements.txt together, so bootstrap
# installs this tag with --no-deps and does not solve that dependency set.
lerobot_rev() {
  printf '%s\n' v0.6.0
}

# Packages the arm import uses that requirements.txt does not already pin.
# draccus and tqdm are imported by the SO-101 classes. The Feetech SDK,
# pyserial, and deepdiff are the [feetech] extra.
lerobot_runtime_specs() {
  cat <<'EOF'
draccus==0.10.0
feetech-servo-sdk>=1.0.0,<2.0.0
pyserial>=3.5,<4.0
deepdiff>=7.0.1,<9.0.0
tqdm>=4.66.0,<5.0.0
EOF
}

# transformers requires huggingface-hub <1.0. requirements.txt is the pin.
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
