# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""bootstrap.sh asks scripts/dep_pins.sh which LeRobot tag to check out.

The ROCm wheel is torchvision 0.26.0. No LeRobot release accepts that wheel
together with numpy below 2 and huggingface-hub below 1.0. v0.6.0 is installed
with --no-deps. An unpinned clone resolved to 0.7.1 and hub 2.4.0, and the
old hub check logged 2.4.0 as OK.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PIPE_ROOT = Path(__file__).resolve().parents[2]
PINS = PIPE_ROOT / "scripts" / "dep_pins.sh"

pytestmark = [pytest.mark.sim, pytest.mark.timeout(30)]


def _call(function: str, arg: str = "") -> str:
    command = f'source "{PINS}" && {function}'
    if arg:
        command += f' "{arg}"'
    done = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def test_lerobot_pin_accepts_the_rocm_torchvision_wheel():
    assert _call("lerobot_rev") == "v0.6.0"


def test_arm_packages_are_installed_without_lerobots_dependency_set():
    specs = _call("lerobot_runtime_specs")
    assert specs.splitlines() == [
        "draccus==0.10.0",
        "feetech-servo-sdk>=1.0.0,<2.0.0",
        "pyserial>=3.5,<4.0",
        "deepdiff>=7.0.1,<9.0.0",
        "tqdm>=4.66.0,<5.0.0",
    ]
    bootstrap = (PIPE_ROOT / "bootstrap.sh").read_text()
    assert "--no-deps -e" in bootstrap
    assert "lerobot_runtime_specs" in bootstrap
    assert "Re-pinning numpy" not in bootstrap
    assert 'uv pip install "$(hub_spec)"' not in bootstrap


def test_hub_2_and_hub_1_are_re_pinned_and_0_35_stays():
    assert _call("hub_pin_action", "0.35.3") == "ok"
    assert _call("hub_pin_action", "0.34.2") == "ok"
    assert _call("hub_pin_action", "0.36.2") == "repin"
    assert _call("hub_pin_action", "1.31.0") == "repin"
    assert _call("hub_pin_action", "2.4.0") == "repin"
    assert _call("hub_pin_action", "none") == "repin"


def test_requirements_and_bootstrap_use_the_same_pin():
    spec = _call("hub_spec")
    assert spec == "huggingface_hub[cli]>=0.34.2,<0.36.0"
    assert spec in (PIPE_ROOT / "requirements.txt").read_text()
    bootstrap = (PIPE_ROOT / "bootstrap.sh").read_text()
    assert "hub_pin_action" in bootstrap
    assert "lerobot_rev" in bootstrap
    assert 'case "$HUBVER"' not in bootstrap
    assert "0.5.2" not in bootstrap
