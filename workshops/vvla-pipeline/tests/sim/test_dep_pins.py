# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""bootstrap.sh asks scripts/dep_pins.sh whether huggingface-hub can stay.

LeRobot v0.5.0 and later require numpy 2 and huggingface-hub 1.x. An unpinned
clone resolved to 0.7.1, which installed hub 2.4.0. The old check treated every
version other than 1.x as acceptable, so 2.4.0 was logged OK and the
transformers import then failed.
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


def test_lerobot_pin_is_the_release_that_keeps_numpy_1_and_hub_0():
    assert _call("lerobot_rev") == "v0.4.4"


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
