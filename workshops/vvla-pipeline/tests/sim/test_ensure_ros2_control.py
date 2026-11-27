# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""scripts/ensure_ros2_control.sh must not need root to confirm the ROS libraries load."""

from __future__ import annotations

import subprocess

import pytest

from notebook_harness import PIPE_ROOT

pytestmark = [pytest.mark.sim, pytest.mark.timeout(120)]


def test_the_ros_library_check_does_not_ask_for_root():
    result = subprocess.run(
        [str(PIPE_ROOT / "scripts" / "ensure_ros2_control.sh")],
        cwd=PIPE_ROOT, capture_output=True, text=True, timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "sudo" not in output
    assert "password" not in output.lower()
