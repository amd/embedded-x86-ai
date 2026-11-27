# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""run_pipeline.sh rejects an argument the orchestrator does not accept.

A mistyped flag used to be ignored, and the script then calibrated the real
arm. The script asks vla_pipeline.main.build_parser before it does any setup.
"""

from __future__ import annotations

import subprocess

import pytest

from notebook_harness import PIPE_ROOT
from vla_pipeline.main import build_parser

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]


def test_an_unknown_argument_stops_before_setup():
    result = subprocess.run(
        [str(PIPE_ROOT / "run_pipeline.sh"), "--digita-twin"],
        cwd=PIPE_ROOT, capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "digita-twin" in output
    assert "calibration" not in output.lower()
    assert "Starting the camera" not in output
    assert "Starting the arm" not in output


def test_help_prints_the_real_flags_and_stops():
    result = subprocess.run(
        [str(PIPE_ROOT / "run_pipeline.sh"), "--help"],
        cwd=PIPE_ROOT, capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "--digital-twin" in result.stdout
    assert "--dry-run" in result.stdout
    assert "calibration" not in output.lower()


def test_the_parser_accepts_the_launch_flags():
    args = build_parser().parse_args([
        "--digital-twin", "--text-commands", "--headless",
        "--dry-run", "--no-console", "--device", "cpu",
    ])
    assert args.digital_twin
    assert args.dry_run
    assert args.text_commands
    assert args.headless
    assert args.no_console
    assert args.device == "cpu"
