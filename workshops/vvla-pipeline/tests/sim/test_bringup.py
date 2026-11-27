# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``so101_sim.bringup`` is the one DDS bring-up both entry scripts run."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_the_notebook_starts_the_pipeline_launch():
    """The notebook arm launch is ``run_pipeline.sh --digital-twin``'s arm launch."""
    from so101_sim.stack import command, launch_argv

    argv = command(arm=True, cameras=False, digital_twin=True, dry_run=False)
    assert argv[1:7] == ["-m", "so101_sim.stack", "--arm", "true", "--cameras", "false"]
    assert "--digital-twin" in argv and "true" in argv and "--dry-run" in argv
    text = " ".join(launch_argv(arm=True, cameras=False, digital_twin=True, dry_run=False))
    assert "strix_vla.launch.py" in text
    assert "arm:=true" in text and "cameras:=false" in text
    assert "digital_twin:=true" in text and "dry_run:=false" in text
    assert "notebook_twin" not in text and "scene:=" not in text
    script = (REPO / "run_pipeline.sh").read_text()
    notebooks = (REPO / "workshop" / "run_notebooks.sh").read_text()
    twin = (REPO / "workshop/common/twin_ros.py").read_text()
    assert 'python -m so101_sim.stack --arm "$1" --cameras "$2"' in script
    assert "python -m so101_sim.bringup pipeline" in script
    assert "python -m so101_sim.stack --arm false --cameras true --digital-twin true --dry-run false" in notebooks
    assert "python -m so101_sim.stack --arm true --cameras false --digital-twin true --dry-run false" in notebooks
    assert "python -m so101_sim.bringup" in notebooks and '"kernel"' in notebooks
    assert "--pipeline-twin-arm" not in script
    assert "popen" not in twin
    assert "ready_timeout_s" not in twin
    assert "notebook_twin" not in twin
    assert "use_private_domain(self.domain_id)" not in twin
    assert "child_domain_env" not in twin


def test_both_entry_scripts_run_the_bringup_module():
    pipeline = (REPO / "run_pipeline.sh").read_text()
    notebooks = (REPO / "workshop" / "run_notebooks.sh").read_text()
    assert "python -m so101_sim.bringup pipeline" in pipeline
    assert "python -m so101_sim.stack" in pipeline
    assert "python -m so101_sim.stack" in notebooks
    assert "so101_sim.bringup" in notebooks and '"kernel"' in notebooks


def test_the_notebook_launch_uses_the_pipeline_domain_and_stack():
    twin = (REPO / "workshop/common/twin_ros.py").read_text()
    bringup = (REPO / "so101_sim/src/so101_sim/bringup.py").read_text()
    assert "use_private_domain(self.domain_id)" not in twin
    assert "notebook_twin=" not in twin
    assert "start(twin_domain_id())" not in bringup
    assert "start(None)" in bringup


def test_the_detector_cell_does_not_start_dds_itself():
    nb = json.loads((REPO / "workshop/notebooks/03_integration/02_digital_twin.ipynb").read_text())
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "ObjectDetector" in src:
            assert "use_private_domain" not in src
            return
    pytest.fail("the detector cell is gone")


@pytest.mark.sim
def test_a_private_domain_bringup_pins_discovery_and_a_second_call_is_harmless():
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import os, sys
            os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
            os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"
            from so101_sim.bringup import ensure_dds_participant
            ensure_dds_participant(int(sys.argv[1]))
            ensure_dds_participant(int(sys.argv[1]))
            import rclpy
            assert rclpy.ok()
            assert rclpy.get_default_context().get_domain_id() == int(sys.argv[1])
            assert os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
            assert os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith("twin_localhost.xml")
        """), "121"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.sim
def test_the_pipeline_bringup_keeps_the_process_domain():
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import os
            os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
            os.environ["ROS_DOMAIN_ID"] = "7"
            from so101_sim.bringup import ensure_dds_participant
            ensure_dds_participant(None)
            import rclpy
            assert rclpy.get_default_context().get_domain_id() == 7
            assert "twin_localhost.xml" not in os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "")
        """)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
