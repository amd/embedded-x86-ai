# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``scripts/install_kernel.sh --register-only`` rewrites the shared kernel for this clone.

The ``vvla-workshop`` kernel.json lives in the user's Jupyter directory, which
every clone on the machine shares. ``workshop/run_notebooks.sh`` calls
``--register-only`` on every launch. These tests use a throwaway Jupyter data
directory (``JUPYTER_DATA_DIR``), so the user's own kernel is not touched.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PIPE_ROOT = Path(__file__).resolve().parents[2]
KERNEL = "vvla-workshop"
INSTALL = PIPE_ROOT / "scripts" / "install_kernel.sh"

pytestmark = [pytest.mark.sim, pytest.mark.timeout(120)]


@pytest.fixture
def jupyter_data(tmp_path):
    data = tmp_path / "jupyter-data"
    spec_dir = data / "kernels" / KERNEL
    spec_dir.mkdir(parents=True)
    return data, spec_dir / "kernel.json"


def _register(data) -> subprocess.CompletedProcess:
    env = {**os.environ, "JUPYTER_DATA_DIR": str(data)}
    return subprocess.run(
        [str(INSTALL), "--register-only"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_kernel_left_by_another_clone_is_replaced(jupyter_data):
    """The shared kernel.json takes this clone's venv, not the one that wrote it last."""
    data, spec = jupyter_data
    spec.write_text(json.dumps({
        "argv": ["/tmp/other-clone/.venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (VVLA workshop)",
        "language": "python",
        "env": {"VIRTUAL_ENV": "/tmp/other-clone/.venv"},
    }))

    done = _register(data)
    assert done.returncode == 0, done.stdout + done.stderr

    written = json.loads(spec.read_text())
    venv_python = str(PIPE_ROOT / ".venv" / "bin" / "python")
    assert written["argv"][0] == venv_python
    assert written["env"]["VIRTUAL_ENV"] == str(PIPE_ROOT / ".venv")
    assert written["env"]["VIRTUAL_ENV"] != "/tmp/other-clone/.venv"
    if Path("/opt/ros/jazzy/setup.bash").is_file():
        assert written["env"].get("AMENT_PREFIX_PATH")


def test_registering_twice_leaves_the_same_kernel(jupyter_data):
    data, spec = jupyter_data
    first = _register(data)
    assert first.returncode == 0, first.stdout + first.stderr
    first_text = spec.read_text()
    second = _register(data)
    assert second.returncode == 0, second.stdout + second.stderr
    assert spec.read_text() == first_text
