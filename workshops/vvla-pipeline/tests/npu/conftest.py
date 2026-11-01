# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""NPU load-order tests. Run them with ``./run_sim_tests.sh``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config):
    del config
    venv = (ROOT / ".venv").resolve()
    # .venv/bin/python symlinks to the system interpreter; sys.prefix does not.
    if Path(sys.prefix).resolve() != venv:
        pytest.exit(
            f"{sys.executable} is not {venv / 'bin' / 'python'}\n./run_sim_tests.sh",
            returncode=3,
        )
