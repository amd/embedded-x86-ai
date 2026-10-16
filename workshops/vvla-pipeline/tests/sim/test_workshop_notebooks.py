# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop README's happy path, headless: every notebook, in the README's order.

``workshop/README.md`` has the student run ``./run_notebooks.sh``, then the
notebooks in ``01_ai_models``, ``02_ros2`` and ``03_integration``. Here each
notebook runs top to bottom in a fresh ``vvla-workshop`` kernel, with
:mod:`notebook_harness`'s stand-ins for the rig's hardware and a scripted
student for its live cells. YOLO must run on the NPU, and the Llama
notebooks must get Llama's answers from the iGPU.

``cd workshop && ./run_notebooks.sh --test`` runs this after the README's setup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from notebook_harness import NOTEBOOKS, PIPE_ROOT, run_notebook
from twin_ros_rig import join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

README = PIPE_ROOT / "workshop" / "README.md"


def _readme_order() -> list[str]:
    m = re.search(r"Read and run them in order:(.*?)\.\s", README.read_text(encoding="utf-8"), re.S)
    return re.findall(r"`(\w+)`", m.group(1)) if m else []


ORDER = _readme_order()
PATHS = [path for folder in ORDER for path in sorted((NOTEBOOKS / folder).glob("*.ipynb"))]

# The student's actions in each live cell (by cell id); any other live cell is watched for 3 s.
SCRIPTS = {
    "03_igpu_llama": {
        "07f51d10d595": [("type", "Why do robots like grammars?"), ("wait_for", "llama: ", 90.0)],
    },
    "02_robot": {
        "cell-10": [("click", "rest pose"), ("sleep", 3.0)],
    },
    "03_intent_node": {
        "578bb604bee3": [
            ("type", "stop"), ("wait_for", "stop fast path", 15.0),
            ("type", "pick up the ball"), ("wait_for", "[LLM parse]", 90.0),
        ],
    },
    "01_behavior": {
        "cell-14": [("type", "wave at me"), ("sleep", 5.0)],
        "cell-16": [("click", "dance"), ("wait_for", "parked:", 90.0)],
    },
}

# What the notebook's own output must show.
EXPECT = {
    "03_igpu_llama": ["llama: "],
    "03_intent_node": ["[LLM parse]", "'pick_place'"],
}

# Notebooks that run YOLO.
YOLO = {"01_npu_yolo", "01_behavior", "02_digital_twin"}

_FALLBACK = re.compile(r"fall(s|ing)? back to (the )?cpu", re.I)


def test_readme_lists_every_notebook_folder():
    folders = sorted(
        p.name for p in NOTEBOOKS.iterdir()
        if p.is_dir() and not p.name.startswith(".") and any(p.glob("*.ipynb"))
    )
    assert ORDER == folders, f"workshop/README.md says run {ORDER}; the notebooks are in {folders}"


@pytest.mark.parametrize("path", PATHS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_notebook_runs(path: Path):
    join_test_domain()
    run = run_notebook(path, SCRIPTS.get(path.stem))
    yolo = [(model, provider) for model, provider in run.sessions if "yolo" in model.lower()]
    if path.stem in YOLO:
        assert yolo, f"{path.name} made no YOLO session"
    off_npu = [(Path(model).name, provider) for model, provider in yolo if provider != "VitisAIExecutionProvider"]
    assert not off_npu, f"{path.name}: YOLO ran off the NPU: {off_npu}"
    fallback = [line.strip() for line in run.output.splitlines() if _FALLBACK.search(line)]
    assert not fallback, f"{path.name} fell back to the CPU: {fallback}"
    for phrase in EXPECT.get(path.stem, ()):
        assert phrase in run.output, f"{path.name} never showed {phrase!r}"
