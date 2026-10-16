# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Digital-twin scenario tests: fixtures, options, video bookkeeping.

    cd workshop && ./run_notebooks.sh --test     # everything, in the workshop README's setup
    pytest tests/sim -m "sim and not hw"         # tier 1: in-process, no ROS/LLM
    pytest tests/sim -m "hw and not docs"        # tier 2: full ROS graph
    pytest tests/sim -m docs                     # every notebook + the Instructor's corner
    pytest tests/sim --sim-yolo-device cpu       # YOLO on the CPU (machines without an NPU)
    pytest tests/sim --sim-video failed          # keep only failing runs' videos
    pytest tests/sim --sim-eyes seg              # ground-truth boxes instead of YOLO
    PROJECT=project pytest tests/sim -k workshop # students' workshop code

YOLO runs on the NPU, so source ``scripts/ryzen_ai_env.sh`` first.

Videos + timelines land in ``_artifacts/sim_tests/<tier>/`` with an
``index.html`` that plays them all.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PIPE_ROOT = HERE.parents[1]
WORKSHOP_PROJECT = PIPE_ROOT / "workshop" / os.environ.get("PROJECT", "solution")

# workshop tree first: common.config locates its project root on sys.path
for p in (str(HERE), str(PIPE_ROOT), str(WORKSHOP_PROJECT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")


def pytest_addoption(parser):
    g = parser.getgroup("sim", "SO-101 digital twin")
    g.addoption("--sim-video", choices=("all", "failed", "none"), default="all",
                help="which scenario recordings to keep (default: all)")
    g.addoption("--sim-video-dir", default=str(PIPE_ROOT / "_artifacts" / "sim_tests"),
                help="where recordings + index.html go")
    g.addoption("--sim-eyes", choices=("yolo", "seg"), default="yolo",
                help="arm-camera detector: the stack's YOLOv26s on the rendered frames, as in the "
                     "notebook (default), or ground-truth boxes from the twin's segmentation")
    g.addoption("--sim-yolo-device", choices=("cpu", "npu"), default="npu",
                help="where the YOLO eyes run (default npu, as in the notebooks; source "
                     "scripts/ryzen_ai_env.sh first, else the eyes fail instead of running on the CPU)")


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: digital-twin scenario (tier 1 unless also hw)")
    config.addinivalue_line("markers", "hw: tier 2 - full ROS 2 graph on the iGPU")
    config.addinivalue_line("markers", "slow: long-running scenario")
    config.addinivalue_line("markers", "docs: runs what workshop/README.md documents (notebooks, commands)")
    config._sim_results = {}


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    setattr(item, f"rep_{rep.when}", rep)
    return rep


def _tier(item) -> str:
    return "tier2" if item.get_closest_marker("hw") else "tier1"


def _slug(nodeid: str) -> str:
    name = nodeid.split("::", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def pytest_sessionfinish(session, exitstatus):
    from so101_sim.record import write_index

    results = getattr(session.config, "_sim_results", {})
    for tier_dir, res in results.items():
        if Path(tier_dir).is_dir():
            write_index(tier_dir, res, title=f"SO-101 digital-twin scenarios ({Path(tier_dir).name})")


# ---------------------------------------------------------------------------
# configs / services
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline_cfg():
    from vla_pipeline.utils.config import load_config

    cfg = load_config()
    cfg["behaviors"]["dance"]["duration_s"] = 5.0  # keep the dance scenario short
    cfg.setdefault("sim", {})["require_hw_gl"] = False
    return cfg


@pytest.fixture(scope="session")
def workshop_cfg():
    from common.config import load_config

    cfg = load_config()
    cfg.setdefault("dance", {})["duration_s"] = 5.0
    cfg.setdefault("sim", {})["require_hw_gl"] = False
    return cfg


@pytest.fixture(scope="session")
def mock_llm():
    """The workshop's llama-server imposter (keyword routing, records payloads)."""
    from common import fixtures

    if "pen" not in fixtures._OBJECT_WORDS:
        fixtures._OBJECT_WORDS = (*fixtures._OBJECT_WORDS, "pen")
    with fixtures.MockLlamaServer() as server:
        yield server


@dataclass
class Twin:
    world: object
    renderer: object


@pytest.fixture(scope="session")
def twins():
    """One SimWorld + RenderService per target stack, created on demand."""
    made: dict[str, Twin] = {}

    def get(target: str, cfg: dict) -> Twin:
        if target not in made:
            from so101_sim.render import RenderService
            from so101_sim.world import SimWorld

            if target == "pipeline":
                from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, REST_POSE
            else:
                from common.motion import GRIPPER_CLOSED_FLOOR, REST_POSE
            sim = dict(cfg.get("sim") or {})
            gm = dict(sim.get("gripper_map") or {})
            gm.setdefault("closed_cal", GRIPPER_CLOSED_FLOOR)
            sim["gripper_map"] = gm
            world = SimWorld({"sim": sim}, rest_pose=REST_POSE)
            made[target] = Twin(world, RenderService(world, require_hw=False))
        return made[target]

    yield get
    for t in made.values():
        t.renderer.close()
        t.world.close()


def _video_path(request, tier: str) -> Path | None:
    mode = request.config.getoption("--sim-video")
    if mode == "none":
        return None
    out = Path(request.config.getoption("--sim-video-dir")) / tier
    return out / f"{_slug(request.node.nodeid)}.mp4"


def _finish(request, rig, tier: str) -> None:
    item = request.node
    call = getattr(item, "rep_call", None)
    setup = getattr(item, "rep_setup", None)
    if call is None:
        status = "failed" if (setup is not None and setup.failed) else "skipped"
        message = ""
    else:
        xfail = hasattr(call, "wasxfail")
        if call.passed and not xfail:
            status = "passed"
        elif call.skipped and xfail:
            status = "xfailed"
        elif call.passed and xfail:
            status = "xpassed"
        else:
            status = "failed" if call.failed else "skipped"
        message = ""
        if call.failed and call.longrepr is not None:
            crash = getattr(call.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(call.longrepr)[-300:]
        elif xfail:
            message = str(call.wasxfail)
    mode = request.config.getoption("--sim-video")
    keep = mode == "all" or (mode == "failed" and status in ("failed", "xpassed"))
    path = rig.finish(status, message, keep=keep)
    if path is not None:
        results = request.config._sim_results.setdefault(str(path.parent), {})
        results[path.stem] = status


@pytest.fixture(scope="session")
def eyes(request, pipeline_cfg, workshop_cfg):
    """``eyes(target)``: that stack's own YOLOv26s ``ObjectDetector`` (one
    session each), or ``None`` with ``--sim-eyes seg``."""
    made: dict[str, object] = {}
    kind = request.config.getoption("--sim-eyes")
    device = request.config.getoption("--sim-yolo-device")

    def get(target: str):
        if kind == "seg":
            return None
        if target not in made:
            if target == "pipeline":
                from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

                cwd = os.getcwd()
                os.chdir(PIPE_ROOT)  # the pipeline config's model paths are repo-relative
                try:
                    made[target] = ObjectDetector(pipeline_cfg, device=device)
                finally:
                    os.chdir(cwd)
            else:
                from models.vision import ObjectDetector

                made[target] = ObjectDetector(workshop_cfg, device=device)
        providers = made[target].session.get_providers()
        if device == "npu" and "VitisAIExecutionProvider" not in providers:
            pytest.fail(
                f"--sim-yolo-device npu, but the {target} detector runs on {providers}: "
                "source scripts/ryzen_ai_env.sh with RYZEN_AI_WHEELS set",
                pytrace=False,
            )
        return made[target]

    return get


@pytest.fixture
def pipeline_rig(request, twins, pipeline_cfg, mock_llm, eyes):
    from rigs import PipelineRig

    mock_llm.requests.clear()
    rig = PipelineRig(
        twins("pipeline", pipeline_cfg), pipeline_cfg, mock_llm.base_url,
        recorder_path=_video_path(request, "tier1"), title=f"pipeline: {request.node.name}",
        detector=eyes("pipeline"),
    )
    yield rig
    _finish(request, rig, "tier1")


@pytest.fixture
def workshop_rig(request, twins, workshop_cfg, mock_llm, eyes):
    from rigs import WorkshopRig

    mock_llm.requests.clear()
    rig = WorkshopRig(
        twins("workshop", workshop_cfg), workshop_cfg, mock_llm.base_url,
        recorder_path=_video_path(request, "tier1"),
        title=f"workshop ({WORKSHOP_PROJECT.name}): {request.node.name}",
        detector=eyes("workshop"),
    )
    yield rig
    _finish(request, rig, "tier1")


@pytest.fixture
def video_path(request):
    """Per-test recording path for tier-2 rigs (None with --sim-video none)."""
    return _video_path(request, "tier2")


@pytest.fixture
def finish_rig(request):
    """Tier-2 tests register their rig here so it is stamped + closed."""
    holder = {}
    yield holder
    if "rig" in holder:
        _finish(request, holder["rig"], "tier2")
