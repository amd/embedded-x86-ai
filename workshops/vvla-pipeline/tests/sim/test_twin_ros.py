# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop behaviors on the notebook twin, over ROS 2.

The context is the section 5 console's (``TwinRig``): ``TwinArmClient``,
the arm camera behind ``FanoutCamera``, the YOLO eyes, and the keyword
parser through ``handle_transcript``. The checks mirror the in-process
``test_workshop_scenarios.py``. Every reading of the twin comes from
``/sim/scene`` and ``/so101/joint_state``.

Tier 2 (ROS 2): ``source scripts/ryzen_ai_env.sh && source /opt/ros/jazzy/setup.bash && pytest tests/sim -m hw``.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np
import pytest

from rigs import HOLD_WATCH_S
from twin_ros_rig import BODY_JOINTS, Probe, TwinRig, default_scene_snapshot, join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(300)]
pytest.importorskip("rclpy")


@pytest.fixture(scope="module")
def probe(workshop_cfg):
    join_test_domain()
    p = Probe(workshop_cfg["robot"]["state_topic"])
    yield p
    p.close()


@pytest.fixture
def rig(workshop_cfg, eyes, probe):
    detector = eyes("workshop")
    if detector is None:
        pytest.skip("--sim-eyes seg reads the in-process twin's segmentation; this twin is over ROS")
    state = workshop_cfg["robot"]["state_topic"]
    assert probe.wait_publishers(state, 0, timeout=15), "a twin already runs on the test domain"
    r = TwinRig(workshop_cfg, detector, scene=default_scene_snapshot(workshop_cfg))
    yield r
    r.close()
    assert probe.wait_publishers(state, 0, timeout=20), "the twin node outlived its test"


def _sample(probe, joint: str, seconds: float, hz: float = 20.0) -> np.ndarray:
    values = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        j = probe.joints()
        if j:
            values.append(j[joint])
        time.sleep(1.0 / hz)
    return np.array(values)


class _GripLog(logging.Handler):
    """The held gripper goal of every confirmed grasp (``_log_grip``).

    A handler on the logger itself, not ``caplog``: with ROS 2 sourced,
    ROS's launch_testing pytest plugin swaps in its own logger class and
    ``caplog`` captures nothing.
    """

    def __init__(self):
        super().__init__(logging.INFO)
        self.goals: list[float] = []

    def emit(self, record) -> None:
        m = re.match(r"grasp: jaw [-\d.]+, goal ([-\d.]+)", record.getMessage())
        if m:
            self.goals.append(float(m.group(1)))


@pytest.fixture
def grip_log():
    handler = _GripLog()
    log = logging.getLogger("common.manipulation")
    level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    yield handler
    log.removeHandler(handler)
    log.setLevel(level)


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_twin_pickup(rig, grip_log, record_property, text, body, spoken):
    start = rig.twin.scene.latest()
    z0 = start["bodies"][body]["xyz"][2]

    rig.run(text, timeout=150)
    time.sleep(HOLD_WATCH_S)  # the hold must last, not just happen
    end = rig.twin.scene.latest(after=time.time())
    held = end["bodies"][body]

    holds = grip_log.goals
    record_property("hold_goal", holds[-1] if holds else None)
    print(f"twin grasp hold goal for {body}: {holds}")

    assert rig.said_any(f"Got the {spoken}"), rig.said
    assert held["grasped"], f"{spoken} is not between both fingers; robot said {rig.said}"
    assert held["xyz"][2] - z0 > 0.03, f"{spoken} rose {1000 * (held['xyz'][2] - z0):.0f} mm"
    x, y, z = held["xyz"]
    assert abs(x - 0.30) < 0.08 and abs(y) < 0.08, f"{spoken} held at ({x:.3f}, {y:.3f}), not in front"
    assert z > 0.08, f"{spoken} was set back down ({z:.3f} m)"
    moved = {
        name: float(np.linalg.norm(np.subtract(end["bodies"][name]["xyz"], pos["xyz"])))
        for name, pos in start["bodies"].items()
        if name != body
    }
    assert all(d < 0.01 for d in moved.values()), f"other props moved (m): {moved}"


def test_twin_drop_it_opens_the_jaws(rig, probe):
    rig.run("pick up the ball", timeout=150)
    assert rig.twin.scene.latest(after=time.time())["bodies"]["sports_ball"]["grasped"], rig.said
    rig.run("drop it", timeout=20)
    time.sleep(1.5)
    assert rig.said_any("Released"), rig.said
    assert probe.joints(after=time.time())["gripper"] > 80.0
    assert not rig.twin.scene.latest(after=time.time())["bodies"]["sports_ball"]["grasped"]


def test_twin_go_home_parks_the_arm(rig, probe):
    from common.motion import REST_POSE

    rig.twin.arm.go_to({"shoulder_pan": 35.0, "shoulder_lift": -40.0, "elbow_flex": 40.0})
    rig.run("go home", timeout=30)
    joints = probe.joints(after=time.time())
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, off


def test_twin_wave_swings_the_base(rig, probe):
    """``run_wave`` swings ``shoulder_pan`` between WAVE_LEFT (-25) and WAVE_RIGHT (+25)."""
    swings: list[np.ndarray] = []
    sampler = threading.Thread(target=lambda: swings.append(_sample(probe, "shoulder_pan", 10.0)))
    sampler.start()
    rig.run("wave at me", timeout=40)
    sampler.join()
    pan = swings[0]
    assert pan.max() - pan.min() > 40.0, f"shoulder_pan spanned {pan.max() - pan.min():.0f} deg"


def test_twin_dance_finishes_parked(rig, probe):
    from common.motion import REST_POSE

    rig.run("dance", timeout=60)
    time.sleep(1.0)
    joints = probe.joints(after=time.time())
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, off


def test_twin_mimic_drives_the_arm_then_stop_holds(rig, probe):
    """Mimic commands reach the twin, the pinch drives the gripper even after a
    grip was locked, and "stop" from another thread halts them.

    The scripted person moves the arm only a few degrees (in-process too), so
    the check is that the commands arrive, not how far they reach. The
    scripted hand opens and closes its pinch every few seconds.
    """
    rig.ctx.shared_state.update(grip_lock=True, grip_lock_pos=45.0)
    start = probe.joints()
    worker = rig.start("copy my movements")
    spans = {k: 0.0 for k in BODY_JOINTS}
    jaw = []
    deadline = time.time() + 4.0
    while time.time() < deadline:
        j = probe.joints()
        for k in BODY_JOINTS:
            spans[k] = max(spans[k], abs(j[k] - start[k]))
        jaw.append(j["gripper"])
        time.sleep(0.05)
    assert max(spans.values()) > 2.0, f"mimic did not move the arm: {spans}"
    assert "grip_lock" not in rig.ctx.shared_state, "mimic left the grip locked"
    assert max(jaw) - min(jaw) > 5.0, f"the pinch did not move the gripper: {min(jaw):.1f}..{max(jaw):.1f}"
    rig.stop()
    worker.join(10.0)
    assert not worker.is_alive(), "mimic kept running after stop"
    time.sleep(1.0)
    first = probe.joints(after=time.time())
    time.sleep(1.0)
    second = probe.joints(after=time.time())
    drift = {k: round(second[k] - first[k], 2) for k in BODY_JOINTS if abs(second[k] - first[k]) > 1.0}
    assert not drift, f"the arm kept moving after stop: {drift}"
