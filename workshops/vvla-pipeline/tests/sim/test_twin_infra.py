# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The notebook twin's plumbing: ``so101_sim.node --notebook-twin`` and ``common.twin_ros``.

Tier 2 (ROS 2): ``source scripts/ryzen_ai_env.sh && source /opt/ros/jazzy/setup.bash && pytest tests/sim -m hw``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from twin_ros_rig import (
    BODY_JOINTS,
    EXTENDED,
    TEST_DOMAIN_ID,
    Probe,
    default_scene_snapshot,
    join_test_domain,
    quat_close,
    xml_poses,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(240)]
pytest.importorskip("rclpy")

SOLUTION = Path(__file__).resolve().parents[2] / "workshop" / "solution"


@pytest.fixture(scope="module")
def probe(workshop_cfg):
    join_test_domain()
    p = Probe(workshop_cfg["robot"]["state_topic"])
    yield p
    p.close()


@pytest.fixture
def state_topic(workshop_cfg):
    return workshop_cfg["robot"]["state_topic"]


@pytest.fixture
def no_twin(probe, state_topic):
    """No twin on the test domain before the test, and none left after it."""
    assert probe.wait_publishers(state_topic, 0, timeout=15), "a twin already runs on the test domain"
    yield
    assert probe.wait_publishers(state_topic, 0, timeout=20), "a twin node outlived its test"


@pytest.fixture
def twin(workshop_cfg, no_twin):
    from common.twin_ros import TwinSession

    session = TwinSession(workshop_cfg)
    yield session
    session.close()


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_default_launch_is_unchanged(workshop_cfg, probe, state_topic, no_twin):
    """Without --notebook-twin: the workshop CameraNode, no twin extras, the live-mimic clamp."""
    from common.config import DEFAULT_CONFIG
    from common.motion import GRASP_LIMITS
    from common.twin_ros import TwinArmClient

    log = tempfile.NamedTemporaryFile(prefix="twin_default_", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "so101_sim.node", "--config", str(DEFAULT_CONFIG),
         "--stack", "workshop", "--sim-cameras", "arm"],
        stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    try:
        assert probe.wait_publishers(state_topic, 1, timeout=60), Path(log.name).read_text()
        assert probe.wait_publishers("/cameras/arm/healthy", 1, timeout=20), "no workshop CameraNode"
        assert "/sim/scene" not in probe.topics()
        assert "/sim/reset_scene" not in probe.services()
        arm = TwinArmClient(workshop_cfg)
        try:
            arm.go_to(EXTENDED)
            arm.go_to({"wrist_flex": 90.0}, limits=GRASP_LIMITS)
            wrist = arm.get_joints()["wrist_flex"]
        finally:
            arm.close()
        # LIVE_MIMIC_LIMITS caps the wrist at 84.4; move_to settles within 2.5 deg
        assert 81.0 <= wrist <= 84.9, f"wrist settled at {wrist:.1f}"
    finally:
        _stop(proc)
        log.close()
        Path(log.name).unlink(missing_ok=True)


def test_grasp_moves_keep_the_grasp_envelope(twin):
    from common.motion import GRASP_LIMITS

    twin.arm.go_to(EXTENDED)
    twin.arm.go_to({"wrist_flex": 90.0}, limits=GRASP_LIMITS)
    grasp = twin.arm.get_joints()["wrist_flex"]
    twin.arm.go_to({"wrist_flex": 0.0})
    twin.arm.go_to({"wrist_flex": 90.0})
    desk = twin.arm.get_joints()["wrist_flex"]
    assert grasp >= 87.5, f"a GRASP_LIMITS move settled at {grasp:.1f}, not near 90"
    assert 64.0 <= desk <= 68.0, f"an ordinary move settled at {desk:.1f}, not at the 67.5 desk cap"


def test_frames_are_stamped_after_the_arm_settles(twin):
    from common.twin_console import FanoutCamera

    twin.arm.go_to({"shoulder_pan": 20.0})
    settled_at = time.time()
    ok, frame = twin.armcam.read_after(settled_at)
    assert ok and frame is not None
    assert twin.armcam.last_stamp > settled_at

    fan = FanoutCamera(twin.armcam)
    try:
        twin.arm.go_to({"shoulder_pan": -20.0})
        settled_at = time.time()
        ok, frame = fan.read_after(settled_at)
        assert ok and frame is not None
        assert fan._wall > settled_at, "FanoutCamera judged the frame by arrival, not capture"
    finally:
        fan.close()


_OTHER_DOMAIN = r"""
import sys, time
import rclpy
from sensor_msgs.msg import JointState
rclpy.init()
node = rclpy.create_node("other_domain_probe")
pub = node.create_publisher(JointState, "/so101/joint_state", 1)
seen = set()
deadline = time.time() + float(sys.argv[1])
while time.time() < deadline:
    msg = JointState()
    msg.name = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    msg.position = [0.0] * 6
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.05)
    seen.update(name for name, _types in node.get_topic_names_and_types())
print("SEEN", " ".join(sorted(seen)), flush=True)
"""


def test_the_twin_domain_is_private(workshop_cfg, no_twin):
    """A joint-state publisher on another domain is invisible to the twin, and the twin to it."""
    from common.twin_ros import TwinSession

    env = {**os.environ, "ROS_DOMAIN_ID": str(TEST_DOMAIN_ID + 1)}
    other = subprocess.Popen(
        [sys.executable, "-c", _OTHER_DOMAIN, "20"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        time.sleep(2.0)  # let the other domain's publisher announce itself
        session = TwinSession(workshop_cfg)
        try:
            joints = session.arm.get_joints()
            assert joints["shoulder_lift"] == pytest.approx(-96.5, abs=2.0), joints
            assert session.scene.latest(after=time.time()) is not None
        finally:
            session.close()
        out, _ = other.communicate(timeout=40)
    finally:
        if other.poll() is None:
            other.kill()
    seen = next(line for line in out.splitlines() if line.startswith("SEEN")).split()[1:]
    assert "/sim/scene" not in seen and "/cameras/arm/image_raw" not in seen, seen


def test_a_second_twin_on_the_domain_is_refused(workshop_cfg, twin):
    from common.twin_ros import TwinSession

    with pytest.raises(RuntimeError, match="already publishes"):
        TwinSession(workshop_cfg)
    assert twin.scene.latest(after=time.time()) is not None, "the first twin stopped"
    assert twin.armcam.read()[0]


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def test_the_node_dies_with_its_kernel(probe, state_topic, no_twin):
    from jupyter_client import KernelManager

    km = KernelManager(kernel_name="vvla-workshop")
    km.start_kernel(env=os.environ.copy())
    kc = km.client()
    kc.start_channels()
    out: list[str] = []
    try:
        kc.wait_for_ready(timeout=60)
        code = (
            f"import sys; sys.path.insert(0, {str(SOLUTION)!r})\n"
            "from common.config import load_config\n"
            "from common.twin_ros import TwinSession\n"
            "twin = TwinSession(load_config())\n"
            "print('TWIN_PID', twin.pid)\n"
        )

        def hook(msg):
            if msg["msg_type"] == "stream":
                out.append(msg["content"]["text"])
            elif msg["msg_type"] == "error":
                out.append("\n".join(msg["content"]["traceback"]))

        reply = kc.execute_interactive(code, timeout=120, output_hook=hook)
        assert reply["content"]["status"] == "ok", "".join(out)
        pid = int(next(t for t in "".join(out).split("\n") if t.startswith("TWIN_PID")).split()[1])
        assert probe.wait_publishers(state_topic, 1, timeout=10)
        os.kill(km.provisioner.process.pid, signal.SIGKILL)
        deadline = time.time() + 5.0
        while not _gone(pid) and time.time() < deadline:
            time.sleep(0.1)
        assert _gone(pid), "the twin node outlived its kernel"
        assert probe.wait_publishers(state_topic, 0, timeout=15)
    finally:
        kc.stop_channels()
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


def test_reset_service_restores_the_scene_and_parks_the_arm(workshop_cfg, no_twin):
    from common.motion import REST_POSE
    from common.twin_ros import TwinSession

    moved = default_scene_snapshot(workshop_cfg)
    moved["bodies"]["sports_ball"][:3] = [0.22, -0.04, 0.015]
    home = xml_poses(workshop_cfg)
    session = TwinSession(workshop_cfg, scene=moved)
    try:
        start = session.scene.latest()
        assert start["bodies"]["sports_ball"]["xyz"][:2] == pytest.approx([0.22, -0.04], abs=0.01)
        session.arm.go_to({"shoulder_pan": 30.0, "shoulder_lift": -40.0, "elbow_flex": 40.0})
        assert session.reset_scene()
        time.sleep(1.5)  # the props settle; the servos reach the parked goal
        scene = session.scene.latest(after=time.time())
        for name, qpos in home.items():
            got = scene["bodies"][name]
            assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
            assert quat_close(got["quat"], qpos[3:7]), name
            assert not got["grasped"], name
        joints = session.arm.get_joints()
        for k in BODY_JOINTS:
            assert joints[k] == pytest.approx(REST_POSE[k], abs=3.0), (k, joints)
        assert joints["gripper"] >= 70.0, joints
        session.arm.go_to({"shoulder_pan": 20.0})
        assert session.arm.get_joints()["shoulder_pan"] > 15.0, "the server drove back to a stale target"
    finally:
        session.close()
