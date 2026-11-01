# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The notebook twin's plumbing: ``so101_sim.node --notebook-twin`` and ``common.twin_ros``.

Tier 2 (ROS 2): ``./run_sim_tests.sh``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from twin_ros_rig import (
    BODY_JOINTS,
    EXTENDED,
    TEST_DOMAIN_ID,
    Probe,
    assert_domain_free,
    default_scene_snapshot,
    join_test_domain,
    quat_close,
    wait_exit,
    xml_poses,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(240)]
pytest.importorskip("rclpy")

SOLUTION = Path(__file__).resolve().parents[2] / "workshop" / "solution"


@pytest.fixture(scope="module")
def probe(workshop_cfg):
    join_test_domain()
    p = Probe(workshop_cfg)
    yield p
    p.close()


@pytest.fixture
def no_twin():
    """No twin on the test domain before the test, and none left after it."""
    assert_domain_free("a twin already runs on the test domain")
    yield
    assert_domain_free("a twin node outlived its test", wait=True)


@pytest.fixture
def twin(workshop_cfg, no_twin):
    from common.twin_ros import TwinSession

    session = TwinSession(workshop_cfg)
    yield session
    session.close()


def _stop(proc: subprocess.Popen) -> None:
    """SIGINT, which rclpy turns into a clean shutdown, then the process's exit."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        proc.wait()


def _first_state_or_exit(probe, proc: subprocess.Popen) -> None:
    """Block until a joint state arrives or ``proc`` exits, whichever is first."""
    up = threading.Event()
    unsubscribe = probe.subscribe(up.set)
    threading.Thread(target=lambda: (proc.wait(), up.set()), daemon=True).start()
    try:
        up.wait()
    finally:
        unsubscribe()


def test_default_launch_is_unchanged(workshop_cfg, probe, no_twin):
    """Without --notebook-twin: the workshop CameraNode, no scene topic, the
    live-mimic clamp. The reset service is every twin's."""
    from common.config import DEFAULT_CONFIG
    from common.motion import GRASP_LIMITS
    from ros.arm_client import Ros2ArmClient
    from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, start_controllers, stop_controllers
    from so101_sim.node import PROPS_SETTLED_TOPIC, RESET_SERVICE

    from std_msgs.msg import Bool

    r = workshop_cfg["robot"]
    log = tempfile.NamedTemporaryFile(prefix="twin_default_", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "so101_sim.node", "--config", str(DEFAULT_CONFIG),
         "--stack", "workshop", "--sim-cameras", "arm",
         "--ros-args", "-r", f"{r['command_topic']}:={HARDWARE_COMMANDS}",
         "-r", f"{r['state_topic']}:={HARDWARE_STATES}"],
        stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    controllers = start_controllers(DEFAULT_CONFIG, "workshop", log=log)
    try:
        _first_state_or_exit(probe, proc)
        assert proc.poll() is None, Path(log.name).read_text()
        probe.first_message("/cameras/arm/healthy", Bool)
        assert PROPS_SETTLED_TOPIC not in probe.topics()
        arm = Ros2ArmClient(workshop_cfg)
        try:
            assert RESET_SERVICE in probe.services()
            arm.go_to(EXTENDED)
            arm.go_to({"wrist_flex": 90.0}, limits=GRASP_LIMITS)
            wrist = arm.get_joints()["wrist_flex"]
        finally:
            arm.close()
        # LIVE_MIMIC_LIMITS caps the wrist at 84.4 in the server, so the 90 deg goal ends there
        assert 81.0 <= wrist <= 84.9, f"wrist settled at {wrist:.1f}"
    finally:
        stop_controllers(controllers)
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


def test_the_props_are_tf_frames_with_a_typed_status(twin, probe):
    """Standard ROS messages: a prop's pose through tf2, the settled flag as
    std_msgs/Bool, the gripper's state on /diagnostics."""
    import rclpy.time
    from rclpy.duration import Duration
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformListener

    from common.motion import GRIPPER_TORQUE_LIMIT
    from common.ros2_control_arm import REPLY_S
    from common.ros_helpers import Spinner
    from so101_sim.node import PROPS_SETTLED_TOPIC, WORLD_FRAME

    scene = twin.scene.latest(after=time.time(), settled=True, timeout=None)
    spin = Spinner("tf_probe")
    try:
        buffer = Buffer()
        TransformListener(buffer, spin.node, spin_thread=False)
        tf = buffer.lookup_transform(WORLD_FRAME, "sports_ball", rclpy.time.Time(), timeout=Duration(seconds=REPLY_S))
        t = tf.transform.translation
        assert [t.x, t.y, t.z] == pytest.approx(scene["bodies"]["sports_ball"]["xyz"], abs=0.005)
    finally:
        spin.close()
    probe.first_message(PROPS_SETTLED_TOPIC, Bool)
    assert scene["gripper"]["torque_limit"] == GRIPPER_TORQUE_LIMIT and not scene["gripper"]["tripped"], scene


def test_a_goal_sent_as_soon_as_the_session_is_ready_is_taken(twin):
    """The session is ready once the controllers are; a goal sent at once arrives."""
    assert twin.arm.move_to({"shoulder_pan": 15.0}) == "arrived"


_OTHER_DOMAIN = r"""
import sys, threading
import rclpy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
rclpy.init()
node = rclpy.create_node("other_domain_probe")
pub = node.create_publisher(JointState, "/so101/joint_state", 1)


def publish():
    msg = JointState()
    msg.name = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    msg.position = [0.0] * 6
    pub.publish(msg)


def report():
    sys.stdin.read()  # the test closes stdin once the twin is up
    print("SEEN", " ".join(sorted(name for name, _types in node.get_topic_names_and_types())), flush=True)
    rclpy.shutdown()


node.create_timer(0.05, publish)
threading.Thread(target=report, daemon=True).start()
print("UP", flush=True)
try:
    rclpy.spin(node)
except ExternalShutdownException:
    pass
"""


def test_the_twin_domain_is_private(workshop_cfg, no_twin):
    """A joint-state publisher on another domain is invisible to the twin, and the
    twin to it: once the twin and its clients are up, that domain lists none of its topics."""
    from common.twin_ros import TwinSession
    from so101_sim.node import PROPS_SETTLED_TOPIC

    env = {**os.environ, "ROS_DOMAIN_ID": str(TEST_DOMAIN_ID + 1)}
    other = subprocess.Popen(
        [sys.executable, "-c", _OTHER_DOMAIN], env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in other.stdout:
            if line.startswith("UP"):
                break
        else:
            pytest.fail("the other domain's publisher exited before it started")
        session = TwinSession(workshop_cfg)
        try:
            joints = session.arm.get_joints()
            assert joints["shoulder_lift"] == pytest.approx(-96.5, abs=2.0), joints
            assert session.scene.latest(after=time.time(), timeout=None) is not None
            other.stdin.close()
            out = other.stdout.read()
            other.wait()
        finally:
            session.close()
    finally:
        if other.poll() is None:
            other.kill()
    seen = next(line for line in out.splitlines() if line.startswith("SEEN")).split()[1:]
    assert PROPS_SETTLED_TOPIC not in seen and "/cameras/arm/image_raw" not in seen, seen


def test_a_second_twin_on_the_domain_is_refused(workshop_cfg, twin):
    from common.twin_ros import TwinSession

    with pytest.raises(RuntimeError, match="already runs on ROS domain"):
        TwinSession(workshop_cfg)
    assert twin.scene.latest(after=time.time(), timeout=None) is not None, "the first twin stopped"
    assert twin.armcam.read()[0]


def test_the_node_dies_with_its_kernel(no_twin):
    from jupyter_client import KernelManager

    km = KernelManager(kernel_name="vvla-workshop")
    km.start_kernel(env=os.environ.copy())
    kc = km.client()
    kc.start_channels()
    out: list[str] = []
    try:
        kc.wait_for_ready(timeout=None)
        code = (
            f"import sys; sys.path.insert(0, {str(SOLUTION)!r})\n"
            "from common.config import load_config\n"
            "from common.twin_ros import TwinSession\n"
            "twin = TwinSession(load_config())\n"
            "print('TWIN_PID', twin.pid, twin.controllers_pid)\n"
        )

        def hook(msg):
            if msg["msg_type"] == "stream":
                out.append(msg["content"]["text"])
            elif msg["msg_type"] == "error":
                out.append("\n".join(msg["content"]["traceback"]))

        reply = kc.execute_interactive(code, timeout=None, output_hook=hook)
        assert reply["content"]["status"] == "ok", "".join(out)
        pid, controllers = map(int, next(t for t in "".join(out).split("\n") if t.startswith("TWIN_PID")).split()[1:])
        os.kill(km.provisioner.process.pid, signal.SIGKILL)
        assert wait_exit(pid), "the twin node outlived its kernel"
        assert wait_exit(controllers), "the controllers outlived their kernel"
    finally:
        kc.stop_channels()
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


def test_reset_service_restores_the_scene_and_parks_the_arm(workshop_cfg, probe, no_twin):
    """The reset parks the arm through the controllers, then puts the props
    back, so the server's first report after it shows the arm parked."""
    from common.motion import REST_POSE
    from common.twin_ros import TwinSession

    moved = default_scene_snapshot(workshop_cfg)
    moved["bodies"]["sports_ball"][:3] = [0.22, -0.04, 0.015]
    home = xml_poses(workshop_cfg)
    session = TwinSession(workshop_cfg, scene=moved)
    try:
        start = session.scene.latest(timeout=None)
        assert start["bodies"]["sports_ball"]["xyz"][:2] == pytest.approx([0.22, -0.04], abs=0.01)
        session.arm.go_to({"shoulder_pan": 30.0, "shoulder_lift": -40.0, "elbow_flex": 40.0})
        assert session.reset_scene()
        reset_at = time.time()
        joints = probe.joints(after=reset_at)
        scene = session.scene.latest(after=reset_at, settled=True, timeout=None)
        for name, qpos in home.items():
            got = scene["bodies"][name]
            assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
            assert quat_close(got["quat"], qpos[3:7]), name
            assert not got["grasped"], name
        for k in BODY_JOINTS:
            assert joints[k] == pytest.approx(REST_POSE[k], abs=3.0), (k, joints)
        assert joints["gripper"] >= 70.0, joints
        session.arm.go_to({"shoulder_pan": 20.0})
        assert session.arm.get_joints()["shoulder_pan"] > 15.0, "the server drove back to a stale target"
    finally:
        session.close()


def _line(proc: subprocess.Popen, prefix: str) -> str:
    """The first stdout line starting with ``prefix``. The process exiting first is a failure."""
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(prefix):
            return line.rstrip()
    pytest.fail(f"process exited before {prefix!r}")


_DISCOVERY_SUB = """
import os, sys
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"
sys.path.insert(0, sys.argv[1])
from common.twin_ros import use_private_domain
use_private_domain(int(sys.argv[2]))
import rclpy
from std_msgs.msg import String
node = rclpy.create_node("twin_discovery_sub")
def on_probe(msg):
    print("SEEN", flush=True)
node.create_subscription(String, "/twin_probe", on_probe, 10)
print("READY", flush=True)
rclpy.spin(node)
"""

_DISCOVERY_PUB = """
import os, sys
from pathlib import Path
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"
sys.path.insert(0, sys.argv[1])
from common.twin_ros import use_private_domain
use_private_domain(int(sys.argv[2]))
assert os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
assert os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith("twin_localhost.xml")
assert os.environ.get("FASTDDS_BUILTIN_TRANSPORTS") == "LARGE_DATA"
import rclpy
from std_msgs.msg import String
node = rclpy.create_node("twin_discovery_pub")
inodes = set()
for entry in Path("/proc/self/fd").iterdir():
    try:
        target = os.readlink(entry)
    except OSError:
        continue
    if target.startswith("socket:[") and target.endswith("]"):
        inodes.add(target[len("socket:["):-1])
listening = 0
for name in ("tcp", "tcp6"):
    table = Path(f"/proc/self/net/{name}")
    if not table.exists():
        continue
    for row in table.read_text().splitlines()[1:]:
        parts = row.split()
        if len(parts) > 9 and parts[3] == "0A" and parts[9] in inodes:
            listening += 1
print(f"TCP {listening}", flush=True)
pub = node.create_publisher(String, "/twin_probe", 10)
def tick():
    msg = String()
    msg.data = "hi"
    pub.publish(msg)
node.create_timer(0.05, tick)
rclpy.spin(node)
"""


def test_localhost_discovery_keeps_large_data_transports():
    """Two processes on the private domain see each other, and LARGE_DATA stays on.

    The publisher is started with ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST``.
    That range would disable builtin transports, so the private-domain setup
    clears it. A TCP listen socket is what ``FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA``
    opens and that range does not.
    """
    domain = "94"
    sub = subprocess.Popen(
        [sys.executable, "-c", _DISCOVERY_SUB, str(SOLUTION), domain],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy(),
    )
    pub = None
    try:
        assert _line(sub, "READY") == "READY"
        pub = subprocess.Popen(
            [sys.executable, "-c", _DISCOVERY_PUB, str(SOLUTION), domain],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy(),
        )
        tcp = _line(pub, "TCP")
        assert int(tcp.split()[1]) >= 1, tcp
        assert _line(sub, "SEEN") == "SEEN"
    finally:
        for proc in (pub, sub):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait()
