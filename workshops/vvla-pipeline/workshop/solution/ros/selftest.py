# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Component 2 self-test harness (PROVIDED) - run: ``python -m ros.selftest``

Exercises YOUR node wiring end-to-end with synthetic backends - a synthetic
camera instead of V4L2, a dry-run arm instead of the Feetech bus, a keyword
brain instead of llama-server - so every topic round-trip is real even with
zero hardware attached.

Transport: real rclpy/DDS in-process. ROS 2 is required (``source
/opt/ros/jazzy/setup.bash``). The arm checks run the arm under ros2_control,
as the rig does: the controllers start with ``launch/so101_control.launch.py``.

The on-hardware equivalents (real cameras, real arm, ros2 CLI inspection)
are listed at the end of each ROS notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from common.config import PROJECT_ROOT, load_config
from common.feedback import Reporter

ARTIFACTS = PROJECT_ROOT / "_artifacts"


class _Runner:
    """Spin a node on a background executor; stop cleanly.

    A thread left inside ``executor.spin()`` aborts the process at exit
    ("terminate called without an active exception"), so the thread spins
    ``spin_once`` in a loop that ``stop`` ends and joins.
    """

    def __init__(self, node):
        from rclpy.executors import SingleThreadedExecutor

        self.node = node
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._running:
            self.executor.spin_once(timeout_sec=0.05)

    def stop(self):
        """Stop the spin thread, shut down the executor and destroy the node."""
        self._running = False
        self._thread.join(timeout=2.0)
        self.executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass


def _graph_feedback(topic: str) -> str:
    """Explain why a topic round-trip went silent."""
    return (
        f"nothing arrived on {topic} - check the topic names and QoS "
        "on both ends (ros2 topic list / ros2 topic info -v)"
    )


def _wait_for(predicate, timeout_s: float, interval: float = 0.02):
    """Poll ``predicate`` until truthy or ``timeout_s`` elapses; return the value or None."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


# =============================================================================
# Checks
# =============================================================================


def check_cameras(rep: Reporter, cfg: dict) -> None:
    """Verify the camera node and client wiring: construction, frame round-trip, health flag."""
    import cv2

    from common.fixtures import SyntheticCamera
    from ros.camera_client import Ros2Camera
    from ros.camera_node import CameraNode

    defaults = dict(cfg["cameras"]["mount"])
    topic = defaults["topic"]

    ok_node, node = rep.run(
        "2.1",
        "camera node constructs (publishers + timer)",
        lambda: CameraNode("mount", defaults, frame_source=SyntheticCamera()),
        hint="create both publishers with best-effort/keep-last/depth-1 QoS "
        "and a timer at 1/fps driving self._tick",
    )
    if not ok_node:
        rep.add("2.2", "camera node publishes frames", "SKIP", "needs TODO 2.1")
        rep.add("2.3", "camera client receives frames", "SKIP", "needs TODO 2.1")
        return
    runner = _Runner(node)

    client = None
    try:
        ok_client, client = rep.run(
            "2.3",
            "camera client constructs (subscription + executor)",
            lambda: Ros2Camera(
                "mount", topic, stale_after_s=defaults.get("stale_after_s", 1.0)
            ),
            hint="subscribe with BEST-EFFORT QoS - the node publishes "
            "best-effort, and a RELIABLE subscriber never matches it",
        )
        if not ok_client:
            rep.add(
                "2.2", "frame round-trip over the image topic", "SKIP", "needs TODO 2.3"
            )
            return

        def roundtrip():
            """Verify a frame published by the node arrives correctly decoded at the client."""
            got = _wait_for(lambda: client.read()[0], 3.0)
            assert got, _graph_feedback(topic)
            ok, frame = client.read()
            assert ok and frame is not None
            assert frame.shape == (defaults["height"], defaults["width"], 3), (
                f"frame shape {frame.shape} != "
                f"({defaults['height']}, {defaults['width']}, 3) - encoding or "
                "step mishandled in the Image message"
            )
            assert int(frame[0, -1, 0]) > 180, (
                "pixels don't match the synthetic pattern - the frame was "
                "re-encoded wrong (bgr8 bytes should pass through unchanged)"
            )
            out = frame.copy()
            cv2.putText(
                out,
                f"round-trip OK via {topic}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            path = ARTIFACTS / "camera_roundtrip.png"
            cv2.imwrite(str(path), out)
            rep.artifact(path)

        rep.run("2.2", "frame round-trip over the image topic", roundtrip)

        # Health flag (TODO 2.2 publishes it alongside every frame).
        def health():
            """Verify the health flag publishes True alongside a live frame source."""
            import rclpy
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import Bool

            probe = rclpy.create_node("selftest_health_probe")
            seen = []
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            probe.create_subscription(
                Bool, "/cameras/mount/healthy", lambda m: seen.append(m.data), qos
            )
            r = _Runner(probe)
            got = _wait_for(lambda: seen, 2.0)
            r.stop()
            assert got, _graph_feedback("/cameras/mount/healthy")
            assert seen[-1] is True, "healthy flag is False with a live source"

        rep.run("2.2", "health flag rides alongside the frames", health)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        runner.stop()


def check_arm(rep: Reporter, cfg: dict) -> None:
    """Verify the arm server node and client wiring over ros2_control:
    construction, state, commands, clamping."""
    from so101_sim.control import HARDWARE_COMMANDS, HARDWARE_STATES, start_controllers, stop_controllers

    from common.config import DEFAULT_CONFIG
    from common.motion import JOINT_LIMITS, REST_POSE, DryRunArm
    from ros.arm_client import Ros2ArmClient
    from ros.arm_node import So101ServerNode

    # The server on the hardware side's topics, as the launch remaps them.
    hardware = {**cfg, "robot": {**cfg["robot"], "command_topic": HARDWARE_COMMANDS, "state_topic": HARDWARE_STATES}}
    ok_node, node = rep.run(
        "2.4",
        "arm server node constructs (cmd sub + state pub)",
        lambda: So101ServerNode(hardware, arm=DryRunArm()),
        hint="subscription on robot.command_topic → _on_command; publisher "
        "on robot.state_topic; best-effort/keep-last/depth-1 both ends",
    )
    if not ok_node:
        for t in ("2.5", "2.6", "2.7", "2.8", "2.9"):
            rep.add(t, "arm checks", "SKIP", "needs TODO 2.4")
        return
    runner = _Runner(node)
    controllers_log = ARTIFACTS / "selftest_controllers.log"
    with open(controllers_log, "w") as log:
        controllers = start_controllers(DEFAULT_CONFIG, "workshop", log=log)

    arm = None
    try:
        ok_client, arm = rep.run(
            "2.7",
            "arm client constructs and sees the controllers' state",
            lambda: Ros2ArmClient(cfg),
            hint="if this times out: TODO 2.7 must subscribe to /joint_states "
            "and create the trajectory action client, and TODO 2.6 must publish "
            f"state for the controllers to start (their log: {controllers_log})",
        )
        if not ok_client:
            for t in ("2.5", "2.6", "2.8", "2.9"):
                rep.add(t, "arm round-trip checks", "SKIP", "needs TODO 2.7")
            return

        def state_is_rest():
            """Verify the state stream reports all six joints at the rest pose."""
            joints = arm.get_joints()
            for k, v in REST_POSE.items():
                got = joints.get(k)
                assert got is not None, (
                    f"state is missing joint '{k}' - publish ALL of "
                    "MOTOR_NAMES, not just what changed"
                )
                assert abs(got - v) < 1.0, f"{k} reads {got:.1f}, expected rest {v:.1f}"

        rep.run(
            "2.6", "state stream carries the rest pose (all six joints)", state_is_rest
        )

        def effort_is_none():
            """Verify a dry-run arm's missing gripper effort survives as None, not 0."""
            load = arm.get_gripper_load()
            assert load is None, (
                f"get_gripper_load() returned {load!r} but the dry-run arm "
                "has no effort feedback - the server must publish NaN and "
                "the client must map NaN → None (not 0!)"
            )

        rep.run(
            "2.9",
            "gripper effort NaN convention survives the round trip",
            effort_is_none,
        )

        def command_roundtrip():
            """Verify a client-sent gripper command is reflected back in the state stream."""
            arm.send_joints({"gripper": 60.0})
            got = _wait_for(
                lambda: abs(arm.get_joints().get("gripper", 0) - 60.0) < 0.5, 2.0
            )
            assert got, (
                "commanded gripper=60 but the state never followed - check "
                "_on_command (radians to degrees, then the target under the lock?), "
                "_publish_state (radians?) and send_joints (the gripper's forward "
                "controller? radians?)\n"
                + _graph_feedback(HARDWARE_COMMANDS)
            )
            print(
                f"          round-trip: gripper {REST_POSE['gripper']:.0f} → 60 [ok] (commanded via "
                f"the gripper's forward controller and {HARDWARE_COMMANDS}, observed via "
                f"{HARDWARE_STATES} and /joint_states)"
            )

        rep.run(
            "2.8",
            "joint command round-trip (client → server → state)",
            command_roundtrip,
        )

        pan_hi_label = int(JOINT_LIMITS["shoulder_pan"][1])

        def clamp_enforced():
            """Verify an out-of-range command is clamped to the joint limit end to end."""
            pan_hi = JOINT_LIMITS["shoulder_pan"][1]
            arm.send_joints({"shoulder_pan": 500.0})
            _wait_for(
                lambda: arm.get_joints().get("shoulder_pan", 0) > pan_hi - 10.0, 2.0
            )
            pan = arm.get_joints().get("shoulder_pan", 0.0)
            assert pan <= pan_hi + 1e-6, (
                f"shoulder_pan reached {pan:.1f}° from a 500° command - "
                "clamp_joints() must gate every command (client AND server)"
            )
            arm.send_joints({"shoulder_pan": 0.0})

        rep.run(
            "2.5",
            f"safety clamp survives the transport (500° → ≤{pan_hi_label}°)",
            clamp_enforced,
        )
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass
        stop_controllers(controllers)
        runner.stop()


def check_intent(rep: Reporter, cfg: dict) -> None:
    """Verify the intent node wiring: construction, routing, the stop path, the noise gate."""
    from common.fixtures import offline_parse_text
    from common.intents import Intent, ParsedCommand
    from ros.intent_node import IntentNode

    calls = []

    def counting_parser(text: str) -> ParsedCommand:
        """Keyword-route ``text`` like the mock brain, recording every call."""
        calls.append(text)
        d = offline_parse_text(text)
        params = {"object": d["object"]} if d.get("object") else {}
        return ParsedCommand(
            Intent.from_string(d["command"]), params=params, transcript=text
        )

    ok_node, node = rep.run(
        "2.10",
        "intent node constructs (String sub + pub, RELIABLE)",
        lambda: IntentNode(cfg, counting_parser),
        hint="std_msgs/String both ways; default reliable QoS (a plain "
        "depth like 10) - intents are commands, not a sensor stream",
    )
    if not ok_node:
        rep.add("2.11", "intent routing checks", "SKIP", "needs TODO 2.10")
        return
    runner = _Runner(node)

    import rclpy
    from std_msgs.msg import String

    n = cfg.get("intent_node", {}) or {}
    probe = rclpy.create_node("selftest_intent_probe")
    received = []
    probe.create_subscription(
        String,
        n.get("intent_topic", "/vla/intent"),
        lambda m: received.append(json.loads(m.data)),
        10,
    )
    pub = probe.create_publisher(
        String, n.get("transcript_topic", "/vla/transcript"), 10
    )
    probe_runner = _Runner(probe)
    time.sleep(0.3)  # let real DDS discovery settle

    try:

        def normal_routing():
            """Verify a plain transcript round-trips through the parser into published intent JSON."""
            pub.publish(String(data="pick up the ball"))
            got = _wait_for(lambda: received, 3.0)
            assert got, _graph_feedback(n.get("intent_topic", "/vla/intent"))
            intent = received[-1]
            assert intent.get("command") == "pick_place", (
                f"published intent {intent} for 'pick up the ball' - expected "
                "pick_place (is the parser being called and its result "
                "serialized?)"
            )
            assert (
                intent.get("object") == "ball"
            ), f"object slot lost: {intent} - carry cmd.object_name through"

        rep.run("2.11", "transcript → intent JSON round-trip", normal_routing)

        def stop_fast_path():
            """Verify a stop transcript publishes immediately without reaching the parser."""
            before = len(calls)
            n_intents = len(received)
            pub.publish(String(data="whoa STOP right now!"))
            got = _wait_for(lambda: len(received) > n_intents, 3.0)
            assert got, "no intent published for a stop transcript"
            assert (
                received[-1].get("command") == "stop"
            ), f"stop transcript produced {received[-1]}"
            assert len(calls) == before, (
                "the LLM parser was invoked for a STOP transcript - the "
                "safety stop must bypass the model entirely (keyword match "
                "on the raw text FIRST)"
            )

        rep.run("2.11", "stop fast path bypasses the LLM", stop_fast_path)

        def noise_gate():
            """Verify a noise transcript is dropped before it reaches the parser."""
            before = len(calls)
            n_intents = len(received)
            pub.publish(String(data="Thank you."))
            time.sleep(0.6)
            assert len(received) == n_intents, (
                f"noise transcript published an intent: {received[-1]} - "
                "Whisper's silence hallucinations must be dropped (is_noise)"
            )
            assert (
                len(calls) == before
            ), "the parser was invoked for a noise transcript - gate first"

        rep.run("2.11", "noise transcripts are dropped before the LLM", noise_gate)
    finally:
        probe_runner.stop()
        runner.stop()


# =============================================================================


def main() -> int:
    """CLI: run every component 2 check over ROS 2 and print a summary."""
    argparse.ArgumentParser(description="Component 2 self-test (requires ROS 2)").parse_args()
    try:
        import rclpy
    except ImportError as e:
        raise SystemExit(f"ROS 2 is required: source /opt/ros/jazzy/setup.bash first ({e})") from e

    ARTIFACTS.mkdir(exist_ok=True)
    rep = Reporter("Component 2 - ROS 2 nodes & topics")
    print("  transport: ROS 2 (rclpy)")
    if not rclpy.ok():
        rclpy.init()

    cfg = load_config()
    check_cameras(rep, cfg)
    check_arm(rep, cfg)
    check_intent(rep, cfg)

    code = rep.summary()
    print("\nOn-hardware next steps (each in its own terminal):")
    print("  python -m ros.camera_node --role mount")
    print("  python -m ros.node_supervisor --only arm --dry-run   (drop --dry-run for the real arm)")
    print("  python -m ros.intent_node --mock")
    print(
        "  ros2 topic hz /cameras/mount/image_raw && ros2 topic echo /joint_states"
    )
    try:
        rclpy.shutdown()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
