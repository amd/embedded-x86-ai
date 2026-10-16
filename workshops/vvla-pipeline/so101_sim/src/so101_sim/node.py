# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 process hosting the digital twin in place of the arm + arm camera.

It runs the host stack's OWN server nodes - ``So101ServerNode`` with the
twin injected as its bus transport, and ``CameraNode`` with a
:class:`~so101_sim.camera.SimCamera` as its frame source - so the topics,
QoS, rates, de-duplication and health flags are byte-for-byte the real
ones. Clients (the orchestrator, the workshop app, ``ros2 topic echo``)
cannot tell the difference.

    python -m so101_sim.node --config config/pipeline.yaml
    python -m so101_sim.node --config workshop/solution/config/workshop.yaml
    python -m so101_sim.node --config config/pipeline.yaml --device-cameras --viewer

Arm and arm camera must live in ONE process (they share the physics), which
is why this replaces ``robot_node --server`` + ``camera_node --role arm``.
Extra topic: ``/sim/view/image_raw`` (the free 3-D view, bgr8).

``--notebook-twin`` (workshop stack) is the twin ``02_digital_twin.ipynb``
starts for its console, on a private ROS domain. It adds, on top of the
default launch:

* ``--scene FILE`` - props and pedestals from ``SimWorld.snapshot_scene``;
* the grasp envelope (``GRASP_LIMITS``) as this process's clamp, so a 90 deg
  standing-grasp wrist reaches the twin;
* the arm camera stamped with the time its scene was captured, not the time
  it was published;
* ``/sim/scene`` (``std_msgs/String`` JSON: each prop's xyz, quat, grasped);
* ``/sim/reset_scene`` (``std_srvs/Trigger``): ``restore_twin`` in this process.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

VIEW_TOPIC = "/sim/view/image_raw"
SCENE_TOPIC = "/sim/scene"
RESET_SERVICE = "/sim/reset_scene"


def _sensor_qos():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )


def _stamp_msg(stamp: float):
    from builtin_interfaces.msg import Time

    sec = int(stamp)
    return Time(sec=sec, nanosec=int(round((stamp - sec) * 1e9)) % 1_000_000_000)


def detect_stack(config_path: str | None, cfg: dict) -> str:
    """``workshop`` for workshop.yaml-style configs, else ``pipeline``."""
    if config_path and Path(config_path).name.startswith("workshop"):
        return "workshop"
    return "pipeline"


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stack_parts(stack: str, cfg: dict, config_path: str | None):
    """(world, arm, ServerNodeClass, CameraNodeClass) for the host stack."""
    if stack == "workshop":
        # workshop/<solution|project>/config/workshop.yaml -> host that variant's ros/
        if config_path:
            pkg = str(Path(config_path).resolve().parents[1])
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
        from common.motion import SimArmClient, sim_world
        from ros.arm_node import So101ServerNode
        from ros.camera_node import CameraNode

        world = sim_world(cfg)
        return world, SimArmClient(cfg, world), So101ServerNode, CameraNode
    from vla_pipeline.robot.robot_node import So101ServerNode
    from vla_pipeline.robot.sim_arm import SimArmClient, sim_world
    from vla_pipeline.vision.camera_node import CameraNode

    world = sim_world(cfg)
    return world, SimArmClient(cfg, world), So101ServerNode, CameraNode


class ViewPublisher:
    """Publishes a twin camera (default: the free 3-D view) as sensor_msgs/Image."""

    def __init__(self, world, role: str = "view", topic: str = VIEW_TOPIC, fps: float = 15.0):
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        from so101_sim.camera import SimCamera

        self._Image = Image
        self.node = Node("so101_sim_view")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self._pub = self.node.create_publisher(Image, topic, qos)
        self._cam = SimCamera(world, role, fps=fps, blocking=False)
        self.node.create_timer(1.0 / fps, self._tick)

    def _tick(self) -> None:
        ok, frame = self._cam.read()
        if not ok:
            return
        msg = self._Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "sim_view"
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self._pub.publish(msg)

    def destroy(self) -> None:
        self._cam.release()
        self.node.destroy_node()


class StampedCameraPublisher:
    """A twin camera on its usual topic, stamped with the scene capture time.

    ``CameraNode`` stamps a frame when it publishes it, so a frame rendered
    just before the arm settled can look newer than the settle. This stamp
    is when the renderer copied the scene, so ``header.stamp > settled_at``
    means the pixels show the settled arm.
    """

    def __init__(self, world, role: str, spec: dict):
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        from so101_sim.camera import SimCamera

        self._Image = Image
        self.role = role
        self.topic = str(spec.get("topic", f"/cameras/{role}/image_raw"))
        fps = float(spec.get("fps", 30.0))
        self.node = Node(f"so101_sim_{role}_camera")
        self._pub = self.node.create_publisher(Image, self.topic, _sensor_qos())
        self._cam = SimCamera(world, role, fps=fps, blocking=False)
        self._last_stamp = None
        self.node.create_timer(1.0 / fps, self._tick)

    def _tick(self) -> None:
        ok, frame = self._cam.read()
        stamp = self._cam.last_stamp
        if not ok or stamp is None or stamp == self._last_stamp:
            return
        self._last_stamp = stamp
        msg = self._Image()
        msg.header.stamp = _stamp_msg(stamp)
        msg.header.frame_id = f"camera_{self.role}"
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self._pub.publish(msg)

    def destroy(self) -> None:
        self._cam.release()
        self.node.destroy_node()


class ScenePublisher:
    """``/sim/scene``: where each prop is and whether both fingers hold it."""

    def __init__(self, world, topic: str = SCENE_TOPIC, hz: float = 20.0):
        from rclpy.node import Node
        from std_msgs.msg import String

        self._String = String
        self._world = world
        self.node = Node("so101_sim_scene")
        self._pub = self.node.create_publisher(String, topic, _sensor_qos())
        self.node.create_timer(1.0 / hz, self._tick)

    def snapshot(self) -> dict:
        world = self._world
        with world.lock:
            stamp = time.time()
            bodies = {
                name: {
                    "xyz": [float(v) for v in world.data.body(name).xpos],
                    "quat": [float(v) for v in world.data.body(name).xquat],
                    "grasped": bool(world.is_grasped(name)),
                }
                for name in world.objects
            }
        return {"stamp": stamp, "bodies": bodies}

    def _tick(self) -> None:
        self._pub.publish(self._String(data=json.dumps(self.snapshot())))

    def destroy(self) -> None:
        self.node.destroy_node()


class TwinControl:
    """``/sim/reset_scene``: the console's reset button, run inside the twin.

    ``restore_twin`` parks the arm and puts the props back, under the
    server's target lock, and the server's target becomes the parked pose so
    its control loop does not drive the arm back to the pre-reset goal.
    """

    def __init__(self, server, arm):
        from rclpy.node import Node
        from std_srvs.srv import Trigger

        self._server = server
        self._arm = arm
        self.node = Node("so101_sim_control")
        self.node.create_service(Trigger, RESET_SERVICE, self._on_reset)

    def _on_reset(self, _request, response):
        from common.motion import restore_twin

        server = self._server
        with server._lock:
            parked = restore_twin(self._arm)
            server._target = dict(parked)
            server._last_cmd_time = time.time()
            server._last_sent_target = None
        response.success = True
        response.message = "arm parked, objects back on the table"
        return response

    def destroy(self) -> None:
        self.node.destroy_node()


def _die_with_parent() -> None:
    """Get SIGTERM when the process that started this twin exits (Linux).

    Set here rather than through ``Popen(preexec_fn=...)``: the notebook
    kernel that starts the twin runs threads, and ``preexec_fn`` is unsafe
    in a threaded parent. rclpy turns the SIGTERM into a clean shutdown.
    """
    import ctypes
    import os
    import signal

    parent = os.getppid()
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    if os.getppid() != parent:  # the parent exited before prctl took effect
        os._exit(0)


def _use_grasp_envelope() -> None:
    """Clamp this process to ``GRASP_LIMITS``.

    The workshop server switches the process to ``LIVE_MIMIC_LIMITS`` because
    the twin arm is live, and that caps ``wrist_flex`` at 84.4 deg. The
    console's client already clamps ordinary moves to the desk cap; only
    grasp moves are sent wider.
    """
    from common.motion import GRASP_LIMITS, set_active_limits

    set_active_limits(GRASP_LIMITS)


def build_nodes(
    cfg: dict,
    *,
    stack: str = "pipeline",
    config_path: str | None = None,
    device_cameras: bool = False,
    view: bool = True,
    notebook_twin: bool = False,
):
    """Create the twin's ROS nodes (rclpy must be initialised).

    Returns ``(world, nodes, closers)``: add ``nodes`` to an executor, call
    each closer on shutdown. Cameras with ``source: sim`` are rendered on the
    twin; ``source: device`` cameras are opened here only if
    ``device_cameras`` (otherwise their usual camera node serves them).
    ``notebook_twin`` adds what ``02_digital_twin.ipynb``'s console needs
    (see the module docstring); it requires the workshop stack.
    """
    from so101_sim.camera import SimCamera

    if notebook_twin and stack != "workshop":
        raise ValueError("--notebook-twin needs the workshop stack")
    world, arm, ServerNode, CameraNode = _stack_parts(stack, cfg, config_path)
    nodes, closers = [], []
    server = ServerNode(cfg, arm=arm)
    nodes.append(server)
    closers.append(server.destroy_node)
    if notebook_twin:
        _use_grasp_envelope()
    for role, spec in (cfg.get("cameras") or {}).items():
        if role not in ("arm", "operator") or not isinstance(spec, dict):
            continue
        source = spec.get("source", "sim" if role == "arm" else "device")
        if source == "sim" and notebook_twin and role == "arm":
            stamped = StampedCameraPublisher(world, role, spec)
            nodes.append(stamped.node)
            closers.append(stamped.destroy)
            continue
        if source == "sim":
            cam = CameraNode(role, spec, frame_source=SimCamera(world, role, blocking=False))
        elif device_cameras:
            cam = CameraNode(role, spec)
        else:
            continue
        nodes.append(cam)
        closers.append(cam.destroy_node)
    if view:
        vp = ViewPublisher(world)
        nodes.append(vp.node)
        closers.append(vp.destroy)
    if notebook_twin:
        scene = ScenePublisher(world)
        control = TwinControl(server, arm)
        nodes.extend((scene.node, control.node))
        closers.extend((scene.destroy, control.destroy))
    return world, nodes, closers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="SO-101 digital twin ROS 2 node")
    ap.add_argument("--config", required=True, help="pipeline.yaml or workshop.yaml")
    ap.add_argument("--stack", choices=["auto", "pipeline", "workshop"], default="auto")
    ap.add_argument("--device-cameras", action="store_true",
                    help="also publish source:device cameras from this process")
    ap.add_argument("--sim-cameras", default=None,
                    help="comma list of camera roles to render on the twin "
                    "(overrides cameras.<role>.source; default: as configured)")
    ap.add_argument("--no-view", action="store_true", help=f"don't publish {VIEW_TOPIC}")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer window")
    ap.add_argument("--notebook-twin", action="store_true",
                    help="the 02_digital_twin.ipynb console twin (workshop stack): grasp "
                    f"envelope, capture-stamped arm camera, {SCENE_TOPIC}, {RESET_SERVICE}")
    ap.add_argument("--scene", default=None,
                    help="with --notebook-twin: JSON from SimWorld.snapshot_scene to start from")
    args, ros_args = ap.parse_known_args()
    if args.scene and not args.notebook_twin:
        ap.error("--scene needs --notebook-twin")
    if args.notebook_twin:
        _die_with_parent()

    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    except ImportError:
        raise SystemExit("rclpy not importable - source /opt/ros/jazzy/setup.bash first") from None

    cfg = _load_yaml(args.config)
    cfg.setdefault("robot", {})["backend"] = "sim"
    if args.viewer:
        cfg.setdefault("sim", {})["viewer"] = True
    for role in filter(None, (args.sim_cameras or "").split(",")):
        cfg.setdefault("cameras", {}).setdefault(role.strip(), {})["source"] = "sim"
    stack = detect_stack(args.config, cfg) if args.stack == "auto" else args.stack
    if args.notebook_twin and stack != "workshop":
        ap.error("--notebook-twin needs the workshop stack")

    rclpy.init(args=ros_args)
    world, nodes, closers = build_nodes(
        cfg,
        stack=stack,
        config_path=args.config,
        device_cameras=args.device_cameras,
        view=not args.no_view,
        notebook_twin=args.notebook_twin,
    )
    if args.scene:
        with open(args.scene, encoding="utf-8") as f:
            world.apply_scene(json.load(f))
        logger.info("scene loaded from %s", args.scene)
    executor = MultiThreadedExecutor(num_threads=4)
    for n in nodes:
        executor.add_node(n)
    logger.info("digital twin up (%s stack): %d nodes", stack, len(nodes))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for close in reversed(closers):
            try:
                close()
            except Exception as e:
                logger.debug("shutdown: %s", e)
        from so101_sim.world import close_shared_world

        close_shared_world()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
