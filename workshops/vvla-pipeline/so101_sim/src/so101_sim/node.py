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
Extra topic: ``/sim/view/image_raw`` (the free 3-D view, bgr8). Extra service:
``/sim/reset_scene`` (``std_srvs/Trigger``): ``restore_twin`` in this process,
with the parked pose made the server's target.

``--notebook-twin`` (workshop stack) is the twin ``02_digital_twin.ipynb``
starts for its console, on a private ROS domain. It adds, on top of the
default launch:

* ``--scene FILE`` - props and pedestals from ``SimWorld.snapshot_scene``;
* the grasp envelope (``GRASP_LIMITS``) as this process's clamp, so a 90 deg
  standing-grasp wrist reaches the twin;
* the arm camera stamped with the time its scene was captured, not the time
  it was published;
* the props (:class:`PropPublisher`): each prop's pose as a TF frame
  (``world`` to ``<prop>``), ``/sim/props_settled`` (``std_msgs/Bool``: every
  prop at rest), and on ``/diagnostics`` which props both fingers hold and the
  gripper's torque limit and overload state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

VIEW_TOPIC = "/sim/view/image_raw"
RESET_SERVICE = "/sim/reset_scene"
WORLD_FRAME = "world"  # the twin's world: the props' TF parent
PROPS_SETTLED_TOPIC = "/sim/props_settled"
DIAGNOSTICS_TOPIC = "/diagnostics"
PROPS_STATUS = "so101_sim: props"  # settled, and <prop>.grasped for each prop
GRIPPER_STATUS = "so101_sim: gripper"  # torque_limit, overload_torque, tripped


def _sensor_qos():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )


def _stamp_msg(stamp: float):
    from builtin_interfaces.msg import Time

    nanoseconds = int(round(stamp * 1_000_000_000))
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


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
    """(world, arm, ServerNodeClass, CameraNodeClass) for the host stack: its
    own arm server and camera node, on its own bus transport over the twin."""
    from so101_sim.control import server_transport_cfg

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
        return world, SimArmClient(server_transport_cfg(cfg), world), So101ServerNode, CameraNode
    from vla_pipeline.robot.robot_node import So101ServerNode
    from vla_pipeline.robot.sim_arm import SimArmClient, sim_world
    from vla_pipeline.vision.camera_node import CameraNode

    world = sim_world(cfg)
    return world, SimArmClient(server_transport_cfg(cfg), world), So101ServerNode, CameraNode


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


class PropPublisher:
    """The props for a console, one stamp per tick: each prop's pose as a TF
    frame (``world`` to ``<prop>``); ``/sim/props_settled``, every prop at rest
    (``SimWorld.props_settled``); and on ``/diagnostics``, the same flag, which
    props both fingers hold, and the gripper's ``Torque_Limit``,
    ``Overload_Torque`` and overload trip (level ERROR once it tripped). The
    stamp is the twin's wall clock, as its camera frames' are."""

    def __init__(self, world, hz: float = 20.0):
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from geometry_msgs.msg import TransformStamped
        from rclpy.node import Node
        from std_msgs.msg import Bool
        from tf2_ros import TransformBroadcaster

        self._msgs = SimpleNamespace(
            DiagnosticArray=DiagnosticArray, DiagnosticStatus=DiagnosticStatus, KeyValue=KeyValue,
            TransformStamped=TransformStamped, Bool=Bool,
        )
        self._world = world
        self.node = Node("so101_sim_props")
        self._tf = TransformBroadcaster(self.node)
        self._settled_pub = self.node.create_publisher(Bool, PROPS_SETTLED_TOPIC, 10)
        self._diagnostics_pub = self.node.create_publisher(DiagnosticArray, DIAGNOSTICS_TOPIC, 10)
        self.node.create_timer(1.0 / hz, self._tick)

    def _tick(self) -> None:
        from so101_sim import MOTOR_NAMES

        m, world = self._msgs, self._world
        with world.lock:
            stamp = _stamp_msg(time.time())
            poses = {
                name: ([float(v) for v in world.data.body(name).xpos], [float(v) for v in world.data.body(name).xquat])
                for name in world.objects
            }
            grasped = {name: bool(world.is_grasped(name)) for name in world.objects}
            settled = bool(world.props_settled())
            torque_limit = int(world.bus.read("Torque_Limit", "gripper", normalize=False))
            overload_torque = int(world.bus.read("Overload_Torque", "gripper", normalize=False))
            tripped = bool(world.protection.tripped[MOTOR_NAMES.index("gripper")])
        frames = []
        for name, (xyz, (qw, qx, qy, qz)) in poses.items():
            tf = m.TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, WORLD_FRAME, name
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = xyz
            tf.transform.rotation.w, tf.transform.rotation.x = qw, qx
            tf.transform.rotation.y, tf.transform.rotation.z = qy, qz
            frames.append(tf)
        self._tf.sendTransform(frames)
        self._settled_pub.publish(m.Bool(data=settled))

        def flag(value: bool) -> str:
            return "true" if value else "false"

        diagnostics = m.DiagnosticArray()
        diagnostics.header.stamp = stamp
        diagnostics.status = [
            m.DiagnosticStatus(
                level=m.DiagnosticStatus.OK, name=PROPS_STATUS, hardware_id="so101_sim",
                message="props at rest" if settled else "props moving",
                values=[m.KeyValue(key="settled", value=flag(settled))]
                + [m.KeyValue(key=f"{name}.grasped", value=flag(held)) for name, held in grasped.items()],
            ),
            m.DiagnosticStatus(
                level=m.DiagnosticStatus.ERROR if tripped else m.DiagnosticStatus.OK,
                name=GRIPPER_STATUS, hardware_id="so101_sim",
                message="overload protection tripped" if tripped else "ok",
                values=[
                    m.KeyValue(key="torque_limit", value=str(torque_limit)),
                    m.KeyValue(key="overload_torque", value=str(overload_torque)),
                    m.KeyValue(key="tripped", value=flag(tripped)),
                ],
            ),
        ]
        self._diagnostics_pub.publish(diagnostics)

    def destroy(self) -> None:
        self.node.destroy_node()


class TwinControl:
    """``/sim/reset_scene``: the consoles' reset button, run inside the twin.

    ``restore_twin`` parks the arm and puts the props back, through the
    server's ``reset_with``: under its target lock, and with the parked pose
    as its target, so its control loop does not drive the arm back to the
    pre-reset goal. A client parks the arm through the controllers first
    (``ControllerArm.reset_twin``), so they hold the parked pose too.
    """

    def __init__(self, server, arm):
        from rclpy.node import Node
        from std_srvs.srv import Trigger

        self._server = server
        self._arm = arm
        self.node = Node("so101_sim_control")
        self.node.create_service(Trigger, RESET_SERVICE, self._on_reset)

    def _on_reset(self, _request, response):
        from so101_sim.world import restore_twin

        self._server.reset_with(lambda: restore_twin(self._arm))
        response.success = True
        response.message = "arm parked, objects back on the table"
        return response

    def destroy(self) -> None:
        self.node.destroy_node()


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
        if role not in ("arm", "mount") or not isinstance(spec, dict):
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
    control = TwinControl(server, arm)
    nodes.append(control.node)
    closers.append(control.destroy)
    if notebook_twin:
        props = PropPublisher(world)
        nodes.append(props.node)
        closers.append(props.destroy)
    return world, nodes, closers


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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
                    "envelope, capture-stamped arm camera, the props' TF frames and status")
    ap.add_argument("--scene", default=None,
                    help="with --notebook-twin: JSON from SimWorld.snapshot_scene to start from")
    args, ros_args = ap.parse_known_args()
    if args.scene and not args.notebook_twin:
        ap.error("--scene needs --notebook-twin")
    if args.notebook_twin:
        # rclpy turns the SIGTERM into a clean shutdown.
        from so101_sim.parent_death import die_with_parent

        die_with_parent()

    from so101_sim.domain_lock import DomainTaken, acquire

    try:
        domain_lock = acquire(int(os.environ.get("ROS_DOMAIN_ID", "0")))  # held until this process exits
    except DomainTaken as e:
        raise SystemExit(str(e)) from None

    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
    except ImportError:
        raise SystemExit("rclpy not importable - source /opt/ros/jazzy/setup.bash first") from None

    from so101_sim.spin import spin_until_shutdown

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
    # One thread: every callback here is short (the bus and the renderer run on
    # their own threads), and rclpy's MultiThreadedExecutor rebuilt its wait set
    # in a tight loop under a 60 Hz command stream, starving the physics.
    executor = SingleThreadedExecutor()
    for n in nodes:
        executor.add_node(n)
    logger.info("digital twin up (%s stack): %d nodes", stack, len(nodes))
    try:
        spin_until_shutdown(executor)
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
        domain_lock.close()


if __name__ == "__main__":
    main()
