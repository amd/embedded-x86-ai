# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The controllers' bring-up against a stand-in controller manager.

``so101_sim.controller_bringup`` loads, configures and activates the
controllers for ``launch/so101_control.launch.py``. It must not depend on a
reply: the real controller manager was seen to carry a request out and lose
its reply, and the stock spawner then waited 10 s (2026-11-25).

The stand-in here serves the three services the bring-up uses and publishes
``/controller_manager/activity`` after every change, as the real one does.
With ``replies=False`` it carries every request out and never answers it. The
real controller manager is covered by ``test_pipeline_ros2_control.py`` and
the stacks' ROS tests, which start the launch.
"""

from __future__ import annotations

import threading

import pytest

from twin_ros_rig import join_test_domain

# Measured 2026-11-25: a bring-up against the stand-in ends within 0.2 s.
pytestmark = [pytest.mark.sim, pytest.mark.timeout(15)]
pytest.importorskip("rclpy")

TARGETS = {
    "joint_state_broadcaster": "active",
    "arm_controller": "active",
    "gripper_forward_controller": "active",
    "arm_forward_controller": "inactive",
    "gripper_action_controller": "inactive",
}
# Every request the bring-up must send for TARGETS, each once, in this order.
REQUESTS = [
    ("load", "joint_state_broadcaster"), ("configure", "joint_state_broadcaster"), ("activate", "joint_state_broadcaster"),
    ("load", "arm_controller"), ("configure", "arm_controller"), ("activate", "arm_controller"),
    ("load", "gripper_forward_controller"), ("configure", "gripper_forward_controller"),
    ("activate", "gripper_forward_controller"),
    ("load", "arm_forward_controller"), ("configure", "arm_forward_controller"),
    ("load", "gripper_action_controller"), ("configure", "gripper_action_controller"),
]


class StandInManager:
    """A controller manager's three services and its activity topic, on a node of its own.

    ``replies=False``: every request is carried out and its reply is never sent (the callback
    does not return until the stand-in closes). ``refuse``: the request whose reply is ``ok=False``.
    """

    def __init__(self, replies: bool = True, refuse: tuple[str, str] | None = None):
        import rclpy
        from controller_manager_msgs.msg import ControllerManagerActivity, NamedLifecycleState
        from controller_manager_msgs.srv import ConfigureController, LoadController, SwitchController
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._Activity, self._State = ControllerManagerActivity, NamedLifecycleState
        self.replies, self.refuse = replies, refuse
        self.requests: list[tuple[str, str]] = []
        self.states: dict[str, str] = {}
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self.node = rclpy.create_node("controller_manager")
        group = ReentrantCallbackGroup()  # an unanswered request must not hold the next one up
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._activity = self.node.create_publisher(ControllerManagerActivity, "/controller_manager/activity", latched)
        self.node.create_service(LoadController, "/controller_manager/load_controller",
                                 lambda req, res: self._serve("load", [req.name], "unconfigured", res),
                                 callback_group=group)
        self.node.create_service(ConfigureController, "/controller_manager/configure_controller",
                                 lambda req, res: self._serve("configure", [req.name], "inactive", res),
                                 callback_group=group)
        self.node.create_service(SwitchController, "/controller_manager/switch_controller",
                                 lambda req, res: self._serve("activate", list(req.activate_controllers), "active", res),
                                 callback_group=group)
        self._executor = MultiThreadedExecutor(num_threads=len(REQUESTS) + 2)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, name="stand-in-manager", daemon=True)
        self._publish()  # the manager's first report: nothing loaded
        self._thread.start()

    def _publish(self) -> None:
        msg = self._Activity()
        for name, label in self.states.items():
            state = self._State()
            state.name, state.state.label = name, label
            msg.controllers.append(state)
        self._activity.publish(msg)

    def _serve(self, action: str, names: list[str], becomes: str, response):
        with self._lock:
            self.requests += [(action, name) for name in names]
            refused = any((action, name) == self.refuse for name in names)
            if not refused:
                self.states.update({name: becomes for name in names})
                self._publish()
        response.ok = not refused
        if not self.replies:
            self._closing.wait()  # the request was carried out; its reply never leaves
        return response

    def close(self) -> None:
        self._closing.set()
        self._executor.shutdown()
        self._thread.join()
        self.node.destroy_node()


@pytest.fixture
def bring_up():
    """``run(manager) -> True | False | None``: the bring-up's result against ``manager``; None
    when it had not ended 5 s after it began, which no case here expects."""
    import rclpy

    from vla_pipeline.robot.ros2_control_arm import Spinner

    join_test_domain()
    if not rclpy.ok():
        rclpy.init()
    opened = []

    def run(manager: StandInManager):
        from sensor_msgs.msg import JointState

        from so101_sim.control import HARDWARE_STATES
        from so101_sim.controller_bringup import Bringup

        host = Spinner("so101_controller_bringup")
        opened.extend([manager, host])
        ended = threading.Event()
        bringup = Bringup(host.node, TARGETS)
        bringup.done.add_done_callback(lambda _future: ended.set())
        # The arm server's state stream: the bring-up looks at its services again on each state.
        states = manager.node.create_publisher(JointState, HARDWARE_STATES, 1)
        manager.node.create_timer(1 / 60, lambda: states.publish(JointState()))
        return bringup.done.result() if ended.wait(5.0) else None

    yield run
    for item in reversed(opened):
        item.close()


def test_the_controllers_come_up_each_request_sent_once(bring_up):
    manager = StandInManager()
    assert bring_up(manager) is True
    assert manager.states == TARGETS
    assert manager.requests == REQUESTS


def test_the_controllers_come_up_when_no_reply_ever_arrives(bring_up):
    manager = StandInManager(replies=False)
    assert bring_up(manager) is True, "the bring-up waited for a reply"
    assert manager.states == TARGETS
    assert manager.requests == REQUESTS


def test_a_step_the_manager_refuses_ends_the_bring_up_as_failed(bring_up):
    manager = StandInManager(refuse=("configure", "arm_controller"))
    assert bring_up(manager) is False
    assert manager.requests[-1] == ("configure", "arm_controller")
