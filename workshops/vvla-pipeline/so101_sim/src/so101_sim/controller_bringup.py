# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Bring the SO-101's controllers up: load, configure, activate.

``launch/so101_control.launch.py`` runs this once, beside the controller
manager. It takes the place of ROS's ``spawner``, which waits for the reply to
every request and, when none comes, sends the request again after 10 s. A
reply can be lost: the controller manager then logs ``failed to send response
... client will not receive response``, the request itself having been carried
out. One lost reply made the controllers 10 s late, past the 15 s a client
waits for them (2026-11-25).

Nothing here waits for a reply. Each request is sent once, and the next step
is taken when the controller manager's own status, ``/controller_manager/
activity``, shows the step before it done: a controller that is not listed is
loaded, an ``unconfigured`` one is configured, an ``inactive`` one that is to
run is activated. A reply that does arrive and reports a failure ends the
bring-up with an error. The controllers are taken in the order given, one step
under way at a time.

    python -m so101_sim.controller_bringup --active joint_state_broadcaster arm_controller \\
        gripper_forward_controller --inactive arm_forward_controller gripper_action_controller
"""

from __future__ import annotations

import argparse
import logging
import sys

from so101_sim.control import HARDWARE_STATES

logger = logging.getLogger(__name__)

MANAGER = "/controller_manager"
ACTIVE, INACTIVE, UNCONFIGURED = "active", "inactive", "unconfigured"
# The controller manager gives a controller this long to switch on: the stock spawner's figure.
SWITCH_TIMEOUT_S = 5.0


class Bringup:
    """The steps towards ``targets`` (controller -> "active" or "inactive", in order).

    ``done`` is set with True when every controller is in its target state, and with False
    when the controller manager reports a failed step.
    """

    def __init__(self, node, targets: dict[str, str], manager: str = MANAGER):
        from controller_manager_msgs.msg import ControllerManagerActivity
        from controller_manager_msgs.srv import ConfigureController, LoadController, SwitchController
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.task import Future
        from sensor_msgs.msg import JointState

        self._node = node
        self._targets = dict(targets)
        self._SwitchController = SwitchController
        self._requests = {
            "load": (node.create_client(LoadController, f"{manager}/load_controller"), LoadController.Request),
            "configure": (node.create_client(ConfigureController, f"{manager}/configure_controller"),
                          ConfigureController.Request),
        }
        self._switch = node.create_client(SwitchController, f"{manager}/switch_controller")
        self._states: dict[str, str] | None = None  # the manager's last report; None before the first
        self._sent: set[tuple[str, str]] = set()  # (controller, step) of every request sent
        self.done = Future()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(ControllerManagerActivity, f"{manager}/activity", self._on_activity, latched)
        # A service that was not ready when the report came is ready by a later arm-server
        # state: each one is the occasion to look again. The hardware side runs on that stream.
        node.create_subscription(
            JointState, HARDWARE_STATES, lambda _msg: self._advance(),
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1),
        )

    def _on_activity(self, msg) -> None:
        self._states = {controller.name: controller.state.label for controller in msg.controllers}
        self._advance()

    def _next_step(self) -> tuple[str, str] | None:
        """The first controller not yet in its target state and the step it needs; None when all are."""
        for name, target in self._targets.items():
            state = self._states.get(name)
            if state is None:
                return name, "load"
            if state == UNCONFIGURED:
                return name, "configure"
            if state == INACTIVE and target == ACTIVE:
                return name, "activate"
            if state != target:
                return name, "wait"  # between two states: the manager's next report says which
        return None

    def _advance(self) -> None:
        """Send the next step's request, once, when the manager's report calls for it."""
        if self.done.done() or self._states is None:
            return
        step = self._next_step()
        if step is None:
            logger.info("controllers up: %s", ", ".join(f"{n} {t}" for n, t in self._targets.items()))
            self.done.set_result(True)
            return
        name, action = step
        if action == "wait" or step in self._sent:
            return
        if action == "activate":
            client, request = self._switch, self._SwitchController.Request()
            request.activate_controllers = [name]
            request.strictness = self._SwitchController.Request.STRICT
            request.timeout.sec = int(SWITCH_TIMEOUT_S)
        else:
            client, make = self._requests[action]
            request = make()
            request.name = name
        if not client.service_is_ready():
            return  # looked at again on the arm server's next state
        self._sent.add(step)
        logger.info("%s %s", action, name)
        client.call_async(request).add_done_callback(lambda reply, step=step: self._on_reply(step, reply))

    def _on_reply(self, step: tuple[str, str], reply) -> None:
        """A reply is not needed to go on. One that says the step failed ends the bring-up."""
        name, action = step
        result = reply.result()
        if result is not None and not result.ok and not self.done.done():
            logger.error("the controller manager could not %s %s%s", action, name,
                         f": {result.message}" if getattr(result, "message", "") else "")
            self.done.set_result(False)


def main(argv=None) -> int:
    import rclpy

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--active", nargs="*", default=[], help="controllers to load, configure and activate")
    parser.add_argument("--inactive", nargs="*", default=[], help="controllers to load and configure only")
    parser.add_argument("--controller-manager", default=MANAGER)
    args, ros_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="[controller_bringup] %(message)s")
    targets = {**{name: ACTIVE for name in args.active}, **{name: INACTIVE for name in args.inactive}}
    rclpy.init(args=ros_args)
    node = rclpy.create_node("so101_controller_bringup")
    try:
        bringup = Bringup(node, targets, args.controller_manager)
        rclpy.spin_until_future_complete(node, bringup.done)  # no time limit: its clients have theirs
        return 0 if bringup.done.result() else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
