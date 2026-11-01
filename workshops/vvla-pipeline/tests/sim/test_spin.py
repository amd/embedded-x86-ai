# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Ctrl-C ends a node's spin cleanly: ``so101_sim.spin.spin_until_shutdown``.

rclpy's signal handler shuts the context down while a timer may be due; the
timer's publish then fails on the closed context. The twin and the camera
nodes spin with ``spin_until_shutdown``, which ends quietly on that error and
raises any error while the context is up.
"""

from __future__ import annotations

import pytest

from twin_ros_rig import TEST_DOMAIN_ID

pytestmark = pytest.mark.sim
pytest.importorskip("rclpy")


@pytest.fixture
def timer_node():
    """``make(callback)``: a node on a private context whose timer runs ``callback(context, publisher)``."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Bool

    made = []

    def make(callback):
        context = Context()
        rclpy.init(context=context, domain_id=TEST_DOMAIN_ID)
        node = rclpy.create_node("spin_probe", context=context)
        publisher = node.create_publisher(Bool, "/spin_probe", 1)
        node.create_timer(0.001, lambda: callback(context, publisher))
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        made.append((context, node, executor))
        return executor

    yield make
    for context, node, executor in made:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown(context=context)


def _shutdown_then_publish(context, publisher) -> None:
    """What a camera timer does when Ctrl-C lands mid-tick."""
    from std_msgs.msg import Bool

    context.shutdown()
    publisher.publish(Bool(data=True))


def test_a_publish_on_the_closed_context_fails_a_plain_spin(timer_node):
    executor = timer_node(_shutdown_then_publish)
    with pytest.raises(Exception, match="context is invalid"):
        executor.spin()


def test_a_publish_on_the_closed_context_ends_the_spin_quietly(timer_node):
    from so101_sim.spin import spin_until_shutdown

    spin_until_shutdown(timer_node(_shutdown_then_publish))


def test_an_error_while_the_context_is_up_is_raised(timer_node):
    from so101_sim.spin import spin_until_shutdown

    def broken(_context, _publisher):
        raise ValueError("a real callback error")

    with pytest.raises(ValueError, match="a real callback error"):
        spin_until_shutdown(timer_node(broken))
