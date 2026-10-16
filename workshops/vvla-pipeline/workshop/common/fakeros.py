# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""An in-memory ROS 2 shim for offline self-tests (PROVIDED).

The ROS self-test harness wants to exercise *your* node wiring even on a
machine without ROS 2 installed. This module implements the tiny slice of
``rclpy`` / ``sensor_msgs`` / ``std_msgs`` the workshop uses - nodes,
publishers, subscriptions, timers, executors, QoS - with an in-process
message broker, and installs itself into ``sys.modules`` so your unmodified
``import rclpy`` code runs against it.

Fidelity notes (deliberate):
- Delivery is synchronous on the publisher's thread.
- QoS *reliability compatibility* is enforced like real DDS: a BEST_EFFORT
  publisher does **not** deliver to a RELIABLE subscriber. If your camera
  frames vanish in the offline test, check your QoS on both ends - the same
  mistake bites on the real rig.
- ``create_timer`` callbacks only run while an executor is spinning.

This is a test double, not ROS. The final word is always the on-hardware run.
"""

from __future__ import annotations

import sys
import threading
import time
import types

# ----------------------------------------------------------------------------
# Message types
# ----------------------------------------------------------------------------


class _TimeMsg:
    """In-memory stand-in for ``builtin_interfaces/Time``."""

    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class _Header:
    """In-memory stand-in for ``std_msgs/Header``."""

    def __init__(self):
        self.stamp = _TimeMsg()
        self.frame_id = ""


class Image:
    """In-memory stand-in for ``sensor_msgs/Image``."""

    def __init__(self):
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


class JointState:
    """In-memory stand-in for ``sensor_msgs/JointState``."""

    def __init__(self):
        self.header = _Header()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class Bool:
    """In-memory stand-in for ``std_msgs/Bool``."""

    def __init__(self, data=False):
        self.data = bool(data)


class String:
    """In-memory stand-in for ``std_msgs/String``."""

    def __init__(self, data=""):
        self.data = str(data)


class Int32:
    """In-memory stand-in for ``std_msgs/Int32``."""

    def __init__(self, data=0):
        self.data = int(data)


# ----------------------------------------------------------------------------
# QoS
# ----------------------------------------------------------------------------


class ReliabilityPolicy:
    """In-memory stand-in for ``rclpy.qos.ReliabilityPolicy``."""

    RELIABLE = "RELIABLE"
    BEST_EFFORT = "BEST_EFFORT"


class HistoryPolicy:
    """In-memory stand-in for ``rclpy.qos.HistoryPolicy``."""

    KEEP_LAST = "KEEP_LAST"
    KEEP_ALL = "KEEP_ALL"


class DurabilityPolicy:
    """In-memory stand-in for ``rclpy.qos.DurabilityPolicy``."""

    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"


class QoSProfile:
    """In-memory stand-in for ``rclpy.qos.QoSProfile``."""

    def __init__(
        self,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=None,
        **_,
    ):
        self.depth = depth
        self.reliability = reliability
        self.history = history
        self.durability = durability


def _as_qos(qos) -> QoSProfile:
    """rclpy accepts a plain int depth (RELIABLE default) or a QoSProfile."""
    return qos if isinstance(qos, QoSProfile) else QoSProfile(depth=int(qos))


# ----------------------------------------------------------------------------
# Broker
# ----------------------------------------------------------------------------


class _Broker:
    """Process-wide in-memory message bus: topic name -> publishers/subscribers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subs: dict = {}  # topic -> list[_Subscription]
        self.pubs: dict = {}  # topic -> list[_Publisher]
        self.qos_drops: list = []  # (topic, pub_node, sub_node) incompatible
        self.errors: list = []  # (topic, node, exception) from callbacks

    def reset(self) -> None:
        """Clear every publisher, subscriber, and recorded drop/error."""
        with self.lock:
            self.subs.clear()
            self.pubs.clear()
            self.qos_drops.clear()
            self.errors.clear()

    def graph(self) -> dict:
        """Snapshot the current topic graph: node names and QoS reliability."""
        with self.lock:
            return {
                "publishers": {
                    t: [(p.node._name, p.qos.reliability) for p in ps]
                    for t, ps in self.pubs.items()
                },
                "subscribers": {
                    t: [(s.node._name, s.qos.reliability) for s in ss]
                    for t, ss in self.subs.items()
                },
            }


BROKER = _Broker()


class _Publisher:
    """In-memory stand-in for an ``rclpy`` publisher handle."""

    def __init__(self, node, msg_type, topic, qos):
        self.node, self.msg_type, self.topic, self.qos = node, msg_type, topic, qos

    def publish(self, msg) -> None:
        """Deliver ``msg`` to every subscriber on this topic with compatible QoS."""
        with BROKER.lock:
            subs = list(BROKER.subs.get(self.topic, []))
        for sub in subs:
            # Real DDS: a BEST_EFFORT publisher can't satisfy a RELIABLE
            # subscriber - the endpoints simply never match. Mimic that.
            if (
                self.qos.reliability == ReliabilityPolicy.BEST_EFFORT
                and sub.qos.reliability == ReliabilityPolicy.RELIABLE
            ):
                BROKER.qos_drops.append((self.topic, self.node._name, sub.node._name))
                continue
            try:
                sub.callback(msg)
            except Exception as e:  # surface, don't hide, callback crashes
                BROKER.errors.append((self.topic, sub.node._name, e))


class _Subscription:
    """In-memory stand-in for an ``rclpy`` subscription handle."""

    def __init__(self, node, msg_type, topic, callback, qos):
        self.node, self.msg_type, self.topic = node, msg_type, topic
        self.callback, self.qos = callback, qos


class _Timer:
    """In-memory stand-in for an ``rclpy`` wall timer."""

    def __init__(self, period_s, callback):
        self.period_s = float(period_s)
        self.callback = callback
        self.next_due = time.monotonic() + self.period_s
        self.cancelled = False

    def cancel(self):
        """Stop the timer from firing again."""
        self.cancelled = True


class _Clock:
    """In-memory stand-in for ``rclpy.clock.Clock``."""

    def now(self):
        """Return self; :meth:`to_msg` does the actual time capture."""
        return self

    def to_msg(self):
        """Capture the current wall-clock time as a :class:`_TimeMsg`."""
        t = time.time()
        return _TimeMsg(sec=int(t), nanosec=int((t % 1.0) * 1e9))


class _Logger:
    """In-memory stand-in for the logger returned by ``Node.get_logger``."""

    def __init__(self, name):
        self._name = name

    def _log(self, level, msg):
        """Print one ``[LEVEL] [node name] message`` line."""
        print(f"[{level}] [{self._name}] {msg}")

    def debug(self, msg):
        """Swallow debug messages (kept quiet by default, like rclpy's)."""
        pass

    def info(self, msg):
        """Log ``msg`` at INFO level."""
        self._log("INFO", msg)

    def warn(self, msg):
        """Log ``msg`` at WARN level."""
        self._log("WARN", msg)

    warning = warn

    def error(self, msg):
        """Log ``msg`` at ERROR level."""
        self._log("ERROR", msg)


# ----------------------------------------------------------------------------
# Node + executors + module-level rclpy API
# ----------------------------------------------------------------------------

_CTX = {"ok": False}


class Node:
    """In-memory stand-in for ``rclpy.node.Node``."""

    def __init__(self, name: str):
        self._name = name
        self._publishers: list = []
        self._subscriptions: list = []
        self._timers: list = []
        self._clock = _Clock()
        self._logger = _Logger(name)

    def create_publisher(self, msg_type, topic, qos_profile=10):
        """Register a publisher on ``topic`` and return its handle."""
        pub = _Publisher(self, msg_type, topic, _as_qos(qos_profile))
        self._publishers.append(pub)
        with BROKER.lock:
            BROKER.pubs.setdefault(topic, []).append(pub)
        return pub

    def create_subscription(self, msg_type, topic, callback, qos_profile=10):
        """Register a subscription callback on ``topic`` and return its handle."""
        sub = _Subscription(self, msg_type, topic, callback, _as_qos(qos_profile))
        self._subscriptions.append(sub)
        with BROKER.lock:
            BROKER.subs.setdefault(topic, []).append(sub)
        return sub

    def create_timer(self, timer_period_sec, callback):
        """Register a periodic callback and return its timer handle."""
        t = _Timer(timer_period_sec, callback)
        self._timers.append(t)
        return t

    def get_clock(self):
        """Return this node's :class:`_Clock`."""
        return self._clock

    def get_logger(self):
        """Return this node's :class:`_Logger`."""
        return self._logger

    def get_name(self):
        """Return the node's name."""
        return self._name

    def destroy_node(self):
        """Unregister every publisher and subscription, and cancel every timer."""
        with BROKER.lock:
            for pub in self._publishers:
                lst = BROKER.pubs.get(pub.topic, [])
                if pub in lst:
                    lst.remove(pub)
            for sub in self._subscriptions:
                lst = BROKER.subs.get(sub.topic, [])
                if sub in lst:
                    lst.remove(sub)
        for t in self._timers:
            t.cancel()


class _ExecutorBase:
    """Shared spin loop for :class:`SingleThreadedExecutor` and :class:`MultiThreadedExecutor`."""

    def __init__(self, num_threads=None):
        self._nodes: list = []
        self._shutdown = threading.Event()

    def add_node(self, node):
        """Add ``node`` to the set of nodes this executor spins."""
        self._nodes.append(node)

    def _run_due_timers(self):
        """Fire every timer, across all added nodes, whose period has elapsed."""
        now = time.monotonic()
        for node in list(self._nodes):
            for t in list(node._timers):
                if not t.cancelled and now >= t.next_due:
                    t.next_due = now + t.period_s
                    try:
                        t.callback()
                    except Exception as e:
                        BROKER.errors.append(("<timer>", node._name, e))

    def spin(self):
        """Block, running due timers, until shutdown or the context stops."""
        while _CTX["ok"] and not self._shutdown.is_set():
            self._run_due_timers()
            time.sleep(0.002)

    def spin_once(self, timeout_sec=0.0):
        """Run due timers once, then optionally sleep up to ``timeout_sec``."""
        self._run_due_timers()
        if timeout_sec:
            time.sleep(min(timeout_sec, 0.002))

    def shutdown(self):
        """Stop this executor's spin loop."""
        self._shutdown.set()


class SingleThreadedExecutor(_ExecutorBase):
    """In-memory stand-in for ``rclpy.executors.SingleThreadedExecutor``."""


class MultiThreadedExecutor(_ExecutorBase):
    """In-memory stand-in for ``rclpy.executors.MultiThreadedExecutor`` (spins single-threaded here)."""


class ExternalShutdownException(Exception):
    """In-memory stand-in for ``rclpy.executors.ExternalShutdownException``."""


def init(args=None):
    """In-memory stand-in for ``rclpy.init``: mark the fake context as ok."""
    _CTX["ok"] = True


def ok() -> bool:
    """In-memory stand-in for ``rclpy.ok``: True while the fake context is initialized."""
    return _CTX["ok"]


def shutdown():
    """In-memory stand-in for ``rclpy.shutdown``: mark the fake context as not ok."""
    _CTX["ok"] = False


def create_node(name: str) -> Node:
    """In-memory stand-in for ``rclpy.create_node``: build a new :class:`Node`."""
    return Node(name)


def spin(node: Node):
    """In-memory stand-in for ``rclpy.spin``: spin ``node`` on a fresh single-threaded executor."""
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------
# sys.modules installation
# ----------------------------------------------------------------------------


def real_ros2_available() -> bool:
    """True when the *real* rclpy is importable (not this shim)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("rclpy")
        return spec is not None and "fakeros" not in str(spec.origin or "")
    except Exception:
        return False


def install(force: bool = False) -> bool:
    """Register the shim as ``rclpy``/``sensor_msgs``/``std_msgs``.

    No-op (returns False) when real ROS 2 is importable, unless ``force``.
    Call this BEFORE importing any module that does ``import rclpy``.
    """
    if not force and real_ros2_available():
        return False

    this = sys.modules[__name__]

    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init, rclpy_mod.ok = init, ok
    rclpy_mod.shutdown, rclpy_mod.create_node = shutdown, create_node
    rclpy_mod.spin = spin
    rclpy_mod.__fakeros__ = True

    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node

    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.QoSProfile = QoSProfile
    qos_mod.ReliabilityPolicy = ReliabilityPolicy
    qos_mod.HistoryPolicy = HistoryPolicy
    qos_mod.DurabilityPolicy = DurabilityPolicy

    exec_mod = types.ModuleType("rclpy.executors")
    exec_mod.SingleThreadedExecutor = SingleThreadedExecutor
    exec_mod.MultiThreadedExecutor = MultiThreadedExecutor
    exec_mod.ExternalShutdownException = ExternalShutdownException

    rclpy_mod.node, rclpy_mod.qos, rclpy_mod.executors = node_mod, qos_mod, exec_mod

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image, sensor_msgs_msg.JointState = Image, JointState
    sensor_msgs.msg = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool, std_msgs_msg.String, std_msgs_msg.Int32 = Bool, String, Int32
    std_msgs.msg = std_msgs_msg

    sys.modules.update(
        {
            "rclpy": rclpy_mod,
            "rclpy.node": node_mod,
            "rclpy.qos": qos_mod,
            "rclpy.executors": exec_mod,
            "sensor_msgs": sensor_msgs,
            "sensor_msgs.msg": sensor_msgs_msg,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
        }
    )
    this.INSTALLED = True
    return True


INSTALLED = False
