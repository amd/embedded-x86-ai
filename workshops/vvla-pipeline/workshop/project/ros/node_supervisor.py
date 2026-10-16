# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Node supervisor - bring the ROS 2 server nodes up alongside the app.

The architecture rule is *"server nodes own hardware; everything else
subscribes."* On the rig that means one process per device: the SO-101 arm
bus, each camera, and (optionally) the Llama intent brain. Rather than make you
open a terminal per node, the live app (and the workshop launcher) start them
here as child processes, wait until each is actually publishing, and tear them
all down on exit - SIGINT first so the arm parks at ``REST_POSE``.

Two properties make this safe to run automatically:

* **Fail-fast (no silent stand-ins).** If a node cannot come up on its real
  backend within the timeout, :meth:`NodeSupervisor.start` raises with the
  child's captured output - it never quietly swaps in a synthetic/dry-run node.
  That mirrors ``app.py``'s no-fallback policy: run on the real hardware, or
  fail loudly so you can diagnose it. Opt into a stand-in explicitly with the
  per-node flags (``dry_run`` / ``synthetic`` / ``mock``).

* **Idempotent / non-interfering.** Before launching a node we check whether
  its primary topic already has a publisher - started by hand, by another
  supervisor, or by a notebook's in-process demo node. If someone already owns
  the topic we leave it alone. So the self-contained ROS notebooks and this
  supervisor never end up double-owning a device topic.

Python API (used by ``app.py``)::

    from ros.node_supervisor import NodeSupervisor
    with NodeSupervisor(cfg, nodes=["arm", "camera_operator"]) as sup:
        ...   # the arm/camera server nodes are live for the duration

CLI::

    python -m ros.node_supervisor              # all nodes, real hardware
    python -m ros.node_supervisor --mock       # intent node uses keyword brain
    python -m ros.node_supervisor --only arm,camera_operator
    python -m ros.node_supervisor --dry-run --synthetic --mock   # laptop/CI
    python -m ros.node_supervisor --digital-twin  # MuJoCo twin: arm + arm camera

The CLI blocks until Ctrl-C (or SIGTERM), then shuts every child down cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ros.node_supervisor")

# Directory that holds the ``ros`` package (this file is ros/node_supervisor.py),
# so children launched with ``python -m ros.<node>`` resolve their imports.
PACKAGE_DIR = Path(__file__).resolve().parents[1]

# Order matters: the arm parks safely and cameras are cheap; the intent node is
# last because it may spin up llama-server.
ALL_NODES = ("arm", "camera_operator", "camera_arm", "intent")


@dataclass
class _Child:
    """One supervised child node process: identity, readiness topic, argv, and captured log."""

    key: str
    label: str
    ready_topic: str
    argv: list[str]
    proc: Optional[subprocess.Popen] = None
    log: list[str] = field(default_factory=list)
    skipped: bool = False


def _node_specs(
    cfg: dict,
    keys,
    *,
    dry_run: bool,
    synthetic: bool,
    mock: bool,
    sim: bool = False,
    config_path=None,
) -> list[_Child]:
    """Build the ordered :class:`_Child` specs for the requested node ``keys``.

    ``sim``: the arm server is the MuJoCo digital twin (``so101_sim.node``),
    which also publishes the arm camera rendered on its gripper - physics and
    camera must share one process, so no separate ``camera_arm`` child.
    """
    cams = cfg.get("cameras") or {}
    robot = cfg.get("robot") or {}
    intent = cfg.get("intent_node") or {}
    specs: dict[str, _Child] = {
        "arm": _Child(
            key="arm",
            label="SO-101 arm server",
            ready_topic=robot.get("state_topic", "/so101/joint_state"),
            argv=["ros.arm_node", *(["--dry-run"] if dry_run else [])],
        ),
        "camera_operator": _Child(
            key="camera_operator",
            label="operator camera",
            ready_topic=(cams.get("operator") or {}).get(
                "topic", "/cameras/operator/image_raw"
            ),
            argv=[
                "ros.camera_node",
                "--role",
                "operator",
                *(["--synthetic"] if synthetic else []),
            ],
        ),
        "camera_arm": _Child(
            key="camera_arm",
            label="arm camera",
            ready_topic=(cams.get("arm") or {}).get("topic", "/cameras/arm/image_raw"),
            argv=[
                "ros.camera_node",
                "--role",
                "arm",
                *(["--synthetic"] if synthetic else []),
            ],
        ),
        "intent": _Child(
            key="intent",
            label="Llama intent node",
            ready_topic=intent.get("intent_topic", "/vla/intent"),
            argv=["ros.intent_node", *(["--mock"] if mock else [])],
        ),
    }
    if sim:
        cfg_file = str(config_path or PACKAGE_DIR / "config" / "workshop.yaml")
        specs["arm"] = _Child(
            key="arm",
            label="SO-101 digital twin (arm + arm camera)",
            ready_topic=robot.get("state_topic", "/so101/joint_state"),
            argv=["so101_sim.node", "--config", cfg_file, "--stack", "workshop",
                  "--sim-cameras", "arm"],
        )
        keys = [k for k in keys if k != "camera_arm"]
        if "arm" not in keys and "camera_arm" in keys:
            keys = ["arm", *keys]
    return [specs[k] for k in keys if k in specs]


class NodeSupervisor:
    """Start/stop the ROS 2 server nodes as supervised child processes."""

    def __init__(
        self,
        cfg: dict,
        nodes=ALL_NODES,
        *,
        dry_run: bool = False,
        synthetic: bool = False,
        mock: bool = False,
        sim: bool = False,
        config_path=None,
        ready_timeout_s: float = 20.0,
    ):
        self.cfg = cfg
        self.ready_timeout_s = float(ready_timeout_s)
        self._children = _node_specs(
            cfg,
            nodes,
            dry_run=dry_run,
            synthetic=synthetic,
            mock=mock,
            sim=sim,
            config_path=config_path,
        )
        self._probe = None  # lazily-created rclpy probe node
        self._stopped = False

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "NodeSupervisor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- discovery probe ------------------------------------------------------
    def _count_publishers(self, topic: str) -> int:
        """How many publishers DDS currently sees on ``topic`` (0 on any error).

        Uses a throwaway rclpy node so we can tell 'already owned' from 'needs
        launching', and confirm a child is really on the wire before returning.
        """
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
            if self._probe is None:
                self._probe = rclpy.create_node("node_supervisor_probe")
            return int(self._probe.count_publishers(topic))
        except Exception:
            return 0

    def _wait_for_publisher(self, child: _Child, deadline: float) -> bool:
        """Block until ``child``'s ready topic has a publisher or ``deadline`` passes."""
        while time.time() < deadline:
            if child.proc is not None and child.proc.poll() is not None:
                return False  # child exited early
            if self._count_publishers(child.ready_topic) > 0:
                return True
            time.sleep(0.25)
        return False

    # -- process plumbing -----------------------------------------------------
    def _spawn(self, child: _Child) -> None:
        """Launch ``child`` as a ``python -m`` subprocess and start draining its output."""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PACKAGE_DIR), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("PYTHONUNBUFFERED", "1")
        child.proc = subprocess.Popen(
            [sys.executable, "-m", *child.argv],
            cwd=str(PACKAGE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # own group → clean signals
        )
        threading.Thread(target=self._drain, args=(child,), daemon=True).start()

    def _drain(self, child: _Child) -> None:
        """Continuously append ``child``'s subprocess output to its rolling log."""
        assert child.proc is not None and child.proc.stdout is not None
        for line in child.proc.stdout:
            child.log.append(line.rstrip("\n"))
            if len(child.log) > 200:
                del child.log[:100]

    def _tail(self, child: _Child, n: int = 15) -> str:
        """Return the last ``n`` lines of ``child``'s captured log, indented for display."""
        return "\n".join(f"    | {ln}" for ln in child.log[-n:]) or "    | (no output)"

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "NodeSupervisor":
        """Launch every not-already-running child and block until each is publishing."""
        import atexit

        atexit.register(self.stop)  # never orphan children on any exit
        started: list[_Child] = []
        for child in self._children:
            if self._count_publishers(child.ready_topic) > 0:
                child.skipped = True
                print(
                    f"nodes: {child.label} already publishing "
                    f"{child.ready_topic} - leaving it alone"
                )
                continue
            print(
                f"nodes: starting {child.label} "
                f"({' '.join(['python', '-m', *child.argv])})"
            )
            self._spawn(child)
            started.append(child)

        deadline = time.time() + self.ready_timeout_s
        for child in started:
            if not self._wait_for_publisher(child, deadline):
                detail = (
                    "exited early"
                    if child.proc and child.proc.poll() is not None
                    else f"no publisher on {child.ready_topic} within "
                    f"{self.ready_timeout_s:.0f}s"
                )
                self.stop()
                raise RuntimeError(
                    f"node '{child.label}' failed to come up ({detail}).\n"
                    f"{self._tail(child)}\n"
                    "  → check the hardware/backend for this node, source ROS 2 "
                    "(/opt/ros/jazzy/setup.bash), or launch it in the intended "
                    "stand-in mode (--dry-run / --synthetic / --mock)."
                )
            print(f"nodes: {child.label} live on {child.ready_topic} [ok]")
        return self

    def stop(self) -> None:
        """SIGINT every child (so the arm parks), then wait for and reap them all."""
        if self._stopped:
            return
        self._stopped = True
        for child in reversed(self._children):
            proc = child.proc
            if proc is None or proc.poll() is not None:
                continue
            try:  # SIGINT → arm parks at REST
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        deadline = time.time() + 5.0
        for child in reversed(self._children):
            proc = child.proc
            if proc is None:
                continue
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._probe is not None:
            try:
                self._probe.destroy_node()
            except Exception:
                pass
            self._probe = None


def main() -> int:
    """CLI: supervise the requested server nodes until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Supervise the ROS 2 server nodes")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--only",
        default=None,
        help=f"comma list from {','.join(ALL_NODES)} (default: all)",
    )
    ap.add_argument("--dry-run", action="store_true", help="arm node: no hardware")
    ap.add_argument("--synthetic", action="store_true", help="camera nodes: no device")
    ap.add_argument("--mock", action="store_true", help="intent node: keyword brain")
    ap.add_argument(
        "--digital-twin", action="store_true", help="arm + arm camera: MuJoCo digital twin"
    )
    ap.add_argument("--ready-timeout", type=float, default=20.0)
    args = ap.parse_args()

    from common.config import load_config
    from common.ros_helpers import ros2_available

    if not ros2_available():
        print(
            "FATAL: rclpy not importable - source ROS 2 first "
            "(source /opt/ros/jazzy/setup.bash).",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(args.config)
    nodes = (
        [n.strip() for n in args.only.split(",") if n.strip()]
        if args.only
        else list(ALL_NODES)
    )

    sup = NodeSupervisor(
        cfg,
        nodes,
        dry_run=args.dry_run,
        synthetic=args.synthetic,
        mock=args.mock,
        sim=args.digital_twin,
        config_path=args.config,
        ready_timeout_s=args.ready_timeout,
    )
    try:
        sup.start()
    except Exception as e:
        print(f"\nFATAL [nodes]: {e}", file=sys.stderr)
        return 2

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    print("nodes: all server nodes up - Ctrl-C to stop.")
    try:
        stop.wait()
    finally:
        print("\nnodes: shutting down server nodes...")
        sup.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
