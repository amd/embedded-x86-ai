# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""Workshop entry point - the component-3 live UI harness (PROVIDED).

Launches the gesture-mimic UI immediately: your webcam on the left with the
detected skeleton, a schematic SO-101 on the right mirroring you, and a HUD
attributing every joint to the TODO whose output drives it - while the real
arm tracks along.

FAIL-FAST POLICY (no silent fallbacks): on the real rig every stage must run
on its intended hardware - the operator camera, the ROS 2 arm, and the AI
accelerators. If a stage can't come up on its real backend, the app exits with
a diagnostic instead of quietly substituting a synthetic/dry-run/keyword
stand-in, so you can see and fix what's actually wrong. The ONE fallback that
stays is NPU→CPU inside ONNX Runtime (the VitisAI→CPU execution-provider
fallback), which is a device placement detail, not a stub.

The stand-ins still exist, but ONLY as explicit opt-in modes for laptops/CI -
never as an automatic fallback:

    python app.py                        # the rig: NPU pose, ROS 2 arm, webcam
    python app.py --dry-run              # explicitly simulate the arm
    python app.py --synthetic            # explicitly skip the camera/hands
    python app.py --dry-run --synthetic --no-llm     # laptop mode, everything faked
    python app.py --dry-run --synthetic --headless --frames 60   # CI-style check
    python app.py --digital-twin         # no robot hardware: MuJoCo twin of the arm

The app also brings up the ROS 2 server nodes it needs (the arm + operator-camera
publishers) as child processes and tears them down on exit - so you don't need
a terminal per node. Pass --no-nodes if you start them yourself.

While the UI runs, type commands into the terminal ("dance", "wave",
"go home", "copy my movements", "stop") - they run through YOUR
handle_transcript gauntlet, parsed by Llama on the iGPU (or the keyword
stand-in only under --no-llm).
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from typing import NoReturn

logger = logging.getLogger("app")


def _die(stage: str, detail: str, hint: str = "") -> "NoReturn":
    """Abort with a clear diagnostic instead of substituting a stand-in.

    The rig is expected to run on real hardware; a stage that can't come up on
    its intended backend is a fault to diagnose, not something to paper over
    with a synthetic/dry-run stub. Pass --dry-run/--synthetic/--no-llm to opt
    into a stand-in explicitly.
    """
    msg = f"\nFATAL [{stage}]: {detail}"
    if hint:
        msg += f"\n       {hint}"
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _start_nodes(cfg, args):
    """Bring up the ROS 2 server nodes this app needs, as child processes.

    'Server nodes own hardware; everything else subscribes' - so instead of a
    terminal per node, the app starts its own arm + operator-camera servers and
    then connects to them as a client. The supervisor is fail-fast and
    idempotent: a node already publishing (you started it, or --no-nodes) is
    left alone. Returns the supervisor (stop it on exit) or None.

    We only launch what the app actually consumes over ROS 2, and never the
    explicit stand-in paths: --dry-run keeps the arm in-process, --synthetic
    keeps the camera in-process, and direct-serial mode needs no server.
    """
    if args.no_nodes:
        return None
    from common.ros_helpers import ros2_available

    if not ros2_available():
        return None  # no rclpy → client build reports it

    wanted = []
    if not args.dry_run and (cfg.get("robot") or {}).get("use_ros2", True):
        wanted.append("arm")
    if not args.synthetic and ((cfg.get("cameras") or {}).get("operator") or {}).get(
        "use_ros2", False
    ):
        wanted.append("camera_operator")
    if not wanted:
        return None

    from ros.node_supervisor import NodeSupervisor

    print("-- starting ROS 2 server nodes --")
    sup = NodeSupervisor(
        cfg, wanted, sim=args.digital_twin and not args.dry_run, config_path=args.config
    )
    try:
        sup.start()
    except Exception as e:
        _die(
            "nodes",
            f"{e}",
            "Pass --no-nodes if you start the server nodes yourself, or "
            "--dry-run/--synthetic to run without that hardware.",
        )
    return sup


def _build_arm(cfg, args):
    from common.motion import DryRunArm
    from common.ros_helpers import ros2_available

    if args.dry_run:
        print("arm: dry-run (explicit --dry-run)")
        return DryRunArm()
    if cfg["robot"].get("use_ros2", True):
        if not ros2_available():
            _die(
                "arm",
                "ROS 2 requested (robot.use_ros2) but rclpy is not importable.",
                "source /opt/ros/jazzy/setup.bash, or set robot.use_ros2: false "
                "for direct serial, or pass --dry-run to simulate the arm.",
            )
        try:
            from ros.arm_client import Ros2ArmClient

            arm = Ros2ArmClient(cfg)
            print("arm: ROS 2 client → " + cfg["robot"]["command_topic"])
            return arm
        except Exception as e:
            _die(
                "arm",
                f"ROS 2 client would not come up ({e}).",
                "Is the arm server node running? "
                "python -m ros.arm_node   (add --dry-run to simulate the arm).",
            )
    else:
        try:
            from common.motion import make_arm_backend

            arm = make_arm_backend(cfg)
            if args.digital_twin:
                print("arm: MuJoCo digital twin (in-process)")
            else:
                print("arm: direct serial (sole bus owner!)")
            return arm
        except Exception as e:
            _die(
                "arm",
                f"Direct serial backend unavailable ({e}).",
                "Check robot.motor_port / permissions "
                "(sudo chmod 666 /dev/ttyACM0), or pass --dry-run.",
            )


def _build_camera(cfg, args):
    from common.fixtures import SyntheticCamera

    if args.synthetic:
        print("camera: synthetic test pattern (explicit --synthetic)")
        return SyntheticCamera()
    try:
        from ros.camera_client import make_camera

        cam = make_camera(cfg, "operator")
    except Exception as e:
        _die(
            "camera",
            f"Could not construct the operator camera client ({e}).",
            "Check cameras.operator in config; pass --synthetic to skip the camera.",
        )
    ok, _ = cam.read()
    if not ok:
        import time

        time.sleep(0.5)  # give a ROS stream a beat to warm up
        ok, _ = cam.read()
    if ok:
        print("camera: operator role via config")
        return cam
    cam.close()
    _die(
        "camera",
        "operator camera is not delivering frames.",
        "Is the camera node publishing? "
        "python -m ros.camera_node --role operator   (or check the device index / "
        "cameras.operator.use_ros2). Pass --synthetic to skip the camera.",
    )


def _build_pose(cfg, args):
    try:
        from models.vision import PoseEstimator

        pose = PoseEstimator(cfg, device=args.device)
    except FileNotFoundError as e:
        _die(
            "pose",
            f"YOLOv26s-pose model missing ({e}).",
            "Export/compile it: scripts/export_yolo26s_pose.py then "
            "scripts/compile_npu_models.py (or set yolo_pose.device: cpu).",
        )
    except Exception as e:
        _die(
            "pose",
            f"PoseEstimator would not initialize ({e}).",
            "Confirm the NPU/VitisAI EP loads (source scripts/ryzen_ai_env.sh); "
            "NPU→CPU execution-provider fallback is allowed, a hard failure is not.",
        )
    print(f"pose: YOLOv26s-pose on {pose.provider}")
    return pose


def _build_hands(cfg, args):
    from common.fixtures import ScriptedHands

    if args.synthetic:
        print("hands: scripted pinch/roll (explicit --synthetic)")
        return ScriptedHands()
    try:
        from models.hands import HandTracker

        hands = HandTracker(cfg)
    except ImportError as e:
        _die(
            "hands",
            f"mediapipe not importable ({e}).",
            "pip install mediapipe, or pass --synthetic for scripted hands.",
        )
    except Exception as e:
        _die(
            "hands",
            f"MediaPipe Hands would not initialize ({e}).",
            "Pass --synthetic to use scripted hands.",
        )
    print("hands: MediaPipe Hands on CPU")
    return hands


def _build_parser(cfg, args):
    from common.fixtures import offline_parse_text
    from common.intents import Intent, ParsedCommand

    def offline(text: str) -> ParsedCommand:
        """Keyword stand-in: route ``text`` into a ParsedCommand (no LLM)."""
        d = offline_parse_text(text)
        params = {"object": d["object"]} if d.get("object") else {}
        return ParsedCommand(
            Intent.from_string(d["command"]), params=params, transcript=text
        )

    if args.no_llm:
        print("brain: keyword stand-in (explicit --no-llm)")
        return offline
    try:
        from models.llm import LlamaIntentParser

        parser = LlamaIntentParser(cfg)
    except Exception as e:
        _die(
            "brain",
            f"llama-server / Llama parser unavailable ({e}).",
            "Check the GGUF model and llama.cpp build (HSA_OVERRIDE_GFX_VERSION "
            "for the iGPU); pass --no-llm for keyword routing.",
        )
    print("brain: Llama 3.2 3B on the iGPU (grammar-constrained)")
    return parser.parse


def _mode_label(args) -> str:
    """The exact command line for this run, so the preflight can name the mode."""
    flags = []
    if args.dry_run:
        flags.append("--dry-run")
    if args.synthetic:
        flags.append("--synthetic")
    if args.no_llm:
        flags.append("--no-llm")
    if args.no_nodes:
        flags.append("--no-nodes")
    if args.digital_twin:
        flags.append("--digital-twin")
    if args.headless:
        flags.append("--headless")
    if args.frames:
        flags.append(f"--frames {args.frames}")
    return "python app.py" + (
        "".join(" " + f for f in flags) if flags else "   # the rig"
    )


def _required_todos(cfg, args) -> set:
    """The TODO ids THIS invocation must have implemented before it can come up.

    Mirrors the builders above: the opt-out flags and the config drop the
    stages they skip, so a laptop run needs far fewer TODOs than the rig. Two
    groups are deliberately never blockers - they're reported as "remaining"
    but startup does not wait on them:
      - 2.10 / 2.11 (the ROS intent node) - app.py parses typed commands
        in-process and never routes through that node.
      - 3.4 (run_dance) - one optional behavior; the UI launches without it,
        and typing "dance" before it's done is handled gracefully at runtime.
    """
    need = {"1.1", "1.2", "1.3"}  # pose always builds an NPU/CPU session
    need |= {"3.1", "3.3"}  # registry is always built; mimic IS the UI
    if not args.frames:
        need |= {"3.2"}  # interactive: the transcript gauntlet
    if not args.no_llm:
        need |= {"1.5", "1.6"}  # llama-server brain
    if not args.synthetic:
        need |= {"1.4"}  # MediaPipe hands
        cams = (cfg.get("cameras") or {}).get("operator") or {}
        if cams.get("use_ros2", False):
            need |= {"2.3"}  # camera client (subscriber)
            if not args.no_nodes:
                need |= {"2.1", "2.2"}  # camera server node we auto-start
    if not args.dry_run and (cfg.get("robot") or {}).get("use_ros2", True):
        need |= {"2.7", "2.8", "2.9"}  # arm client (pub/sub + send)
        if not args.no_nodes:
            need |= {"2.4", "2.5", "2.6"}  # arm server node we auto-start
    return need


def _preflight(cfg, args) -> bool:
    """List the TODOs still to do instead of crashing on the first one.

    Returns True when the app should stop now - a TODO this run needs is still
    a stub - after printing the checklist (the caller then exits cleanly, code
    0). Returns False when everything THIS invocation needs is implemented, so
    the real build runs and fails fast on hardware exactly as before.
    """
    from common import todos

    left = set(todos.remaining())
    if not left:
        return False  # nothing stubbed anywhere - carry on

    def _k(s):
        return [int(x) for x in s.split(".")]

    needed = _required_todos(cfg, args)
    blocking = sorted(left & needed, key=_k)
    other = sorted(left - set(blocking), key=_k)

    print("\n== TODO preflight ==")
    print("  " + todos.progress_line())

    if not blocking:
        # Something's left, but not for this mode - let the app come up.
        print(
            f"  {len(other)} TODO(s) remain but none are needed to launch "
            f"'{_mode_label(args).strip()}'."
        )
        if other:
            print("  (still to do elsewhere: " + todos.compact(other) + ")")
        return False

    print(
        f"\n'{_mode_label(args).strip()}' still needs {len(blocking)} "
        f"TODO(s) implemented before it can start:\n"
    )
    print(todos.format_list(blocking))
    if other:
        print(
            "\nAlso still to do (not needed for this run mode): " + todos.compact(other)
        )
    print(
        "\nEach stub points at the notebook that teaches it; diff against "
        "solution/ when stuck."
    )
    print("Check your progress without the rig:")
    print("    python selftest.py                 # whole suite + solution parity")
    print("    python -m models.selftest          # component 1 only")
    print("    python -m ros.selftest --fake      # component 2 (no ROS 2 needed)")
    print("    python -m integration.selftest     # component 3 only")
    print("\nOr shrink what THIS run needs with the laptop/CI flags:")
    print("    python app.py --dry-run --synthetic --no-llm")
    return True


def main() -> int:
    """Build the stack, run the live UI (or the --frames harness), and return the exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="VVLA workshop live UI")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="no robot hardware")
    ap.add_argument("--synthetic", action="store_true", help="no camera needed")
    ap.add_argument(
        "--device",
        choices=["npu", "cpu"],
        default=None,
        help="override the ONNX device",
    )
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument(
        "--digital-twin",
        action="store_true",
        help="drive the MuJoCo digital twin (arm + arm camera) instead of the rig",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="run N mimic frames then exit (harness mode)",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="keyword intent routing instead of llama-server",
    )
    ap.add_argument(
        "--no-nodes",
        action="store_true",
        help="don't auto-start the ROS 2 server nodes (arm/camera) - "
        "use this if you're running them yourself",
    )
    args = ap.parse_args()

    from common.config import PROJECT_ROOT, load_config

    cfg = load_config(args.config)
    if args.digital_twin:
        cfg.setdefault("robot", {})["backend"] = "sim"
        cfg.setdefault("cameras", {}).setdefault("arm", {})["source"] = "sim"

    # Fail-soft on unfinished work: list the TODOs this run needs instead of
    # crashing on the first stub. Heavy imports come AFTER this, so a fresh
    # checkout prints its checklist even before every dependency is installed.
    if _preflight(cfg, args):
        return 0

    from integration.dispatch import WorkshopContext, build_registry, handle_transcript
    from common.motion import mapper_from_config

    supervisor = _start_nodes(cfg, args)

    print("-- building the stack --")
    ctx = WorkshopContext(
        cfg=cfg,
        arm=_build_arm(cfg, args),
        camera=_build_camera(cfg, args),
        pose=_build_pose(cfg, args),
        hands=_build_hands(cfg, args),
        mapper=mapper_from_config(cfg),
        headless=args.headless,
        max_frames=args.frames,
    )

    try:
        ctx.registry = build_registry()
    except Exception as e:
        _die(
            "registry",
            f"build_registry() failed ({e}).",
            "Typed commands cannot be dispatched - fix the behavior registry.",
        )

    parse = _build_parser(cfg, args)

    # ---------------- harness mode: N frames, snapshot, exit ----------------
    if args.frames:
        from integration.mimic import run_mimic

        run_mimic(ctx)
        snap = ctx.shared_state.get("last_ui_frame")
        if snap is not None:
            import cv2

            out = PROJECT_ROOT / "_artifacts"
            out.mkdir(exist_ok=True)
            path = out / "ui_snapshot.png"
            cv2.imwrite(str(path), snap)
            print(f"UI snapshot: {path}")
        joints = ctx.arm.get_joints()
        print(f"final arm pose: { {k: round(v, 1) for k, v in joints.items()} }")
        print("mimic ran end-to-end: frames → pose → mapper → arm [ok]")
        if supervisor is not None:
            supervisor.stop()
        return 0

    # ---------------- interactive mode ----------------
    from common.intents import is_stop

    commands: queue.Queue = queue.Queue()

    def stdin_loop():
        """Always-hot input thread (mirrors the pipeline's listener).

        Behaviors run on the MAIN thread (cv2 previews want that), so while
        one is running the dispatch loop below is blocked - a typed 'stop'
        must therefore interrupt from HERE, by setting the shared stop event
        every behavior's stop_check polls. Everything else is queued for
        dispatch once the current behavior yields.
        """
        while True:
            try:
                line = input()
            except EOFError:
                ctx.stop_event.set()  # end any running behavior
                commands.put("__quit__")
                return
            line = line.strip()
            if not line:
                continue
            if is_stop(line):  # safety stop: interrupt NOW
                ctx.say("Stopping.")
                ctx.stop_event.set()
                continue
            if line.lower() in ("quit", "exit"):
                ctx.stop_event.set()  # stop the behavior, then exit
                commands.put("__quit__")
                return
            commands.put(line)

    threading.Thread(target=stdin_loop, daemon=True).start()

    print("\n-- live UI --")
    print(
        "type: 'dance', 'wave', 'go home', 'copy my movements', 'stop', "
        "'quit'   ('q' in the window also stops a behavior)"
    )
    print("starting gesture mimic...\n")

    from integration.mimic import run_mimic

    run_mimic(ctx)
    while True:
        try:
            text = commands.get(timeout=0.2)
        except queue.Empty:
            continue
        if text.lower() in ("__quit__", "quit", "exit"):
            break
        try:
            handle_transcript(ctx, text, parse)
        except NotImplementedError as e:
            # A behavior (or the gauntlet itself) isn't implemented yet - say so
            # and keep the UI alive instead of dropping a traceback mid-session.
            ctx.say(f"(not wired up yet: {e})")
        # Seamless switching: drain queued commands except the newest, so a
        # backlog typed during a behavior doesn't replay stale ones.
        newest = None
        while True:
            try:
                newest = commands.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            commands.put(newest)

    print("bye - parking the arm.")
    try:
        ctx.arm.go_to_rest(duration_s=2.0)
    finally:
        for closer in (
            getattr(ctx.camera, "close", None),
            getattr(ctx.hands, "close", None),
            getattr(ctx.arm, "close", None),
        ):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        if supervisor is not None:  # stop the arm/camera server nodes we spawned
            supervisor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
