# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pipeline orchestrator: microphone → Whisper (CPU by default, optional NPU) → Llama intent (iGPU) →
behavior dispatch, with an always-hot listener for the safety stop.

Architecture
------------
- A **listener thread** runs :class:`FastListener` continuously (VAD-based,
  ~0.35 s end-of-speech) and pushes transcripts onto a queue - the mic stays
  hot *while behaviors run*, which is what makes a spoken "stop" able to
  interrupt a running behavior.
- ``STOP`` is matched with plain keywords on the raw transcript
  (:data:`STOP_KEYWORDS`) before any LLM round trip, then sets the shared
  stop event that every behavior's ``stop_check`` polls (including mid
  ``go_to`` interpolation). The 'q' key in any preview window and Ctrl-C
  do the same.
- Every other transcript goes through the grammar-constrained intent parser
  and is dispatched via the :class:`CommandRegistry` - adding a voice command
  never touches this file (see :mod:`vla_pipeline.llm.intents`).
- Behaviors run on the **main thread** (OpenCV preview windows want that);
  a new command interrupts the current behavior (graceful stop → switch).

Run
---
    python -m vla_pipeline.main                  # full pipeline
    python -m vla_pipeline.main --text-commands  # type instead of speak
    python -m vla_pipeline.main --dry-run        # no robot hardware
"""

from __future__ import annotations

import argparse
import logging
import re
import threading
from dataclasses import dataclass, field

from vla_pipeline.llm.intents import (
    CommandRegistry,
    Intent,
    ParsedCommand,
)
from vla_pipeline.utils.config import ensure_dirs, load_config

logger = logging.getLogger(__name__)

registry = CommandRegistry()


# =============================================================================
# Context shared with behavior handlers
# =============================================================================


@dataclass
class PipelineContext:
    """Shared state passed to every behavior handler: robot arm, vision models, camera clients, and stop/say hooks."""

    cfg: dict
    arm: object
    pose: object | None = None  # PoseEstimator (NPU)
    detector: object | None = None  # ObjectDetector (NPU)
    hands: object | None = None  # HandTracker (CPU)
    operator_camera: object | None = None  # CameraClient
    arm_camera: object | None = None  # CameraClient
    shared_state: dict = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    headless: bool = False

    def stop_check(self) -> bool:
        """Return True if a stop has been requested."""
        return self.stop_event.is_set()

    def say(self, message: str) -> None:
        """Print message to the console and log it."""
        print(f"{message}")
        logger.info("[say] %s", message)


# =============================================================================
# Behavior handlers (the registry IS the dispatch table)
# =============================================================================


@registry.register(Intent.GESTURE_MIMIC)
def _h_gesture_mimic(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_gesture_mimic`."""
    from vla_pipeline.behaviors.gesture_mimic import run_gesture_mimic

    if ctx.pose is None or ctx.operator_camera is None:
        ctx.say("Gesture mimic needs the operator camera and pose model.")
        return
    if ctx.shared_state.pop("grip_lock", None):
        ctx.shared_state.pop("grip_lock_pos", None)
        ctx.say("Your pinch drives the gripper now. Open your fingers to let go.")
    run_gesture_mimic(
        ctx.cfg,
        ctx.arm,
        ctx.pose,
        ctx.hands,
        ctx.operator_camera,
        stop_check=ctx.stop_check,
        show_preview=not ctx.headless,
    )


@registry.register(Intent.FETCH_BLOCK)
def _h_fetch_block(ctx: PipelineContext, cmd: ParsedCommand) -> bool:
    """Dispatch to :func:`run_fetch_block`."""
    from vla_pipeline.behaviors.fetch_block import run_fetch_block

    return run_fetch_block(
        ctx.cfg,
        ctx.arm,
        ctx.pose,
        ctx.operator_camera,
        stop_check=ctx.stop_check,
        feedback=ctx.say,
    )


@registry.register(Intent.PICK_PLACE)
def _h_pick_place(ctx: PipelineContext, cmd: ParsedCommand) -> bool:
    """Dispatch to :func:`run_pick_place` for the requested object."""
    from vla_pipeline.behaviors.pick_place import run_pick_place

    if ctx.detector is None:
        ctx.say("Object detection isn't available - can't do pick and place.")
        return False
    return run_pick_place(
        ctx.cfg,
        ctx.arm,
        ctx.detector,
        ctx.pose,
        ctx.arm_camera,
        ctx.operator_camera,
        object_name=cmd.object_name or "",
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        feedback=ctx.say,
    )


@registry.register(Intent.DANCE)
def _h_dance(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_dance`."""
    from vla_pipeline.behaviors.dance import run_dance

    run_dance(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


@registry.register(Intent.RELEASE)
def _h_release(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_release`."""
    from vla_pipeline.behaviors.grip import run_release

    run_release(
        ctx.arm,
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        feedback=ctx.say,
    )


@registry.register(Intent.GRIP)
def _h_grip(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_grip`."""
    from vla_pipeline.behaviors.grip import run_grip

    run_grip(
        ctx.cfg,
        ctx.arm,
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        arm_camera=ctx.arm_camera,
        feedback=ctx.say,
    )


@registry.register(Intent.WAVE)
def _h_wave(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_wave`."""
    from vla_pipeline.behaviors.simple import run_wave

    run_wave(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


@registry.register(Intent.HOME)
def _h_home(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_home`."""
    from vla_pipeline.behaviors.simple import run_home

    run_home(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


# =============================================================================
# Stop fast path
# =============================================================================

def is_stop(transcript: str) -> bool:
    """Return True if transcript contains a stop keyword."""
    from vla_pipeline.robot.arm_interface import ensure_common_path

    ensure_common_path()
    from common.intents import is_stop as workshop_is_stop

    return workshop_is_stop(transcript)


# =============================================================================
# Noise / hallucination gate
# =============================================================================
# Whisper, fed the first burst of room noise the moment the mic goes hot,
# reliably hallucinates a few stock phrases ("you", "thank you.", "Bye.",
# subtitle credits, etc.). With the grammar forcing every transcript into one
# valid command, such a phantom utterance used to dispatch a real behavior - the
# robot appearing to wave/dance on its own at startup. We drop these before they
# ever reach the LLM. The set is matched on the transcript stripped of
# whitespace/punctuation and lowercased; multi-word phrases are matched whole.
DEFAULT_NOISE_PHRASES = (
    "",
    "you",
    "thank you",
    "thanks",
    "thank you.",
    "bye",
    "bye.",
    "goodbye",
    "hello",
    "hi",
    "hmm",
    "uh",
    "um",
    "ah",
    "oh",
    "yeah",
    "okay",
    "ok",
    "the",
    "a",
    "so",
    "and",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by the amara.org community",
    ".",
    "..",
    "...",
)


_NON_SPEECH_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip surrounding whitespace and trailing punctuation/quotes."""
    return re.sub(r"[\s\.\,\!\?\"\'`]+$", "", (text or "").strip().lower()).strip()


def make_noise_gate(cfg: dict):
    """Build an ``is_noise(text) -> bool`` predicate from config.

    ``audio.noise_phrases`` (optional) extends/overrides the defaults; a very
    short transcript with no alphabetic content is always treated as noise.
    """
    a = cfg.get("audio", {}) or {}
    phrases = a.get("noise_phrases")
    phrase_set = {
        (_normalize_transcript(p))
        for p in (phrases if phrases is not None else DEFAULT_NOISE_PHRASES)
    }

    def is_noise(text: str) -> bool:
        """Return True if text looks like noise/hallucination rather than a real command."""
        # Whisper tags non-speech as "[BLANK_AUDIO]", "(sighs)", "*music*" - drop those.
        norm = _normalize_transcript(_NON_SPEECH_TAG.sub(" ", text or ""))
        if not norm:
            return True
        if not any(c.isalpha() for c in norm):
            return True  # pure punctuation / digits
        return norm in phrase_set

    return is_noise


# =============================================================================
# Orchestrator
# =============================================================================


def apply_sim_overrides(cfg: dict) -> dict:
    """Switch ``cfg`` to the digital twin: sim arm + sim arm camera, real operator camera."""
    cfg.setdefault("robot", {})["backend"] = "sim"
    cams = cfg.setdefault("cameras", {})
    cams.setdefault("arm", {})["source"] = "sim"
    cams.setdefault("operator", {}).setdefault("source", "device")
    return cfg


def build_context(cfg: dict, args) -> PipelineContext:
    """Build the :class:`PipelineContext`, wiring up the arm, cameras, and vision models.

    Each vision component is optional: a failure to construct it is logged and
    left as None so the pipeline still runs with degraded capabilities.
    """
    from vla_pipeline.robot.arm_interface import DryRunArm, make_arm

    arm = DryRunArm(verbose=False) if args.dry_run else make_arm(cfg)
    ctx = PipelineContext(cfg=cfg, arm=arm, headless=args.headless)

    from vla_pipeline.vision.camera import make_camera

    try:
        ctx.operator_camera = make_camera(cfg, "operator")
    except Exception as e:
        logger.warning("Operator camera unavailable: %s", e)
    try:
        ctx.arm_camera = make_camera(cfg, "arm")
    except Exception as e:
        logger.warning("Arm camera unavailable: %s", e)

    device = args.device
    try:
        from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

        ctx.pose = PoseEstimator(cfg, device=device)
    except Exception as e:
        logger.warning("Pose model unavailable (%s) - mimic/handoff degraded.", e)
    try:
        from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

        ctx.detector = ObjectDetector(cfg, device=device)
    except Exception as e:
        logger.warning("Detector unavailable (%s) - pick_place disabled.", e)
    try:
        from vla_pipeline.vision.mediapipe_hands import HandTracker

        ctx.hands = HandTracker(cfg)
    except Exception as e:
        logger.warning("Hand tracker unavailable: %s", e)
    return ctx


def main() -> None:
    """Parse CLI args, build the pipeline context, and run the listen/dispatch loop until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Strix VLA pipeline orchestrator")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--text-commands",
        action="store_true",
        help="type commands instead of speaking (skips Whisper)",
    )
    parser.add_argument("--dry-run", action="store_true", help="no robot hardware")
    parser.add_argument("--headless", action="store_true", help="no preview windows")
    parser.add_argument(
        "--device",
        choices=["npu", "cpu"],
        default=None,
        help="override the device for all ONNX models",
    )
    parser.add_argument(
        "--digital-twin",
        action="store_true",
        help="drive the MuJoCo digital twin instead of the real arm "
        "(robot.backend=sim, arm camera rendered on the gripper)",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="with --digital-twin: skip the twin console window",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.digital_twin:
        apply_sim_overrides(cfg)
    ensure_dirs(cfg)
    ctx = build_context(cfg, args)

    console = None
    if args.digital_twin and not (args.headless or args.no_console):
        from vla_pipeline.console import TwinConsole

        console = TwinConsole(ctx, voice="typed" if args.text_commands else "mic")

    from vla_pipeline.dispatch import Dispatcher
    from vla_pipeline.llm.llama_intent import LlamaIntentParser

    llm = LlamaIntentParser(cfg)
    # Ignore any transcript for a short grace window after launch - the mic goes
    # hot before the user is ready and the first frames are usually room noise
    # that Whisper hallucinates into words. Voice ('stop' included) only starts
    # being honored once the window passes. Tunable via audio.startup_grace_s.
    dispatcher = Dispatcher(
        ctx,
        llm.parse,
        registry,
        is_stop=is_stop,
        is_noise=make_noise_gate(cfg),
        startup_grace_s=float((cfg.get("audio", {}) or {}).get("startup_grace_s", 1.5)),
    )
    _on_transcript = dispatcher.on_transcript
    shutdown = threading.Event()
    if console is not None:
        console.attach(dispatcher)

    # ---------------- listener thread (always hot) ----------------
    if not args.text_commands:
        from vla_pipeline.audio.vad_listener import FastListener
        from vla_pipeline.audio.whisper_npu import WhisperNPU

        whisper = WhisperNPU(cfg, device=args.device)
        listener = FastListener(cfg, whisper.transcribe)

        def _listen() -> None:
            """Run the always-hot listener loop, forwarding transcripts to :func:`_on_transcript`."""
            listener.run_forever(
                on_transcript=lambda t: _on_transcript(t),
                stop_check=shutdown.is_set,
                on_level=console.on_level if console is not None else None,
                on_busy=console.note_asr if console is not None else None,
            )

        threading.Thread(target=_listen, daemon=True, name="listener").start()
    else:

        def _stdin_loop() -> None:
            """Read typed commands from stdin (--text-commands mode) and forward them to :func:`_on_transcript`."""
            while not shutdown.is_set():
                try:
                    line = input("you> ")
                except EOFError:
                    shutdown.set()
                    break
                if line.strip():
                    _on_transcript(line.strip())

        threading.Thread(target=_stdin_loop, daemon=True, name="stdin").start()

    # ---------------- dispatch loop (main thread, unless the console owns it) ----------------
    ctx.say(
        "Ready. Try: 'copy my movements', 'pick up the ball', 'wave', "
        "'dance', 'hold this', 'go home' - or 'stop' at any time. "
        "In the twin window: space stops, r resets the arm and the table."
    )
    try:
        if console is None:
            dispatcher.run(shutdown)
        else:
            worker = threading.Thread(
                target=dispatcher.run, args=(shutdown,), daemon=True, name="dispatch"
            )
            worker.start()
            console.run(shutdown, on_stop=lambda: _on_transcript("stop"))
            ctx.stop_event.set()
            worker.join(timeout=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()
        ctx.stop_event.set()
        logger.info("Shutting down - parking arm.")
        for closer in (
            getattr(ctx.operator_camera, "close", None),
            getattr(ctx.arm_camera, "close", None),
            getattr(ctx.hands, "close", None),
        ):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        try:
            ctx.arm.close()
            if args.digital_twin:
                from so101_sim import close_shared_world

                close_shared_world()
        finally:
            llm.close()


if __name__ == "__main__":
    main()
