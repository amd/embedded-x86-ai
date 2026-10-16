# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Dispatch: how a sentence becomes a behavior.

The orchestrator never hardcodes behaviors. Handlers register on a
:class:`~common.intents.CommandRegistry`, transcripts run a gauntlet -
stop fast path, noise gate, LLM - and whatever survives is dispatched to
whichever handler claimed that intent. Adding a voice command to the real
pipeline never touches the dispatch loop; that property comes from what you
build here.

Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Harness:  ``python app.py --dry-run --synthetic``
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from common.intents import CommandRegistry, Intent, ParsedCommand, is_noise, is_stop

logger = logging.getLogger(__name__)


@dataclass
class WorkshopContext:
    """Everything a behavior handler may need (PROVIDED).

    Handlers receive ``(ctx, cmd)`` - the arm, the models, the camera, the
    mapper, shared state, and the stop event, in one bag. ``stop_check`` is
    the function every behavior loop must poll.
    """

    cfg: dict
    arm: object
    pose: object = None  # PoseEstimator (or scripted stand-in)
    hands: object = None  # HandTracker (or scripted stand-in)
    camera: object = None  # camera client (ROS 2 / direct / synthetic)
    detector: object = None  # ObjectDetector, for pick_place
    arm_camera: object = None  # the wrist camera, for grip and pick_place
    mapper: object = None  # MimicMapper
    registry: CommandRegistry = None
    shared_state: dict = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    headless: bool = False
    max_frames: int = 0  # 0 = unlimited (harness uses a small N)

    def stop_check(self) -> bool:
        """True while an emergency stop is in effect."""
        return self.stop_event.is_set()

    def say(self, message: str) -> None:
        """Print (and log) one line of robot speech."""
        print(f"{message}")
        logger.info("[say] %s", message)


def build_registry() -> CommandRegistry:
    """Register every behavior this project ships on a fresh registry.

    Map :class:`Intent` members to the handler functions in
    ``integration.mimic`` and ``integration.behaviors``:
    GESTURE_MIMIC → ``run_mimic``, DANCE → ``run_dance``, WAVE →
    ``run_wave``, HOME → ``run_home``. (STOP is deliberately NOT a handler -
    it's an interrupt, handled before dispatch ever happens.)

    The provided manipulation behaviors (grip, pick_place, fetch_block, in
    ``common.manipulation``) are added after yours.
    """
    from common.manipulation import register_manipulation
    from integration.behaviors import run_dance, run_home, run_wave
    from integration.mimic import run_mimic

    registry = CommandRegistry()
    # >>> TODO 3.1: build_registry - notebooks/03_integration/01_behavior.ipynb
    registry.add(Intent.GESTURE_MIMIC, run_mimic)
    registry.add(Intent.DANCE, run_dance)
    registry.add(Intent.WAVE, run_wave)
    registry.add(Intent.HOME, run_home)
    # <<< TODO 3.1
    register_manipulation(registry)
    return registry


def handle_transcript(ctx: WorkshopContext, text: str, parse) -> ParsedCommand:
    """Route one raw transcript through the full gauntlet. Returns what ran.

    The order is a safety property - implement it exactly:

    1. **Stop fast path**: :func:`~common.intents.is_stop` on the RAW text →
       ``ctx.stop_event.set()``, say so, return ``ParsedCommand(STOP)``.
       Never send a stop through the LLM.
    2. **Noise gate**: :func:`~common.intents.is_noise` → say nothing, do
       nothing, return ``ParsedCommand(UNKNOWN)``.
    3. **Parse**: ``cmd = parse(text)``.
    4. **Dispatch**: UNKNOWN → tell the user you didn't catch a command.
       No handler registered → say that. Otherwise CLEAR the stop event,
       run ``handler(ctx, cmd)``, and clear it again afterwards (a stale
       stop must never kill the *next* behavior).
    """
    # >>> TODO 3.2: handle_transcript - notebooks/03_integration/01_behavior.ipynb
    if is_stop(text):
        ctx.say("Stopping.")
        ctx.stop_event.set()
        return ParsedCommand(Intent.STOP, transcript=text)
    if is_noise(text):
        logger.info("Ignoring %r (noise gate).", text)
        return ParsedCommand(Intent.UNKNOWN, transcript=text)

    cmd = parse(text)
    if cmd.intent in (Intent.UNKNOWN, Intent.STOP):
        if cmd.intent is Intent.UNKNOWN:
            ctx.say("I didn't catch a command in that.")
        return cmd
    handler = ctx.registry.get(cmd.intent)
    if handler is None:
        ctx.say(f"No behavior registered for '{cmd.intent.value}'.")
        return cmd

    ctx.stop_event.clear()
    logger.info("=== %s %s ===", cmd.intent.value, cmd.params or "")
    try:
        handler(ctx, cmd)
    finally:
        ctx.stop_event.clear()
    return cmd
    # <<< TODO 3.2
