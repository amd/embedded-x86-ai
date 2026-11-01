# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Simple behaviors: dance (your TODO), wave and home (the provided template).

``run_wave``/``run_home`` are the canonical shape of a behavior handler:
``(ctx, cmd)`` in, motion through ``ctx.arm``, ``ctx.stop_check`` polled so a
spoken "stop" interrupts mid-move, park at the end. ``run_dance`` is the
same shape with a beat clock - the choreography (:data:`common.motion.
DANCE_MOVES`) is a provided static asset; the control flow is the exercise.

Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Harness:  ``python app.py --dry-run --synthetic`` (then type ``dance``)
"""

from __future__ import annotations

import itertools
import logging
import random
import time

from common.motion import DANCE_MOVES, WAVE_LEFT, WAVE_RIGHT, WAVE_UP

logger = logging.getLogger(__name__)


def run_dance(ctx, cmd=None, duration_s=None, seed=None) -> None:
    """Beat-paced keyframe choreography until the clock (or a stop) ends it.

    Read ``dance.bpm`` and ``dance.duration_s`` from ``ctx.cfg`` (honor the
    ``duration_s`` override). One beat is ``60/bpm`` seconds. Shuffle the
    move order (:data:`~common.motion.DANCE_MOVES`), then cycle through it;
    play each move's keyframes ×4 with ``ctx.arm.go_to(keyframe,
    duration_s=beat*0.9, stop_check=ctx.stop_check)``. Check the deadline
    AND ``ctx.stop_check()`` between keyframes - a dance that can't be
    stopped is a safety bug, not a feature. A stop holds the current pose;
    an explicit ``home`` command parks the arm.
    """
    d = ctx.cfg.get("dance", {}) or {}
    # >>> TODO 3.4: run_dance - notebooks/03_integration/01_behavior.ipynb
    bpm = float(d.get("bpm", 100))
    duration_s = float(d.get("duration_s", 20.0)) if duration_s is None else duration_s
    beat = 60.0 / bpm
    rng = random.Random(seed)

    logger.info("Dancing for %.0f s at %.0f BPM", duration_s, bpm)
    order = list(DANCE_MOVES)
    rng.shuffle(order)
    move_cycle = itertools.cycle(order)

    deadline = time.time() + duration_s
    try:
        while time.time() < deadline:
            if ctx.stop_check():
                break
            move = next(move_cycle)
            logger.info("[dance] move: %s", move)
            for keyframe in DANCE_MOVES[move] * 4:
                if time.time() >= deadline or ctx.stop_check():
                    break
                ctx.arm.go_to(
                    keyframe, duration_s=beat * 0.9, stop_check=ctx.stop_check
                )
    finally:
        if ctx.stop_check():
            logger.info("Dance stopped - holding position.")
        else:
            logger.info("Dance over - returning to rest pose.")
            ctx.arm.go_to_rest(stop_check=ctx.stop_check)
    # <<< TODO 3.4


def run_wave(ctx, cmd=None) -> None:
    """PROVIDED: raise the arm and wave ``wave.cycles`` times (the template)."""
    cycles = int(ctx.cfg.get("wave", {}).get("cycles", 3))
    logger.info("[wave] waving %d times", cycles)
    ctx.arm.go_to(WAVE_UP, duration_s=2.0, stop_check=ctx.stop_check)
    for _ in range(cycles):
        if ctx.stop_check():
            break
        ctx.arm.go_to(WAVE_LEFT, duration_s=0.4, stop_check=ctx.stop_check)
        ctx.arm.go_to(WAVE_RIGHT, duration_s=0.4, stop_check=ctx.stop_check)
    ctx.arm.go_to_rest()


def run_home(ctx, cmd=None) -> None:
    """PROVIDED: return to the rest pose."""
    logger.info("[home] returning to rest pose")
    ctx.arm.go_to_rest(stop_check=ctx.stop_check)
