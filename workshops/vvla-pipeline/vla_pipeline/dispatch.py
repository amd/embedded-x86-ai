# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Transcript gauntlet + behavior dispatch loop, shared by ``main`` and tests.

``Dispatcher.on_transcript`` is what the always-hot listener (Whisper) or
the ``--text-commands`` stdin loop calls for every utterance:

1. startup grace window (the mic's first noisy moments are ignored);
2. stop fast path - a stop keyword sets ``ctx.stop_event`` immediately,
   without an LLM round trip, interrupting whatever behavior is running;
3. noise / hallucination gate;
4. queue for the dispatch loop - and, if a behavior is running, set the stop
   event so it ends gracefully and the new command takes over (switch).

``Dispatcher.run`` (main thread) parses each queued transcript with the
grammar-constrained LLM, looks the intent up in the registry, runs the
behavior, parks the arm if it crashes, and drains the queue to the newest
command so a backlog never replays stale behaviors.

Every step is reported to ``listeners`` (``fn(event, **data)``) so the
digital-twin tests can timeline it next to the simulation video.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from vla_pipeline.llm.intents import Intent, ParsedCommand

logger = logging.getLogger(__name__)


class Dispatcher:
    """Owns the transcript queue and runs behaviors from it."""

    def __init__(
        self,
        ctx,
        parse,
        registry,
        *,
        is_stop,
        is_noise=None,
        startup_grace_s: float = 0.0,
        clock=time.monotonic,
    ):
        self.ctx = ctx
        self.parse = parse
        self.registry = registry
        self.is_stop = is_stop
        self.is_noise = is_noise or (lambda _t: False)
        self.startup_grace_s = float(startup_grace_s)
        self._clock = clock
        self.start_time = clock()
        self.transcripts: queue.Queue[str] = queue.Queue()
        self.listeners: list = []
        self.current: ParsedCommand | None = None
        self.busy = threading.Event()  # set while a behavior runs
        self.holding = threading.Event()  # set while the console is resetting the scene

    def _emit(self, event: str, **data) -> None:
        for fn in list(self.listeners):
            try:
                fn(event, **data)
            except Exception as e:  # pragma: no cover - listener bugs stay local
                logger.debug("dispatch listener failed: %s", e)

    # ---------------------------------------------------------------- input
    def on_transcript(self, text: str) -> None:
        """Gate one transcript (grace, stop, noise) and queue it for dispatch."""
        logger.info("Heard: %r", text)
        self._emit("heard", text=text)
        if self._clock() - self.start_time < self.startup_grace_s:
            logger.info("Ignoring %r (startup grace period).", text)
            self._emit("ignored", text=text, reason="grace")
            return
        if self.is_stop(text):
            self.ctx.say("Stopping.")
            self.ctx.stop_event.set()
            self._emit("stop", text=text)
            return
        if self.is_noise(text):
            logger.info("Ignoring %r (noise/hallucination gate).", text)
            self._emit("ignored", text=text, reason="noise")
            return
        self.transcripts.put(text)
        self._emit("queued", text=text)
        if self.busy.is_set():
            # A new command switches behaviors: stop the running one gracefully;
            # run() then dispatches the newest queued command.
            logger.info("New command while %s runs - stopping it to switch.",
                        self.current.intent.value if self.current else "a behavior")
            self.ctx.stop_event.set()

    # ------------------------------------------------------------- dispatch
    def dispatch(self, text: str) -> ParsedCommand | None:
        """Parse and run one transcript on the calling thread."""
        cmd = self.parse(text)
        self._emit("parsed", text=text, intent=cmd.intent.value, params=dict(cmd.params or {}))
        if cmd.intent in (Intent.UNKNOWN, Intent.STOP):
            if cmd.intent is Intent.UNKNOWN:
                self.ctx.say("I didn't catch a command in that.")
            return cmd
        handler = self.registry.get(cmd.intent)
        if handler is None:
            self.ctx.say(f"No behavior registered for '{cmd.intent.value}'.")
            return cmd
        self.ctx.stop_event.clear()
        logger.info("=== %s %s ===", cmd.intent.value, cmd.params or "")
        self.current = cmd
        self.busy.set()
        self._emit("behavior_start", intent=cmd.intent.value, params=dict(cmd.params or {}))
        ok = True
        try:
            # a handler that returns False ran but didn't achieve its goal
            ok = handler(self.ctx, cmd) is not False  # main thread (cv2 previews)
        except Exception as e:
            ok = False
            logger.exception("Behavior %s crashed - arm to rest.", cmd.intent.value)
            self._emit("behavior_crash", intent=cmd.intent.value, error=repr(e))
            try:
                self.ctx.arm.go_to_rest()
            except Exception:
                pass
        finally:
            stopped = self.ctx.stop_event.is_set()
            self.ctx.stop_event.clear()
            self.current = None
            self.busy.clear()
            self._emit("behavior_end", intent=cmd.intent.value, ok=ok, stopped=stopped)
        return cmd

    def drain_to_newest(self) -> None:
        """Drop queued transcripts except the newest one."""
        newest = None
        while True:
            try:
                newest = self.transcripts.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            self.transcripts.put(newest)

    def discard(self) -> None:
        """Drop every queued transcript."""
        while True:
            try:
                self.transcripts.get_nowait()
            except queue.Empty:
                return

    def run(self, shutdown: threading.Event) -> None:
        """Dispatch loop until ``shutdown`` is set."""
        while not shutdown.is_set():
            try:
                text = self.transcripts.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.holding.is_set():
                continue
            self.dispatch(text)
            if self.holding.is_set():
                self.discard()
                continue
            self.drain_to_newest()
