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
   without an LLM round trip, interrupting whatever behavior is running,
   and cancels a command whose parse it overlaps;
3. noise / hallucination gate;
4. queue for the dispatch loop - and, if a behavior is running, set the stop
   event so it ends gracefully and the new command takes over (switch).

``Dispatcher.run`` (main thread) parses each queued transcript with the
grammar-constrained LLM, looks the intent up in the registry, runs the
behavior, and drains the queue to the newest command so a backlog never
replays stale behaviors. It blocks on the queue;
``Dispatcher.stop`` wakes it to see its ``shutdown`` event.

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

_STOP = object()  # queued by Dispatcher.stop: run() wakes and checks its shutdown event


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
        self.transcripts: queue.Queue[tuple[int, str] | object] = queue.Queue()
        self.listeners: list = []
        self.current: ParsedCommand | None = None
        self.busy = threading.Event()  # set while a behavior runs
        self.holding = threading.Event()  # set while the console is resetting the scene
        self._work = threading.Condition()
        self._pending = 0  # transcripts queued and not yet run or dropped, the running one included
        self._stops = 0  # stops heard so far

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
            with self._work:
                self._stops += 1
                self.ctx.stop_event.set()
            self.ctx.say("Stopping.")
            self._emit("stop", text=text)
            return
        if self.is_noise(text):
            logger.info("Ignoring %r (noise/hallucination gate).", text)
            self._emit("ignored", text=text, reason="noise")
            return
        with self._work:
            self._pending += 1
            generation = self._stops
        self.transcripts.put((generation, text))
        self._emit("queued", text=text)
        if self.busy.is_set():
            # A new command switches behaviors: stop the running one gracefully;
            # run() then dispatches the newest queued command.
            logger.info("New command while %s runs - stopping it to switch.",
                        self.current.intent.value if self.current else "a behavior")
            self.ctx.stop_event.set()

    # ------------------------------------------------------------- dispatch
    def dispatch(self, text: str, generation: int | None = None) -> ParsedCommand | None:
        """Parse and run one transcript on the calling thread.

        A stop heard while the LLM parses cancels the command it returns. The
        check and the stop-event clear stay under the lock stops are counted under.
        """
        with self._work:
            stops = self._stops if generation is None else generation
            cancelled = self._stops != stops
        if cancelled:
            logger.info("Stop heard before %r was dispatched - command not parsed.", text)
            self._emit("cancelled", text=text)
            return None
        try:
            cmd = self.parse(text)
        except Exception:
            logger.exception("Intent parse failed for %r.", text)
            self.ctx.say("I couldn't understand that command because the language service is unavailable.")
            self._emit("parse_error", text=text)
            return None
        self._emit("parsed", text=text, intent=cmd.intent.value, params=dict(cmd.params or {}))
        if cmd.intent in (Intent.UNKNOWN, Intent.STOP):
            if cmd.intent is Intent.UNKNOWN:
                self.ctx.say("I didn't catch a command in that.")
            return cmd
        handler = self.registry.get(cmd.intent)
        if handler is None:
            self.ctx.say(f"No behavior registered for '{cmd.intent.value}'.")
            return cmd
        with self._work:
            cancelled = self._stops != stops
            if not cancelled:
                self.ctx.stop_event.clear()
        if cancelled:
            logger.info("Stop heard while %r was parsed - %s not run.", text, cmd.intent.value)
            self._emit("cancelled", text=text, intent=cmd.intent.value)
            return cmd
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
            logger.exception("Behavior %s crashed - holding position.", cmd.intent.value)
            self._emit("behavior_crash", intent=cmd.intent.value, error=repr(e))
        finally:
            stopped = self.ctx.stop_event.is_set()
            self.ctx.stop_event.clear()
            self.current = None
            self.busy.clear()
            logger.info("=== %s ended (%s) ===", cmd.intent.value, "stopped" if stopped else "ok" if ok else "failed")
            self._emit("behavior_end", intent=cmd.intent.value, ok=ok, stopped=stopped)
        return cmd

    def _take_all(self) -> list[tuple[int, str]]:
        """Every queued transcript, oldest first; a queued stop stays queued."""
        texts, stops = [], 0
        while True:
            try:
                item = self.transcripts.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                stops += 1
            else:
                texts.append(item)
        for _ in range(stops):
            self.transcripts.put(_STOP)
        return texts

    def _finished(self, count: int) -> None:
        if count:
            with self._work:
                self._pending -= count
                self._work.notify_all()

    def drain_to_newest(self) -> None:
        """Drop queued transcripts except the newest one."""
        texts = self._take_all()
        if texts:
            self.transcripts.put(texts[-1])
        self._finished(max(len(texts) - 1, 0))

    def discard(self) -> None:
        """Drop every queued transcript."""
        self._finished(len(self._take_all()))

    def wait_idle(self, timeout: float | None) -> bool:
        """True once no transcript is queued and no behavior runs; False when ``timeout`` passes first
        (``None``: no limit)."""
        with self._work:
            return self._work.wait_for(lambda: self._pending == 0, timeout)

    def stop(self) -> None:
        """Wake :meth:`run` so it sees its ``shutdown`` event (set that first)."""
        self.transcripts.put(_STOP)

    def run(self, shutdown: threading.Event) -> None:
        """Dispatch loop until ``shutdown`` is set and :meth:`stop` is called."""
        while not shutdown.is_set():
            item = self.transcripts.get()
            if item is _STOP:
                continue
            generation, text = item
            try:
                if not self.holding.is_set():
                    self.dispatch(text, generation)
            finally:
                self._finished(1)
            if self.holding.is_set():
                self.discard()
                continue
            self.drain_to_newest()
