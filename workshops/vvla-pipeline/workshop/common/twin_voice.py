# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The twin console's voice path (PROVIDED): what you say or type, one command at a time.

Section 5 of ``notebooks/03_integration/02_digital_twin.ipynb`` builds one
:class:`VoiceCommands`. While the console runs, every utterance, spoken or
typed, goes to :meth:`VoiceCommands.on_transcript`: "stop" acts at once,
noise is shown and dropped, anything else waits in a queue. :meth:`VoiceCommands.dispatch_loop`
runs the newest waiting command through the workshop's ``handle_transcript``
(``integration/dispatch.py``), with Llama on the iGPU picking the command.
:meth:`VoiceCommands.listen` takes one sentence before the console runs: it
prints what Whisper heard and the command Llama picks, and runs nothing.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

from common import speech
from common.fixtures import offline_parse_text
from common.imaging import resolve_class
from common.intents import CommandRegistry, Intent, ParsedCommand, is_noise, is_stop

RESET_LINE = "Reset. Arm parked, objects back on the table."
# How long the microphone may take to open before the console falls back to
# typing: it opened 0.18 s after the first start and 0.006 s after later ones
# (common.speech logs it); a missing device reports its error at once.
MIC_OPEN_S = 1.0

# True: run() also shows each command's record as JSON output, which the
# notebook tests read from the kernel (tests/sim/notebook_harness.py sets it).
REPORT_RECORDS = False


def keyword_parse(text: str) -> ParsedCommand:
    """The stand-in for Llama: match keywords in the text."""
    d = offline_parse_text(text)
    params = {"object": d["object"]} if d.get("object") else {}
    return ParsedCommand(Intent.from_string(d["command"]), params=params, transcript=text)


class VoiceCommands:
    """The microphone, Whisper, the intent parser and the queue of commands.

    ``ctx`` gets a registry whose behaviors report to ``console`` while they
    run. ``trace`` keeps one record per command: what was asked, what the
    robot said, and whether the named object ended up held, from ``twin``'s
    scene readings.

    One instance listens per process: a new one closes the previous one, so
    re-running the notebook cell does not leave a second microphone thread
    hearing every sentence.
    """

    _current: VoiceCommands | None = None

    def __init__(self, cfg: dict, ctx, console, twin):
        from integration.dispatch import build_registry, handle_transcript

        if VoiceCommands._current is not None:
            VoiceCommands._current.close()
        self.ctx, self.console, self.twin = ctx, console, twin
        self._handle = handle_transcript
        self._parse_intent, self.llm = self._brain(cfg)
        behaviors = build_registry()
        ctx.registry = CommandRegistry()
        for intent in behaviors.intents():
            ctx.registry.add(intent, console.track(intent.value, behaviors.get(intent)))
        self._work = threading.Condition()  # guards the two below; notified when either changes
        self._waiting: deque[str] = deque()  # commands not yet run, oldest first
        self._running = False  # a behavior runs
        self.holding = threading.Event()  # set while reset restores the scene
        self.trace: list[dict] = []
        self._listening: queue.Queue[str] | None = None  # set while listen() takes the sentences
        self._live = threading.Event()  # set while dispatch_loop runs: the console takes spoken commands
        # looked up on the module so a test can swap the microphone out
        whisper = speech.Whisper(cfg)
        self.mic = speech.MicListener(
            cfg, whisper.transcribe, self._utterance, on_level=console.on_level, on_drop=self._dropped
        ).start()
        console.transcribing = lambda: self.mic.transcribing
        if not self.mic.opened.wait(MIC_OPEN_S):
            self.mic.error = f"it did not open within {MIC_OPEN_S:.0f} s"
        if self.mic.error:
            console.voice = "typed"
            print(f"no microphone ({self.mic.error}) - type commands under the console")
        VoiceCommands._current = self

    @staticmethod
    def _brain(cfg: dict):
        """Llama on the iGPU (llama-server, started if needed), else the keyword match."""
        try:
            from models.llm import LlamaIntentParser

            llm = LlamaIntentParser(cfg)
            return llm.parse, llm
        except Exception as e:
            print(f"llama-server unavailable ({e}) - a keyword match picks the commands")
            return keyword_parse, None

    def parse(self, text: str) -> ParsedCommand:
        """Pick one command for ``text``, and show it in the console."""
        cmd = self._parse_intent(text)
        self.console.parsed(cmd)
        return cmd

    def _utterance(self, text: str) -> None:
        """A spoken sentence: to listen() while it waits, to the console while it runs, else dropped."""
        listening = self._listening
        if listening is not None:
            listening.put(text)
        elif self._live.is_set():
            self.on_transcript(text)

    def _dropped(self, reason: str) -> None:
        """Speech that never became a transcript: shown where a transcript would have gone."""
        if self._listening is not None:
            print(f"(not heard: {reason})")
        elif self._live.is_set():
            self.console.ignored(reason)

    def _explain(self, text: str) -> None:
        """Print the command the parser picks for ``text``, and how long it took."""
        t0 = time.perf_counter()
        cmd = self._parse_intent(text)
        ms = 1000 * (time.perf_counter() - t0)
        who = "Llama" if self.llm is not None else "The keyword match"
        if cmd.intent is Intent.UNKNOWN:
            print(f"{who} found no command in it ({ms:.0f} ms).")
            return
        params = ", ".join(f"{k}: {v}" for k, v in (cmd.params or {}).items() if v)
        print(f"{who} picked: {cmd.intent.value}" + (f" ({params})" if params else "") + f", in {ms:.0f} ms")

    def listen(self, seconds: float = 20.0, example: str = "could you pick up the ball for me") -> None:
        """Wait for one spoken command; print what Whisper heard and the command Llama picks.

        Nothing runs: the sentence does not reach the command queue. Without a
        microphone, ``example`` stands in for the spoken sentence.
        """
        if self.mic.error:
            print(f'Without a microphone, here is a typed sentence: "{example}"')
            self._explain(example)
            return
        print(f'Say a command, for example "pick up the ball". Listening for {seconds:.0f} s...')
        heard: queue.Queue[str] = queue.Queue()
        self._listening = heard
        try:
            deadline = time.monotonic() + seconds
            while (left := deadline - time.monotonic()) > 0:
                try:
                    text = heard.get(timeout=left)
                except queue.Empty:
                    break
                print(f'Whisper heard: "{text}"')
                if is_noise(text):
                    print("That is noise, so it is ignored. Say a command.")
                    continue
                if is_stop(text):
                    print("That is a stop: it acts at once, without asking Llama.")
                else:
                    self._explain(text)
                return
            print(f"Nothing heard in {seconds:.0f} s. Is the microphone muted?")
        finally:
            self._listening = None

    def on_transcript(self, text: str) -> None:
        """Every utterance, spoken or typed: a stop acts now; a new command replaces the running one."""
        if is_noise(text):  # a cough, keyboard clicks, silence: shown, never run
            self.console.ignored(text)
            return
        self.console.heard(text)
        if is_stop(text):
            self.console.stopped()
            self._handle(self.ctx, text, self.parse)  # ctx.request_stop(): the running behavior ends
            return
        with self._work:
            self._waiting.append(text)
            running = self._running
            self._work.notify_all()
        if running:
            self.ctx.stop_event.set()  # the running behavior ends; dispatch_loop runs the new command

    def run(self, text: str) -> dict:
        """Run one command through ``handle_transcript`` and record what it did to the named object."""
        named = offline_parse_text(text).get("object")
        cls = resolve_class(named) if named else None
        body = cls.replace(" ", "_") if cls else None
        before = self.twin.scene.reading(body) if body else None
        said, say = [], self.ctx.say
        self.ctx.say = lambda message: (said.append(message), say(message))
        try:
            self._handle(self.ctx, text, self.parse)
        finally:
            self.ctx.say = say
        after = self.twin.scene.reading(body, after=time.time()) if body else None
        record = {
            "text": text,
            "body": body,
            "said": said,
            "grasped": after["grasped"] if after else None,
            "z0": before["xyz"][2] if before else None,
            "z": after["xyz"][2] if after else None,
        }
        self.trace.append(record)
        if REPORT_RECORDS:
            from IPython.display import display

            display({"application/json": record}, raw=True)
        return record

    def drain(self) -> None:
        """Throw away every command still waiting."""
        with self._work:
            self._waiting.clear()

    def _notify_when(self, finished: threading.Event) -> None:
        """Wake dispatch_loop once ``finished`` is set (set by the console cell)."""
        finished.wait()
        with self._work:
            self._work.notify_all()

    def dispatch_loop(self, finished: threading.Event) -> None:
        """Run commands one at a time until ``finished``; a newer command replaces a waiting one.

        Spoken commands count only while this runs: speech before or after
        the console session is dropped, never queued.
        """
        self.drain()
        self._live.set()
        threading.Thread(target=self._notify_when, args=(finished,), daemon=True, name="session-end").start()
        try:
            while True:
                with self._work:
                    self._work.wait_for(lambda: self._waiting or finished.is_set())
                    if finished.is_set():
                        return
                    text = self._waiting[-1]  # only the newest command runs
                    self._waiting.clear()
                    if self.holding.is_set():
                        continue
                    self._running = True
                try:
                    self.run(text)
                except Exception as e:
                    self.console.say(f"(that failed: {e})")
                finally:
                    with self._work:
                        self._running = False
                        self._work.notify_all()
        finally:
            self._live.clear()

    def reset_scene(self, _=None) -> None:
        """Stop the behavior, park the arm with the jaws open, and put the objects back."""
        self.holding.set()
        self.ctx.request_stop()  # a command still being parsed does not start either
        self.drain()
        with self._work:  # the running behavior gets 3 s to stop
            self._work.wait_for(lambda: not self._running, 3.0)
        self.drain()
        try:
            self.ctx.shared_state.pop("grip_lock", None)
            self.ctx.shared_state.pop("grip_lock_pos", None)
            # the twin program parks the arm and puts the objects back (/sim/reset_scene)
            if self.twin.reset_scene():
                self.console.say(RESET_LINE)
            else:
                self.console.say("(reset failed: the twin program did not answer)")
        finally:
            self.ctx.stop_event.clear()
            self.holding.clear()

    def close(self) -> None:
        """Stop the microphone, and llama-server if this process started it."""
        if VoiceCommands._current is self:
            VoiceCommands._current = None
        self.mic.stop()
        if self.llm is not None:
            self.llm.close()
