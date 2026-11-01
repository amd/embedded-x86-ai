# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The stop fast path and the LLM parse: the pipeline's dispatcher and the
workshop's ``handle_transcript``.

The stand-in LLM hears "stop" while it is still parsing, as the always-hot
listener delivers it on the rig.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim


def _dispatcher(parse):
    from vla_pipeline.dispatch import Dispatcher
    from vla_pipeline.llm.intents import Intent

    ran, said, events = [], [], []
    ctx = SimpleNamespace(say=said.append, stop_event=threading.Event(), arm=None)
    registry = {Intent.WAVE: lambda _ctx, cmd: ran.append(cmd.intent)}
    d = Dispatcher(ctx, parse, registry, is_stop=lambda text: text == "stop")
    d.listeners.append(lambda event, **_data: events.append(event))
    return d, ctx, ran, said, events


def test_a_stop_said_while_the_llm_parses_cancels_the_command():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    held = {}

    def parse(text):
        held["dispatcher"].on_transcript("stop")
        return ParsedCommand(Intent.WAVE, transcript=text)

    d, ctx, ran, said, events = _dispatcher(parse)
    held["dispatcher"] = d
    d.dispatch("wave at me")

    assert ran == [], "the command the stop overlapped ran anyway"
    assert "cancelled" in events, events
    assert ctx.stop_event.is_set(), "the stop event was cleared"
    assert "Stopping." in said


def test_a_stop_said_before_the_command_does_not_cancel_it():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    d, ctx, ran, _said, events = _dispatcher(lambda text: ParsedCommand(Intent.WAVE, transcript=text))
    d.on_transcript("stop")
    d.dispatch("wave at me")

    assert ran == [Intent.WAVE]
    assert "cancelled" not in events
    assert not ctx.stop_event.is_set()


def test_a_stop_discards_a_command_already_queued():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    d, _ctx, ran, _said, events = _dispatcher(
        lambda text: ParsedCommand(Intent.WAVE, transcript=text)
    )
    d.on_transcript("wave at me")
    d.on_transcript("stop")

    generation, text = d.transcripts.get_nowait()
    d.dispatch(text, generation)

    assert ran == [], "a command queued before stop ran anyway"
    assert "cancelled" in events, events


def test_a_parse_failure_does_not_kill_dispatch():
    d, _ctx, ran, said, events = _dispatcher(
        lambda _text: (_ for _ in ()).throw(RuntimeError("llama unavailable"))
    )

    assert d.dispatch("wave at me") is None
    assert ran == []
    assert "parse_error" in events
    assert any("language service" in message for message in said)


def _workshop():
    from common.intents import CommandRegistry, Intent
    from integration.dispatch import WorkshopContext

    ran = []
    registry = CommandRegistry()
    registry.add(Intent.WAVE, lambda _ctx, cmd: ran.append(cmd.intent))
    ctx = WorkshopContext(cfg={}, arm=None, registry=registry)
    ctx.say = lambda _message: None
    return ctx, ran


def test_the_workshop_drops_a_command_whose_parse_a_stop_overlapped():
    from common.intents import Intent, ParsedCommand
    from integration.dispatch import handle_transcript

    ctx, ran = _workshop()

    def parse(text):
        handle_transcript(ctx, "stop", parse)
        return ParsedCommand(Intent.WAVE, transcript=text)

    handle_transcript(ctx, "wave at me", parse)

    assert ran == [], "the command the stop overlapped ran anyway"
    assert ctx.stop_event.is_set(), "the stop event was cleared"


def test_a_stop_said_before_a_workshop_command_does_not_cancel_it():
    from common.intents import Intent, ParsedCommand
    from integration.dispatch import handle_transcript

    ctx, ran = _workshop()
    handle_transcript(ctx, "stop", None)
    handle_transcript(ctx, "wave at me", lambda text: ParsedCommand(Intent.WAVE, transcript=text))

    assert ran == [Intent.WAVE]
    assert not ctx.stop_event.is_set()
