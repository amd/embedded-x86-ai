# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The speech front end against the real Gemma 4: sentences in, commands out, as a user hears it.

A stand-in microphone plays sentences in a synthetic voice (espeak-ng), in real
time, into the real ``open_voice`` session: the offset removal, Silero, the
utterance cut, the level raise and the request to a ``llama-server`` this test
starts on a free port, on the integrated graphics. The sound is made as a quiet
built-in microphone delivers it: samples around +1028, speech at a tenth of
full scale.

Needs the Gemma 4 files, the Silero model and the built llama-server
(``scripts/ensure_voice.sh``), and ``libespeak-ng1``. A missing piece fails the
test; nothing is skipped.

What is asserted is what must hold: one request per sentence, each with its
command; the command delivered when Gemma has written the call, before the words
it heard; a question delivers nothing. The seconds from the end of a sentence to
its command are printed, not asserted.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from synthetic_voice import OFFSET, RATE, microphone_bytes, spoken
from vla_pipeline.llm.intents import Intent
from vla_pipeline.utils.config import load_config

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(180)]

LEAD_S = 1.5        # silence before the first sentence
GAP_S = 2.5         # silence after each sentence: room for Gemma's answer
# (sentence, the command it must give, its object). The last is a question: no command.
SENTENCES = [
    ("pick up the ball", Intent.PICK_PLACE, "ball"),
    ("fetch the cube", Intent.FETCH_BLOCK, ""),
    ("go home", Intent.HOME, ""),
    ("stop robot", Intent.STOP, ""),
    ("open your hand", Intent.RELEASE, ""),
    ("what is the weather like today", Intent.UNKNOWN, ""),
]


def test_spoken_sentences_give_their_commands_one_request_each_the_call_before_the_words(monkeypatch, tmp_path):
    parts, begins, ends = [np.zeros(int(LEAD_S * RATE))], [], []
    for sentence, _, _ in SENTENCES:
        begins.append(sum(len(part) for part in parts) / RATE)
        parts.append(spoken(sentence))
        ends.append(sum(len(part) for part in parts) / RATE)
        parts.append(np.zeros(int(GAP_S * RATE)))
    sound = microphone_bytes(np.concatenate(parts))
    played_s = len(sound) / 2 / RATE
    answered = threading.Event()  # set when the parser has answered one request per sentence

    class Microphone:
        """Stands in for sounddevice.RawInputStream: the sound above, a 20 ms frame every 20 ms of
        real time, as a device delivers it; then the offset alone. It fails the run when the sound
        has been over for 30 s and the parser still has not answered a request for every sentence."""

        began: float | None = None

        def __init__(self, **settings):
            assert settings["samplerate"] == RATE and settings["latency"] == "low"
            self.position = 0

        def __enter__(self):
            Microphone.began = time.monotonic()
            return self

        def __exit__(self, *_):
            return False

        def read(self, samples):
            self.position += samples
            if self.position / RATE > played_s + 30:
                raise RuntimeError("no answer for every sentence 30 s after the sound ended")
            rest = Microphone.began + self.position / RATE - time.monotonic()
            if rest > 0:
                time.sleep(rest)  # the device's own clock
            chunk = sound[(self.position - samples) * 2:self.position * 2]
            return chunk + np.full(samples - len(chunk) // 2, OFFSET, "<i2").tobytes(), False

    import sounddevice

    from vla_pipeline.llm.voice import open_voice

    monkeypatch.setattr(sounddevice, "RawInputStream", Microphone)
    cfg = load_config()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg["gemma4-e4b-it"]["port"] = sock.getsockname()[1]  # a server of this test's own
    cfg.setdefault("system", {})["log_dir"] = str(tmp_path)

    session = open_voice(cfg)
    try:
        assert not session.error, f"the voice front end is not ready ({session.error}); run scripts/ensure_voice.sh"
        events: list[tuple[str, float, object]] = []
        requests: list[tuple[float, object, float]] = []  # when it ended, Gemma's whole answer, seconds to the call

        def at() -> float:
            return time.monotonic() - Microphone.began

        ask = session.llm.parse_audio

        def recorded(wav, on_call=None):
            """The parser's own request, with its answer kept: what Gemma made of each utterance."""
            answer = ask(wav, on_call=on_call)
            requests.append((at(), answer, session.llm.last_call_s))
            if len(requests) == len(SENTENCES):
                answered.set()
            return answer

        monkeypatch.setattr(session.llm, "parse_audio", recorded)
        session.start(
            lambda text: events.append(("command", at(), json.loads(text))),
            stop_check=answered.is_set,
            on_heard=lambda words: events.append(("heard", at(), words)),
        )
        assert session.opened.wait(30) and not session.error, session.error
        session._thread.join(played_s + 60)  # the microphone thread ends when ``answered`` is set
        assert not session._thread.is_alive(), "the microphone thread did not end"
    finally:
        session.close()

    print()
    commands = [(when, data) for kind, when, data in events if kind == "command"]
    delivered = iter(commands)
    seconds = []
    for (sentence, intent, obj), end, (ended_at, answer, call_s) in zip(SENTENCES, ends, requests):
        line = f"{sentence!r:34} -> {answer.intent.value}({answer.object_name or ''}) heard {answer.transcript!r}"
        if intent is not Intent.UNKNOWN:
            when, _ = next(delivered)
            seconds.append(when - end)
            line += (f"; command {when - end:.2f} s after the last word "
                     f"(request {call_s:.2f} s to the call, answer complete {ended_at - end:.2f} s after it)")
        print(line)
    seconds.sort()
    print(f"seconds from the last word to the command, min / median / max: "
          f"{seconds[0]:.2f} / {seconds[len(seconds) // 2]:.2f} / {seconds[-1]:.2f}; "
          f"server log: {tmp_path / 'llama-server.log'}")

    # One request per sentence, in the order spoken, each with its command and object: no half
    # sentence was sent, nothing was sent twice.
    assert len(requests) == len(SENTENCES), [answer for _, answer, _ in requests]
    assert [(answer.intent, answer.object_name or "") for _, answer, _ in requests] == [
        (intent, obj) for _, intent, obj in SENTENCES]
    # A command is delivered for each sentence that is one, once, in order; the question's answer
    # is `unknown` (asserted above from Gemma's own answer) and delivers nothing.
    expected = [(intent.value, obj) for _, intent, obj in SENTENCES if intent is not Intent.UNKNOWN]
    assert [(data["command"], data["object"]) for _, data in commands] == expected
    assert all(data["heard"] == "" for _, data in commands)  # delivered at the call: the words follow
    # After each command its words, once: the order of the events is command, heard, command, heard...
    assert [kind for kind, _, _ in events] == ["command", "heard"] * len(expected), events
    # Each command came after its own sentence had ended and before the next one began, and
    # before its request's answer was complete.
    command_sentences = [index for index, (_, intent, _) in enumerate(SENTENCES) if intent is not Intent.UNKNOWN]
    for index, (when, _) in zip(command_sentences, commands):
        following = begins[index + 1] if index + 1 < len(begins) else played_s
        assert ends[index] < when < following, (SENTENCES[index][0], when)
        assert when < requests[index][0]
    # Gemma wrote down the words of every command.
    assert all(words for kind, _, words in events if kind == "heard"), events


# ---- the stop rule, typed, on the real model -----------------------------------------------------
# "stop robot" is the only stop. The tests that hold this rule elsewhere use the keyword match
# and a stand-in server; when Gemma's answer became a list, the real model took a bare "stop"
# for a stop and no test saw it. These ask the real model.
STOP_ORDERS = ["stop robot", "Please STOP ROBOT now", "robot stop", "stop robot now"]
NOT_STOP_ORDERS = ["stop", "halt", "freeze", "stop it", "abort", "stop talking"]


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    """The real Gemma on a llama-server of this module's own; a missing piece fails, nothing is skipped."""
    from vla_pipeline.llm.llama_intent import LlamaIntentParser

    cfg = load_config()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg["gemma4-e4b-it"]["port"] = sock.getsockname()[1]
    cfg.setdefault("system", {})["log_dir"] = str(tmp_path_factory.mktemp("gemma-typed"))
    parser = LlamaIntentParser(cfg)  # FileNotFoundError names the missing file and scripts/ensure_voice.sh
    try:
        parser.warm_up()
        yield parser
    finally:
        parser.close()


def test_a_typed_stop_robot_stops_and_a_bare_stop_does_not(gemma):
    answers = [(text, Intent.STOP, gemma.parse(text).intent) for text in STOP_ORDERS]
    answers += [(text, Intent.UNKNOWN, gemma.parse(text).intent) for text in NOT_STOP_ORDERS]
    print()
    for text, expected, got in answers:
        print(f"{text!r:24} -> {got.value:8} should be {expected.value}")
    wrong = [(text, got.value, expected.value) for text, expected, got in answers if got is not expected]
    assert not wrong, wrong


@pytest.mark.xfail(strict=True, reason='the real Gemma answers a typed "home" alone as home, with every wording '
                                       'of the home line tried (2026-11-25); the instructions say it is unknown')
def test_a_typed_home_alone_is_no_command(gemma):
    assert gemma.parse("go home").intent is Intent.HOME
    assert gemma.parse("home").intent is Intent.UNKNOWN
