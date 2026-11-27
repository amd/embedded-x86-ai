# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Stop and home phrases, the utterance cut that feeds Gemma, and its answer read as it is written.

No microphone and no model download. "stop robot" is the only stop phrase.
"go home" is the only home phrase. The real Gemma is asked in
``test_voice_live.py``; the stand-in server here proves the client's reading
of a stream only.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from vla_pipeline.audio.gemma_listen import FRAME_MS, OffsetRemover, Utterance, Utterances, at_level, dbfs
from vla_pipeline.llm.intents import (
    FIELDS,
    Intent,
    ParsedCommand,
    call_from_partial,
    command_from_json,
    heard_from_partial,
    keyword_parse,
)
from vla_pipeline.llm.llama_intent import LlamaIntentParser, answer_pieces
from vla_pipeline.llm.voice import VoiceSession
from vla_pipeline.main import is_stop
from vla_pipeline.utils.config import load_config

# llama.cpp's grammar lexer: a name is letters, digits, and '-'. '_' ends the
# name, and '"' ends a string even inside a character class. Both made
# llama-server reject intent.gbnf with "expecting newline or end".
_GRAMMAR_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

pytestmark = pytest.mark.sim


def test_stop_robot_is_the_only_stop_phrase():
    assert is_stop("stop robot")
    assert is_stop("Please STOP ROBOT now")
    for text in ("stop", "halt", "freeze", "stop it", "stop talking", "abort"):
        assert not is_stop(text), text


def test_stop_json_is_a_stop_without_a_second_parse():
    raw = '{"command": "stop", "object": ""}'
    assert is_stop(raw)
    cmd = command_from_json(raw)
    assert cmd is not None and cmd.intent.value == "stop"
    assert command_from_json("pick up the ball") is None


def test_heard_phrase_is_kept_with_the_command():
    raw = '{"command": "pick_place", "object": "football", "heard": "pick up the football"}'
    cmd = command_from_json(raw)
    assert cmd is not None
    assert cmd.intent.value == "pick_place"
    assert cmd.object_name == "football"
    assert cmd.transcript == "pick up the football"


def test_go_home_is_the_only_home_phrase():
    assert keyword_parse("go home").intent.value == "home"
    for text in ("home", "rest", "reset", "reset position"):
        assert keyword_parse(text).intent.value != "home", text


def test_keyword_pick_still_names_the_object():
    cmd = keyword_parse("pick up the ball")
    assert cmd.intent.value == "pick_place"
    assert cmd.object_name == "ball"


# ---- the microphone's frames cut into utterances -------------------------------------------------


def test_an_utterance_is_handed_over_on_the_frame_that_ends_it():
    frame = bytes(640)  # 20 ms at 16 kHz
    cut = Utterances(16000, max_s=1.0, end_silence_ms=100, pre_roll_ms=40, min_voice_ms=40)
    assert cut.end_silence_frames == 5 and cut.max_frames == 50 and cut.before.maxlen == 2
    clock = iter(range(10_000))  # each frame's capture time: its number, 20 ms apart

    def feed(pattern: str) -> list[Utterance | None]:
        """One frame per character, 'v' speech and '.' not; what each returned."""
        return [cut.feed(frame, c == "v", next(clock) * FRAME_MS / 1000) for c in pattern]

    def frames(utterance: Utterance) -> int:
        return len(utterance.pcm) // 640

    # Silence, then speech, then silence: handed over on the fifth silent frame, not a frame later,
    # with the two frames from before the speech: 2 + 4 + 5 frames.
    assert feed("....vvvv....") == [None] * 12 and cut.speaking
    (first,) = feed(".")
    assert frames(first) == 11 and not cut.speaking
    # Its capture times: the speech ended with frame 7, it was handed over with frame 12.
    assert first.spoke_until == pytest.approx(7 * 0.02) and first.ended == pytest.approx(12 * 0.02)
    # A pause shorter than the end silence stays inside the utterance. It begins right after the
    # one before it, whose last two frames are the sound in front of it: Gemma hears a sentence
    # badly that starts at a clip's first sample.
    assert feed("vv...vv....") == [None] * 11
    (second,) = feed(".")
    assert frames(second) == 2 + 2 + 3 + 2 + 5
    # Less speech than min_voice_ms (a click) is dropped.
    cut.clear()
    assert feed("v.....") == [None] * 6 and not cut.speaking
    # One that reaches the longest length is handed over as it is; the speech after it is the
    # next, again with the two frames before it.
    cut.clear()
    assert feed("v" * 49) == [None] * 49
    (longest,) = feed("v")
    assert frames(longest) == 50 and not cut.speaking
    *nothing, after = feed("vv.....")
    assert nothing == [None] * 6 and frames(after) == 2 + 2 + 5
    # After a break in the microphone's frames nothing of the old sound is kept.
    feed("vvv")
    cut.clear()
    assert not cut.speaking and not cut.before and feed(".....") == [None] * 5
    with pytest.raises(ValueError, match="Invalid PCM frame length"):
        cut.feed(bytes(10), True, 0.0)


def test_silence_does_not_hand_over_an_utterance():
    cut = Utterances(16000, max_s=6.0, end_silence_ms=300, pre_roll_ms=300, min_voice_ms=100)
    frame = bytes(640)
    assert all(cut.feed(frame, False, index * 0.02) is None for index in range(500))
    assert not cut.speaking


@pytest.mark.parametrize("rate", [16000, 48000])
def test_the_microphones_constant_offset_is_taken_out_and_the_speech_is_left(rate):
    step = rate * FRAME_MS // 1000
    time_s = np.arange(rate) / rate
    tone = 2000 * np.sin(2 * np.pi * 300 * time_s)  # in the range of speech
    samples = np.round(tone + 1028).astype("<i2")  # a built-in microphone: samples around +1028
    frames = [samples[i:i + step].tobytes() for i in range(0, rate, step)]
    assert -31 < dbfs(frames[10]) < -22  # with the offset in, the level is the offset's and the tone's together

    remover = OffsetRemover(rate)
    cleaned = np.concatenate([np.frombuffer(remover.feed(frame), "<i2") for frame in frames]).astype(float)
    assert abs(cleaned[rate // 2:].mean()) < 2  # the offset is gone
    assert abs(np.abs(cleaned[rate // 2:]).max() - 2000) < 30  # the tone is as loud as it was
    assert np.abs(cleaned[:step]).max() < 2100  # no thump at the start: it begins settled on the first sample

    # Silence with the offset reads as silence, not as -30 dBFS.
    quiet = OffsetRemover(rate)
    offset_only = np.full(step, 1028, "<i2").tobytes()
    assert dbfs(offset_only) > -31 and dbfs(quiet.feed(offset_only)) < -90
    # After a break in the frames it settles anew on the next first sample.
    quiet.reset()
    assert dbfs(quiet.feed(np.full(step, -500, "<i2").tobytes())) < -90


def test_an_utterance_is_sent_at_a_fixed_level_and_silence_is_not_blown_up():
    time_s = np.arange(16000) / 16000
    speech = np.round(300 * np.sin(2 * np.pi * 300 * time_s)).astype("<i2")  # quiet: about -40 dBFS
    loud = np.frombuffer(at_level(speech.tobytes(), peak=0.5, most_gain=100), "<i2")
    assert abs(int(np.abs(loud).max()) - 16384) < 200  # half of full scale
    # The gain has a ceiling: quiet speech is raised by most_gain and no further.
    capped = np.frombuffer(at_level(speech.tobytes(), peak=0.5, most_gain=20), "<i2")
    assert abs(int(np.abs(capped).max()) - 6000) < 20
    # One click does not set the scale: the speech still comes up, the click is clipped.
    clicked = speech.copy()
    clicked[100] = 32000
    out = np.frombuffer(at_level(clicked.tobytes(), peak=0.5, most_gain=100), "<i2")
    assert abs(int(np.abs(out[200:]).max()) - 16384) < 400 and out[100] == 32767
    # Sound that is already loud is brought down to the same level; nothing stays nothing.
    assert abs(int(np.abs(np.frombuffer(at_level((speech * 100).astype("<i2").tobytes(), 0.5, 20), "<i2")).max()) - 16384) < 200
    assert at_level(bytes(640), 0.5, 20) == bytes(640) and at_level(b"", 0.5, 20) == b""


def test_the_listening_settings_are_the_utterance_cuts():
    g = load_config()["gemma4-e4b-it"]
    for key in ("window_s", "end_silence_ms", "pre_roll_ms", "min_dbfs", "vad_threshold", "min_voice_ms",
                "sent_peak", "most_gain", "max_age_s"):
        assert key in g, key
    assert 0 < g["end_silence_ms"] < g["window_s"] * 1000 and 0 < g["sent_peak"] <= 1 and g["most_gain"] >= 1


def test_reset_button_is_the_orange_control_at_the_bottom():
    from vla_pipeline.console import RESET_BUTTON, TILE_H, TILE_W, VIEW_H, hits_reset_button, pointer_hits_reset

    x, y, w, h = RESET_BUTTON
    assert hits_reset_button(x, y)
    assert hits_reset_button(x + w - 1, y + h - 1)
    assert not hits_reset_button(x - 1, y)
    assert not hits_reset_button(x, y + h)
    image = (TILE_W * 3, TILE_H + VIEW_H)
    # Pointer already in image pixels.
    assert pointer_hits_reset(x + 4, y + 4, image, (0, 0, *image))
    # Pointer in a half-size window whose origin is not the image origin.
    assert pointer_hits_reset((x + 4) // 2, (y + 4) // 2, image, (80, 120, image[0] // 2, image[1] // 2))
    assert not pointer_hits_reset(10, 10, image, (80, 120, image[0] // 2, image[1] // 2))


# ---- the grammar ---------------------------------------------------------------------------------


def _llama_grammar_fault(grammar: str) -> str | None:
    """The fault llama.cpp reports, or None when this file would parse.

    Names stop at '_' (``heard_char`` is read as the rule ``heard``). A
    double quote inside ``[^"]`` ends the string instead of the class.
    """
    i, n = 0, len(grammar)
    while i < n:
        ch = grammar[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":
            while i < n and grammar[i] not in "\r\n":
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n and grammar[i] != '"':
                i += 2 if grammar[i] == "\\" and i + 1 < n else 1
            i += 1
            continue
        if ch == "[":
            i += 1
            if i < n and grammar[i] == "^":
                i += 1
            while i < n and grammar[i] != "]":
                if grammar[i] == '"':
                    return "quote inside a character class"
                i += 1
            i += 1
            continue
        if ch in _GRAMMAR_NAME:
            while i < n and grammar[i] in _GRAMMAR_NAME:
                i += 1
            if i < n and grammar[i] == "_":
                return "name stops before '_'"
            continue
        i += 1
    return None


def test_intent_grammar_is_one_llama_cpp_accepts():
    grammar = Path("config/intent.gbnf").read_text(encoding="utf-8")
    assert _llama_grammar_fault(grammar) is None
    # The answer is a list of the fields in their order: the call first, the words heard last.
    rules = [line for line in grammar.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert rules[0].split() == 'root ::= "[" ws cmd ws "," ws obj ws "," ws said ws "]"'.split()
    assert FIELDS == ("command", "object", "heard")
    # The grammar names every command, and no other.
    cmd_rule = grammar[grammar.index("cmd     ::="):grammar.index("# Object")]
    assert sorted(part.strip('\\"') for part in cmd_rule.split() if part.startswith('"\\"')) == sorted(
        intent.value for intent in Intent)
    # The two forms that made llama-server answer 400 for every spoken command.
    assert _llama_grammar_fault("heard_char ::= [a-z]\n") == "name stops before '_'"
    assert _llama_grammar_fault('hchar ::= [^"a]\n') == "quote inside a character class"


# ---- the answer read as it is written: the call before the words heard ---------------------------

ANSWER = '["pick_place", "water bottle", "Pick up the water bottle"]'
CALL_ENDS = ANSWER.index(', "Pick') + 1  # the comma after the object completes the call


def test_the_call_is_read_from_the_text_so_far_once_its_last_character_is_in():
    for length in range(CALL_ENDS):
        assert call_from_partial(ANSWER[:length]) is None, ANSWER[:length]
    for length in range(CALL_ENDS, len(ANSWER) + 1):
        call = call_from_partial(ANSWER[:length])
        assert call.intent is Intent.PICK_PLACE and call.params == {"object": "water bottle"}
        assert call.transcript == ""  # the words are not part of the call
    assert heard_from_partial(ANSWER[:CALL_ENDS]) == ""
    assert heard_from_partial(ANSWER[: ANSWER.index("Pick up the wa") + len("Pick up the wa")]) == "Pick up the wa"
    assert heard_from_partial(ANSWER) == "Pick up the water bottle"
    # An empty object stays out of the parameters; a command the model may not name is unknown.
    stop = call_from_partial('["stop", "", "st')
    assert stop.intent is Intent.STOP and stop.params == {}
    assert heard_from_partial('["stop", "", "st') == "st"
    assert call_from_partial('["fly", "",').intent is Intent.UNKNOWN
    assert call_from_partial("") is None and call_from_partial("not json") is None
    # Written without spaces, as a server may write it, it reads the same.
    tight = call_from_partial('["pick_place","water bottle","Pick')
    assert tight.params == {"object": "water bottle"} and heard_from_partial('["stop","","go') == "go"


def _events(*pieces: str) -> list[bytes]:
    lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() for piece in pieces]
    return [*lines, b"", b"data: [DONE]", b"data: " + json.dumps({"choices": [{"delta": {"content": "late"}}]}).encode()]


def test_the_servers_events_give_the_answers_pieces_up_to_done():
    assert list(answer_pieces(_events("[", '"stop"', ", "))) == ["[", '"stop"', ", "]
    with pytest.raises(RuntimeError, match="llama-server reported: no model"):
        list(answer_pieces([b'data: {"error": "no model"}']))


class StreamingStandIn:
    """A stand-in for llama-server's stream: it writes the call, then waits for ``go`` before it
    writes the words heard. Plumbing only; the real server is asked in ``test_voice_live.py``."""

    def __init__(self) -> None:
        self.go = threading.Event()
        self.released: list[bool] = []
        self.requests: list[dict] = []
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                stand_in.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self._send(ANSWER[:CALL_ENDS])
                    # Held until the client's on_call says it has the call (or the test's bound).
                    stand_in.released.append(stand_in.go.wait(10))
                    self._send(ANSWER[CALL_ENDS:])
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    stand_in.released.append(False)  # the client closed: it wanted the call only

            def _send(self, text):
                for character in text:
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": character}}]}).encode() + b"\n\n")
                self.wfile.flush()

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def close(self) -> None:
        self.go.set()
        self.server.shutdown()
        self.thread.join()


def _parser(port: int) -> LlamaIntentParser:
    cfg = load_config()
    cfg["gemma4-e4b-it"].update(host="127.0.0.1", port=port, autostart=False)
    return LlamaIntentParser(cfg)


def test_the_client_hands_over_the_call_while_the_words_are_still_unwritten():
    stand_in = StreamingStandIn()
    parser = _parser(stand_in.port)
    try:
        calls = []

        def on_call(call, seconds):
            calls.append((call, seconds))
            stand_in.go.set()  # only now may the server write the words heard

        command = parser.parse_audio(b"RIFF", on_call=on_call)
        # The server was released by on_call, not by its bound: the call came before the words.
        # (The library's default read buffer of 512 bytes would hold the call back here.)
        assert stand_in.released == [True]
        assert len(calls) == 1 and calls[0][0].transcript == ""
        assert (calls[0][0].intent, calls[0][0].params) == (command.intent, command.params)
        assert command.intent is Intent.PICK_PLACE and command.transcript == "Pick up the water bottle"
        assert calls[0][1] == parser.last_call_s and 0 < parser.last_call_s <= parser.last_total_s
        assert stand_in.requests[0]["stream"] is True and stand_in.requests[0]["grammar"] == parser.grammar

        # A caller that drops the call (on_call answers False) does not read the words. (``go``
        # stays set from here on: the server writes on at once, into a connection the client closed.)
        dropped = parser.parse_audio(b"RIFF", on_call=lambda call, seconds: False)
        assert dropped.intent is Intent.PICK_PLACE and dropped.transcript == ""

        # A typed sentence is its own words: the request ends at the call, the sentence stands in.
        typed = parser.parse("pick up the water bottle")
        assert typed.intent is Intent.PICK_PLACE and typed.params == {"object": "water bottle"}
        assert typed.transcript == "pick up the water bottle"
    finally:
        parser.close()
        stand_in.close()


def test_the_session_delivers_the_command_at_the_call_then_its_words_and_nothing_for_no_command(monkeypatch):
    """The request thread of the speech session, with a stand-in parser and a stand-in microphone
    that hands out three utterances' worth of frames and then ends."""
    answers = [
        (ParsedCommand(Intent.PICK_PLACE, {"object": "ball"}), "pick up the ball"),
        (ParsedCommand(Intent.UNKNOWN), "nice weather"),
        # The same command again is a sentence of its own: utterances do not overlap, nothing is dropped.
        (ParsedCommand(Intent.PICK_PLACE, {"object": "ball"}), "pick up the ball"),
        (ParsedCommand(Intent.STOP), ""),  # Gemma made out no words: the words event still comes, empty
    ]
    order, wanted = [], []

    class Parser:
        def parse_audio(self, wav, on_call=None):
            call, words = answers[len(wanted)]
            wanted.append(on_call(call, 0.25))  # False: the session has no use for the words
            order.append("words written")
            return ParsedCommand(call.intent, call.params, words)

        def close(self):
            pass

    rate, frame = 16000, bytes(640)
    speech = [(frame, True)] * 10 + [(frame, False)] * 20  # 200 ms of speech, then the end silence and more
    frames = iter([(frame, False)] * 5 + speech * len(answers))
    finished = threading.Event()

    class Microphone:
        def __init__(self, **settings):
            assert settings["latency"] == "low" and settings["samplerate"] == rate

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, samples):
            item = next(frames, None)
            if item is None:
                finished.set()
                return frame, False
            Detector.speech = item[1]
            return item[0], False

    class Detector:
        speech = False
        threshold = 0.5

        def __init__(self, *_):
            pass

        def feed(self, pcm, sample_rate):
            return 1.0 if Detector.speech else 0.0

        def reset(self):
            pass

    import sounddevice

    from vla_pipeline.llm import voice

    monkeypatch.setattr(sounddevice, "RawInputStream", Microphone)
    monkeypatch.setattr(voice, "_Silero", Detector)
    cfg = load_config()
    cfg["gemma4-e4b-it"]["min_dbfs"] = -200  # the stand-in frames are digital silence; the detector decides
    session = VoiceSession("gemma4-e4b-it", None, Parser())
    session._cfg, session._gemma_cfg = cfg, cfg["gemma4-e4b-it"]
    done = threading.Event()

    def on_heard(words):
        order.append(("heard", words))
        if len([item for item in order if item != "words written" and item[0] == "heard"]) == 3:
            done.set()  # the last delivered command's words: the request thread has answered every utterance

    session.start(lambda text: order.append(("command", json.loads(text))), stop_check=finished.is_set, on_heard=on_heard)
    assert done.wait(10), order
    session.stop()
    ball = {"command": "pick_place", "object": "ball", "heard": ""}
    assert order == [
        ("command", ball), "words written", ("heard", "pick up the ball"),
        "words written",
        ("command", ball), "words written", ("heard", "pick up the ball"),
        ("command", {"command": "stop", "object": "", "heard": ""}), "words written", ("heard", ""),
    ]
    assert wanted == [True, False, True, True]
