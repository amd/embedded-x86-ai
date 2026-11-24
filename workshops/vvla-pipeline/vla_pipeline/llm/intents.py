# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Command intents shared by the LLM parser, orchestrator, and behaviors.

This revision makes the command set **extensible**: a command is an
``Intent`` plus a parameter dict (``ParsedCommand``), and behaviors register
themselves on a :class:`CommandRegistry` instead of being hardcoded in the
orchestrator's dispatch. Adding a new voice command =

1. add the enum member + one line in :data:`INTENT_SYSTEM_PROMPT`,
2. add its name to ``config/intent.gbnf``'s ``cmd`` rule,
3. decorate a handler with ``@registry.register(Intent.MY_COMMAND)``.

``STOP`` is special: the orchestrator matches it with plain keywords on the
raw transcript (no LLM round trip) so the safety stop is as fast as possible,
and it interrupts any running behavior via the shared stop event.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Intent(str, Enum):
    """Recognized voice-command intents, plus STOP and UNKNOWN sentinels."""

    GESTURE_MIMIC = "gesture_mimic"
    FETCH_BLOCK = "fetch_block"
    PICK_PLACE = "pick_place"
    DANCE = "dance"
    GRIP = "grip"
    RELEASE = "release"
    WAVE = "wave"
    HOME = "home"
    STOP = "stop"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "Intent":
        """Parse value into an Intent, or UNKNOWN if it doesn't match a member."""
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ParsedCommand:
    """An intent plus its parameters (e.g. the object name for pick_place)."""

    intent: Intent
    params: dict[str, str] = field(default_factory=dict)
    transcript: str = ""

    @property
    def object_name(self) -> str | None:
        """Object name parameter, if any (used by pick_place)."""
        return self.params.get("object")


# The only spoken stop. A bare "stop" is crowd noise on a tradeshow floor.
# Matched as a phrase: a check for the word "stop" would also fire on
# "stop talking".
STOP_KEYWORDS = ("stop robot",)


# Gemma's answer is a JSON list of three strings, in this order (config/intent.gbnf fixes it).
# A list and not an object with named keys: fewer tokens before the call is complete (measured
# on the so101 workshop, 2026-11-23: 12 tokens to the call instead of 21). The words heard come
# last, so the call is complete, and acted on, while they are still being written.
FIELDS = ("command", "object", "heard")

# The values of the first two hold no quote and no backslash (the grammar's word rule). The
# comma after the object's closing quote is what completes the call.
_CALL_SO_FAR = re.compile(r'\s*\[\s*"([a-z_]*)"\s*,\s*"([a-z ]*)"\s*,')
_HEARD_SO_FAR = re.compile(r'\s*"([^"\\]*)')  # what follows the call


def call_from_partial(text: str) -> ParsedCommand | None:
    """The call (command, object) once the answer's text so far holds all of it; None before.

    The words heard are not in the result: they follow.
    """
    found = _CALL_SO_FAR.match(text or "")
    if found is None:
        return None
    command, obj = found.groups()
    params = {"object": obj.strip()} if obj.strip() else {}
    return ParsedCommand(Intent.from_string(command), params=params)


def heard_from_partial(text: str) -> str:
    """The words heard as far as the answer's text has them: all of them in a finished answer."""
    call = _CALL_SO_FAR.match(text or "")
    found = _HEARD_SO_FAR.match(text, call.end()) if call else None
    return found.group(1).strip() if found else ""


def command_from_json(text: str) -> ParsedCommand | None:
    """A command the speech front end already resolved, handed on as a JSON object, or None."""
    raw = (text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
        command = data["command"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    params: dict[str, str] = {}
    obj = str(data.get("object", "") or "").strip()
    if obj:
        params["object"] = obj
    heard = str(data.get("heard", "") or "").strip()
    return ParsedCommand(Intent.from_string(str(command)), params=params, transcript=heard)


_KEYWORD_ROUTES = [
    (r"\bstop robot\b", "stop", None),
    (r"\b(release|open|drop)\b|\blet go\b", "release", None),
    (r"\b(copy|mirror|follow|mimic)\b", "gesture_mimic", None),
    (r"\b(block|cube)\b", "fetch_block", None),
    (r"\b(pick|grab|bring|fetch|get)\b", "pick_place", "OBJECT"),
    (r"\b(dance|groove|party|moves)\b", "dance", None),
    (r"\b(hold|take)\b", "grip", None),
    (r"\b(wave|hi|hello|greet)\b", "wave", None),
    (r"\bgo home\b", "home", None),
]
_OBJECT_WORDS = (
    "ball", "bottle", "cup", "mug", "phone", "apple", "banana", "book",
    "toothbrush", "remote", "orange", "teddy",
)


def keyword_parse(text: str) -> ParsedCommand:
    """Keyword stand-in used when the language model is not available."""
    low = (text or "").lower()
    for pattern, command, obj_mode in _KEYWORD_ROUTES:
        if re.search(pattern, low):
            params = {}
            if obj_mode == "OBJECT":
                obj = next((word for word in _OBJECT_WORDS if word in low), "")
                if obj:
                    params["object"] = obj
            return ParsedCommand(Intent.from_string(command), params=params, transcript=text)
    return ParsedCommand(Intent.UNKNOWN, transcript=text)


class CommandRegistry:
    """Maps intents to behavior entry points; the orchestrator just dispatches.

    Handlers receive ``(ctx, cmd: ParsedCommand)`` where ``ctx`` is the
    orchestrator's :class:`~vla_pipeline.main.PipelineContext` (arm, cameras,
    models, shared state, stop_check).
    """

    def __init__(self) -> None:
        self._handlers: dict[Intent, Callable] = {}

    def register(self, intent: Intent):
        """Return a decorator that registers fn as the handler for intent."""

        def deco(fn: Callable) -> Callable:
            """Register fn as the handler for intent, raising if one is already registered."""
            if intent in self._handlers:
                raise ValueError(f"handler for {intent} already registered")
            self._handlers[intent] = fn
            return fn

        return deco

    def get(self, intent: Intent) -> Callable | None:
        """Return the handler registered for intent, or None."""
        return self._handlers.get(intent)

    def intents(self) -> list[Intent]:
        """Return the list of intents that have a registered handler."""
        return list(self._handlers)


# Few-shot system prompt for the intent LLM. Kept here so the parser and any
# eval scripts share one source of truth. The grammar (config/intent.gbnf)
# guarantees the output shape; this prompt steers the choice.
INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm on a crowded floor. Other people
are talking. Ignore background conversation, a second speaker, questions,
quotes, negations, and anything not addressed to the robot: those are
"unknown". Map a clear instruction to exactly one command and answer ONLY
with a JSON list ["<name>", "<object or empty string>", "<words>"]: the command, then "object", then "heard".
"object" is the thing to act on for pick_place (one or two plain words,
e.g. "ball", "water bottle"); for every other command use "".
"heard" is the words the person said to the robot, and nothing else.
Use "" for "heard" when you cannot make out the words.

Commands:
- "gesture_mimic": the user wants the arm to copy / mirror / follow their own
  arm and hand movements ("copy me", "mirror my hand", "follow my arm").
- "fetch_block": the user wants the block/cube from its usual spot brought to
  them ("get me the block", "bring the cube here").
- "pick_place": the user names an object to pick up / grab / hand over
  ("pick up the ball", "grab the bottle and give it to me"). Put the object
  name in "object".
- "dance": the user wants the robot to dance / groove / party.
- "grip": the user is handing the robot something to hold ("hold this",
  "take this pencil", "grab what I give you").
- "release": the user wants the jaws opened, here, so they can take the
  object or drop it ("release", "open", "drop it", "let go",
  "open your hand").
- "wave": the user wants a wave / greeting ("wave at me", "say hi").
- "home": the speaker says "go home". "home", "rest", and "reset" alone
  are "unknown".
- "stop": the speaker says "stop robot". "stop", "halt", "freeze",
  "stop it", "abort", and "stop talking" are "unknown".
- "unknown": anything else.
"""
