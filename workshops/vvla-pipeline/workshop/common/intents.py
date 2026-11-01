# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Command intents, registry, and noise gate (PROVIDED).

These are the pipeline's *static assets*: the intent vocabulary, the system
prompt that steers the LLM, the stop keywords, and the noise-phrase filter.
They are provided because they would be copy-paste in any implementation -
the interesting work is *wiring them up* (see ``models/llm.py``,
``ros/intent_node.py``, and ``integration/dispatch.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Intent(str, Enum):
    """Voice-command vocabulary the pipeline recognizes."""

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
        """Parse a command string into an :class:`Intent`, defaulting to UNKNOWN."""
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ParsedCommand:
    """An intent plus its parameters (e.g. the object name for pick_place)."""

    intent: Intent
    params: dict = field(default_factory=dict)
    transcript: str = ""

    @property
    def object_name(self):
        """The object slot for pick_place-style commands, or None."""
        return self.params.get("object")


# Transcript keywords that trigger STOP without an LLM round trip. Matched
# case-insensitively as whole words on the raw transcript - the safety stop
# must never wait on a language model.
STOP_KEYWORDS = ("stop", "halt", "freeze", "abort", "emergency")

_STOP_RE = re.compile(r"\b(" + "|".join(STOP_KEYWORDS) + r")\b", re.IGNORECASE)


def is_stop(transcript: str) -> bool:
    """True when the raw transcript contains a stop keyword."""
    return bool(_STOP_RE.search(transcript or ""))


class CommandRegistry:
    """Maps intents to behavior entry points; the orchestrator just dispatches.

    Handlers receive ``(ctx, cmd: ParsedCommand)``. Registering handlers is
    *your* job (``integration/dispatch.py``); the registry mechanics are not.
    """

    def __init__(self) -> None:
        self._handlers: dict[Intent, Callable] = {}

    def register(self, intent: Intent):
        """Decorator form: register the wrapped handler for ``intent``."""

        def deco(fn: Callable) -> Callable:
            """Register ``fn`` as the handler for ``intent``, rejecting a duplicate."""
            if intent in self._handlers:
                raise ValueError(f"handler for {intent} already registered")
            self._handlers[intent] = fn
            return fn

        return deco

    def add(self, intent: Intent, fn: Callable) -> None:
        """Non-decorator registration (same rules as :meth:`register`)."""
        self.register(intent)(fn)

    def get(self, intent: Intent):
        """Return the handler registered for ``intent``, or None."""
        return self._handlers.get(intent)

    def intents(self) -> list:
        """List every intent with a registered handler."""
        return list(self._handlers)


# Few-shot system prompt for the intent LLM. The GBNF grammar guarantees the
# output *shape*; this prompt steers the *choice*.
INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm. Map the user's spoken request to
exactly one command and answer ONLY with a JSON object
{"command": "<name>", "object": "<object or empty string>"}.
"object" is the thing to act on for pick_place (one or two plain words,
e.g. "ball", "water bottle"); for every other command use "".

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
- "home": return to the rest/home position ("go home", "reset position",
  "rest now").
- "stop": stop moving immediately ("stop", "halt", "freeze").
- "unknown": anything else.
"""


# Whisper reliably hallucinates a few stock phrases on silence/noise; with a
# grammar forcing every transcript into a valid command, a phantom utterance
# would dispatch a real behavior. Drop these before they reach the LLM.
DEFAULT_NOISE_PHRASES = (
    "",
    "you",
    "thank you",
    "thanks",
    "bye",
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


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip surrounding whitespace and trailing punctuation."""
    return re.sub(r"[\s\.\,\!\?\"\'`]+$", "", (text or "").strip().lower()).strip()


# Whisper's non-speech annotations: [BLANK_AUDIO], (sighs), *music*.
_NON_SPEECH_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


def is_noise(text: str) -> bool:
    """True for known silence-hallucination phrases and empty transcripts."""
    norm = _normalize_transcript(_NON_SPEECH_TAG.sub(" ", text or ""))
    if not norm:
        return True
    if not any(c.isalpha() for c in norm):
        return True  # pure punctuation / digits
    return norm in {_normalize_transcript(p) for p in DEFAULT_NOISE_PHRASES}
