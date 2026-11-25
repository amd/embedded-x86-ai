# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``02_digital_twin.ipynb`` section 4, spoken to: several commands in a row, each one run.

The other notebook tests type their commands (``test_twin_notebook.py``); the
speech front end is spoken to without the notebook (``test_voice_live.py``).
Here the two meet, the way a person uses the notebook: the kernel's microphone
is a stand-in that plays, in real time, sentences of a synthetic voice
(:class:`notebook_harness.SpokenInput`). From there on everything is the
notebook's own: Silero, the utterance cut, the real Gemma on a llama-server
the kernel starts, the console's queue, the behaviors on the twin.

The kernel is this module's own. Its notebook runs through ``c0000014``, whose
``voice.listen()`` asks for one spoken command and gets one; the test then
runs the console cell and speaks to it. What is checked is what the notebook
and the twin report: the console's record of each command, the arm's joints,
the scene.

``cd workshop && ./run_notebooks.sh --test``
"""

from __future__ import annotations

import asyncio
import time

import pytest

from notebook_harness import SpokenInput, until
from test_twin_notebook import (  # noqa: F401  (observers: the module's fixture, used by name)
    LISTEN_SENTENCE,
    _arm_moved,
    _assert_dispatcher_dead,
    notebook_kernel,
    observers,
)
from twin_ros_rig import BODY_JOINTS

# The hang guard is test_twin_notebook.py's, for the same kernel start: measured here
# (2026-11-25), the kernel's start took 36 s and the five spoken sentences 30 s.
pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(120)]

# The commands spoken, in this order, and between the first two a question, which is no command.
# The sentences are ones the real Gemma took right on every clip of the synthetic voice
# (three clips each, 2026-11-24).
QUESTION = "what is the weather like today"
COMMANDS = ["wave", "pick_place", "release", "home"]


@pytest.fixture(scope="module")
def microphone(tmp_path_factory):
    return SpokenInput(tmp_path_factory.mktemp("spoken") / "microphone.pcm")


@pytest.fixture(scope="module")
def kernel(observers, microphone):  # noqa: F811
    yield from notebook_kernel(microphone)


def test_listen_printed_the_command_gemma_picked(kernel):
    """Cell c0000014's ``voice.listen()`` heard the sentence said to it and named its command."""
    said = kernel.watcher.text()
    assert "Gemma picked: pick_place (object: ball)" in said, (
        f"voice.listen() did not report the command of {LISTEN_SENTENCE!r}:\n{said[-1500:]}")


def test_spoken_commands_in_a_row_each_run(observers, kernel, microphone):
    """Wave, a question, pick up the ball, open your hand, go home: spoken one after the other,
    each command runs when its turn comes, and the question runs nothing."""
    from common.motion import REST_POSE

    probe, scene = observers
    seen = {}

    async def say(console, sentence: str, command: str) -> dict:
        """Say ``sentence``; the console's record of ``command``, made when its behavior has ended."""
        mark = console.watcher.mark()
        await asyncio.to_thread(microphone.say, sentence)
        return await until(lambda: console.watcher.spoken_record(command, since=mark),
                           f"the record of the spoken {sentence!r}", console.watcher)

    async def script(console):
        start = await asyncio.to_thread(probe.joints)
        waving = asyncio.create_task(
            until(lambda: _arm_moved(probe, start), "the arm to start waving", probe))
        seen["wave"] = await say(console, "give me a wave", "wave")
        await waving  # the arm left its pose: the wave was run, not only recorded
        # A question is speech and reaches Gemma, which finds no command in it. The command
        # after it must still run: the console's whole trace is checked at the end.
        await asyncio.to_thread(microphone.say, QUESTION)
        seen["pick"] = await say(console, "pick up the ball", "pick_place")
        seen["ball"] = await asyncio.to_thread(scene.reading, "sports_ball", after=time.time())
        seen["release"] = await say(console, "open your hand", "release")
        seen["dropped"] = await asyncio.to_thread(scene.reading, "sports_ball", after=time.time())
        seen["home"] = await say(console, "go home", "home")
        seen["joints"] = await asyncio.to_thread(probe.joints, after=time.time())

    namespace = kernel.run(script)

    assert seen["pick"]["delivered"]["object"] == "ball", seen["pick"]
    assert "Got the sports ball." in seen["pick"]["said"], seen["pick"]
    ball = seen["ball"]
    assert ball["grasped"] and ball["xyz"][2] > 0.08, f"the ball is not held up after the spoken pick: {ball}"
    assert "Released." in seen["release"]["said"], seen["release"]
    assert not seen["dropped"]["grasped"], f"the ball is still held after 'open your hand': {seen['dropped']}"
    joints = seen["joints"]
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, f"the arm is not at rest after the spoken 'go home': {off}"
    # Every command spoken was run, once, in the order spoken; the question ran nothing.
    ran = [seen[name]["delivered"]["command"] for name in ("wave", "pick", "release", "home")]
    assert ran == COMMANDS
    assert [record["text"] for record in namespace["trace"]] == [
        seen[name]["text"] for name in ("wave", "pick", "release", "home")], namespace["trace"]
    _assert_dispatcher_dead(namespace)
