# Agent Mistakes

## 2026-11-24 — Researched and planned for the wrong workshop on an assumption

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Plan a faster voice control (first with d1-Omni-2B, then by porting the Gemma 4 latency fixes).
- **Mistake:** Researched the so101-smolvla workshop and presented a full plan for it, although the repository had been switched to branch `gemma4-e4b-it_voice_intent` and the user meant `workshops/vvla-pipeline`. The plan was also given as a long chat reply outside planning mode, with the open decisions at its end instead of asked as questions first.
- **Evidence:** The user's corrections: "why are you not in planning mode? do you have questions for me ???" and, to the question which workshop, "vvla-pipeline" and "look at vvla !!!!". `git reflog` showed the checkout to `gemma4-e4b-it_voice_intent` at 19:06, which I had seen and noted before continuing with so101.
- **Cause:** The session's start-up snapshot named branch so101-smolvla, and I kept to it after seeing that the branch had changed, on the reasoning that the session "started there".
- **Correction:** The so101 plan was dropped. The workshop and the scope were asked as questions, and the approved plan is for `workshops/vvla-pipeline` on the current branch.
- **Prevention:** When the checked-out branch differs from the session's snapshot, or two places hold the thing the request names, ask which one is meant before researching; for a request to "make a plan", enter planning mode and ask the open questions first.
- **Status:** corrected

## 2026-11-24 — The list answer made a typed bare "stop" a stop command; no test had asked the real model

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Port the Gemma 4 voice latency fixes to `workshops/vvla-pipeline` (commit `58fab42`), among them Gemma's answer as a JSON list instead of an object.
- **Mistake:** Changed the shape of Gemma's answer without checking, on the real model, the sentences the instructions say must give no command. With the list shape a typed "stop" and "stop it" are answered `stop`; with the object shape they were `unknown`. The workshop's rule is that only "stop robot" stops.
- **Evidence:** `~/.cache/vvla-d1-probe/gemma_old_vs_new.py`, 2026-11-24 (`/tmp/vvla-gemma-oldnew-1.log`): the same 28 typed sentences to the same server with the instructions and grammar of `1d231e1` and with the present ones: 24 right before, 22 now; the two that differ are "stop" and "stop it". ("halt", "freeze", "abort" and "home" were already answered as commands before.)
- **Cause:** The tests that hold the "only stop robot" rule use the keyword match and a stand-in server; the real-model checks I ran (the 15 self-test sentences, the live test) hold one sentence that must give no command, a question.
- **Correction:** None yet: whether to return to the object shape for typed sentences, reword the instructions, or accept it is put to the user.
- **Prevention:** When the instructions or the shape of a model's answer change, run the sentences that must give no command on the real model before and after, not only the sentences that are commands.
- **Status:** unresolved
- **Amendment, 2026-11-25 — status: corrected.** The user chose a fix on the Gemma branch. The "stop" line of `INTENT_SYSTEM_PROMPT` (`vla_pipeline/llm/intents.py`) now says the two words "stop robot" together are the command and the word "stop" alone is unknown; the list answer stays. Measured on the real model (`~/.cache/vvla-d1-probe/gemma_stop_tune.py`, `/tmp/vvla-gemma-stop-1.log`): 258 right of 284 typed sentences and spoken clips instead of 241, and "stop", "halt", "freeze", "stop it", "abort" and "stop talking" are `unknown` typed and spoken. `tests/sim/test_voice_live.py` now asks the real model those sentences. Not corrected, and older than this mistake: a typed "home" alone is still answered `home`; it is in the same file as an expected failure.

## 2026-11-24 — Reported scores for d1-Omni-2B on sentences that had become its own example phrases

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Tune the decision model's command question (`workshops/vvla-pipeline`, branch `d1-omni-2b-voice_intent`).
- **Mistake:** Scored a new question set on the sentences it was tuned on after writing several of those sentences into the answers' example phrases ("stop talking", "do not pick up the ball", "the weather is nice"), and reported the result ("typed must do nothing 13 of 14, spoken 60 of 70") as the model's performance.
- **Evidence:** `~/.cache/vvla-d1-probe/d1_tune3.py` with a held-out set, 2026-11-24 (`/tmp/vvla-d1-tune-3.log`): the same question set is right on 187 of 214 tuned-on items and on 163 of 204 held-out ones; "must do nothing" spoken falls from 55 of 70 to 46 of 70.
- **Cause:** The test sentences and the candidate example phrases were written in the same sitting from the same pool, with no set kept apart.
- **Correction:** A held-out set (sentences in no answer's phrases) is scored beside the tuned-on set, and both numbers are reported and written at `COMMAND_ANSWERS` in `vla_pipeline/llm/intents.py`.
- **Prevention:** Before tuning a prompt against sentences, set aside sentences that never enter the prompt, and report those beside the tuned-on score.
- **Status:** corrected

## 2026-11-25 — Resolved a merge conflict in a tracked file with an inline script

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Merge `gemma4-e4b-it_voice_intent` into `d1-omni-2b-voice_intent` and resolve the conflict in `workshops/vvla-pipeline/README.md`.
- **Mistake:** Rewrote the conflicting paragraph of the README with an inline Python script. The user's standing rule is that tracked files are edited through the Edit tool only, so that the old and the new text are shown.
- **Evidence:** The command of 2026-11-25 that read `README.md`, matched the conflict markers with a regular expression and wrote the file back; the rule is in the session's memory ("never edit files through inline scripts; user wants old/new diffs").
- **Cause:** The two sides of the conflict were long single-line paragraphs, and a script was the shorter way to keep one side and take one phrase from the other. The rule was not checked before the shortcut.
- **Correction:** The result was reported to the user with the phrase that came from each side: the Gemma branch's paragraph, with "answers them, once with the real Gemma and once with the real d1-Omni-2B" from the d1 branch. The file's content is as intended; the other conflict was resolved with the Edit tool.
- **Prevention:** Resolve conflict markers with the Edit tool, one marker block per edit, however long the lines are.
- **Status:** corrected

## 2026-11-25 — Named a cause for the hanging launch from one reproduction, and planned a fix on it

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Find why a `ros2 launch` of `workshops/vvla-pipeline` did not exit after SIGINT (`tests/sim/test_entry_points.py::test_the_twin_and_controllers_die_with_the_app`).
- **Mistake:** Told the user the launch hangs "when its output goes to a pipe whose reader is gone and the stop arrives as the controllers have just come up", after one hang in eight probe runs, and proposed a fix for that (a relay for the launch's output). I had not read ROS launch's code or asked the hung process where it was. I also added a 10 s kill to the wrapper and reported the result as stable while the hang still occurred.
- **Evidence:** `~/.cache/vvla-bringup-probe/mini_stop.py`: a launch with a dead output pipe, stopped at the moment a process of its own exits, exited in 16 of 16 tries. The hung launch's own state (`hang_hunt.py`): 28 events in its queue and its event task waiting on that queue, a lost wake-up. `launch/launch_service.py:381-385` leaves Python's SIGINT handler in place and swallows its `KeyboardInterrupt`. With that handler replaced: 0 hangs in 70 stops (1 in 14 before), and 0 of 12 against 12 of 12 for one SIGINT during the launch's start. The user's corrections: "sounds like there is still a race" and "analyze the code ... and find the rootcause".
- **Cause:** I took two things that were present in the one hang I had (a dead pipe, a stop close to a process exit) for its cause, and reached for more test runs instead of the code of the process that hung.
- **Correction:** `so101_sim/ros_launch.py` starts every launch with a SIGINT handler that does nothing in Python, so no `KeyboardInterrupt` is raised inside the launch's event loop; `tests/sim/test_ros_launch.py` holds it. The wrapper's 10 s kill stays as the guarantee behind it, and its text no longer names the pipe.
- **Prevention:** For a process that hangs, read where it waits (its code, its own state dump) before naming a cause; a cause is stated only after an experiment that removes it removes the fault, and one that should not matter leaves the fault.
- **Status:** corrected
