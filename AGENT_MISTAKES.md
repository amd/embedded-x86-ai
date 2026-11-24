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
