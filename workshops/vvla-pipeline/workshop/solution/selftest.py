# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""End-to-end workshop check (PROVIDED) - run: ``python selftest.py``

One command to answer the question you actually care about: *is the only thing
standing between me and the solution the TODOs I haven't written yet?*

It does three things, all without a robot, an NPU, a GPU, or a webcam. ROS 2
is required (``source /opt/ros/jazzy/setup.bash``): component 2 runs on it.

1. **TODO scan** - reads your source and lists which of the 21 TODOs are still
   stubs (the same checklist ``app.py`` shows on startup).
2. **The three component harnesses** - runs ``models.selftest``,
   ``ros.selftest`` and ``integration.selftest`` and tallies every
   PASS / FAIL / TODO / SKIP into one scoreboard. A FAIL means code you wrote
   does the wrong thing; a TODO means a stub you haven't filled in yet.
3. **Solution parity (the oracle)** - runs those same three harnesses against
   ``solution/`` and confirms they come back green there. That's the proof the
   harness itself is trustworthy: if the reference passes every check this
   machine can run, then "no FAILs on your tree" really does mean the only gap
   left is your remaining TODOs.

Verdict and exit code:
  - any FAIL            -> exit 1  ("fix this - it's not a missing TODO")
  - no FAIL, TODOs left -> exit 0  (or 1 with --strict; "here's what's left")
  - everything passes   -> exit 0  ("matches the solution end to end")

Flags:
  --strict       treat remaining TODOs as failure (handy for CI / "am I done?")
  --no-solution  skip the solution oracle (e.g. you deleted solution/)
  --quiet        show only the scoreboards, not each harness's full output
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import todos

HERE = Path(__file__).resolve().parent  # this tree (project/ or solution/)

# The three component harnesses, in workshop order.
_TALLY = re.compile(r"PASS\s+(\d+)\s+FAIL\s+(\d+)\s+TODO\s+(\d+)\s+SKIP\s+(\d+)")


_COMPONENTS = [
    ("Component 1  models       ", ["-m", "models.selftest"]),
    ("Component 2  ros          ", ["-m", "ros.selftest"]),
    ("Component 3  integration  ", ["-m", "integration.selftest"]),
]


def _run_harness(argv, cwd: Path, echo: bool):
    """Run one harness as a subprocess. Returns (returncode, pass, fail, todo, skip).

    Counts are parsed from its summary line; a crash (no summary) is surfaced
    as a synthetic FAIL so it can never be silently swallowed.
    """
    proc = subprocess.run(
        [sys.executable, *argv], cwd=str(cwd), capture_output=True, text=True
    )
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
    if echo:
        print(out.rstrip("\n"))
    matches = _TALLY.findall(out)
    if not matches:
        if echo:
            print("  (!) no summary line - the harness crashed before finishing")
        return proc.returncode, 0, 1, 0, 0  # count the crash as a FAIL
    p, f, t, s = (int(x) for x in matches[-1])
    return proc.returncode, p, f, t, s


def _run_suite(cwd: Path, *, echo: bool):
    """Run all three harnesses in ``cwd``; return the summed (pass, fail, todo, skip)."""
    tot = [0, 0, 0, 0]
    rows = []
    for label, argv in _COMPONENTS:
        if echo:
            print(f"\n{'=' * 74}\n>>> {label.strip()}   ({' '.join(argv)})\n{'=' * 74}")
        _, p, f, t, s = _run_harness(argv, cwd, echo)
        rows.append((label, p, f, t, s))
        for i, v in enumerate((p, f, t, s)):
            tot[i] += v
    return tot, rows


def _scoreboard(title: str, rows, tot) -> None:
    """Print a compact PASS/FAIL/TODO/SKIP table for a suite run."""
    print(f"\n{title}")
    print(f"  {'':26}{'PASS':>6}{'FAIL':>6}{'TODO':>6}{'SKIP':>6}")
    for label, p, f, t, s in rows:
        print(f"  {label:26}{p:>6}{f:>6}{t:>6}{s:>6}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':26}{tot[0]:>6}{tot[1]:>6}{tot[2]:>6}{tot[3]:>6}")


def main() -> int:
    """Scan TODOs, run the three harnesses, check the solution oracle, print a verdict."""
    ap = argparse.ArgumentParser(
        description="Run the whole workshop self-test suite end to end."
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any TODO is still unimplemented",
    )
    ap.add_argument(
        "--no-solution",
        action="store_true",
        help="skip the solution parity (oracle) check",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="show only the scoreboards, not each harness's full output",
    )
    args = ap.parse_args()
    echo = not args.quiet

    print("=" * 74)
    print("  VVLA workshop - end-to-end self test")
    print("  (ROS 2 required; no NPU / GPU / camera needed - those backends are synthetic)")
    print("=" * 74)

    # -- 1. static TODO scan ------------------------------------------------
    left = todos.remaining(HERE)
    print("\n-- TODO scan --")
    print("  " + todos.progress_line(HERE))
    if left:
        print(todos.format_list(left))
    else:
        print("  every TODO is implemented - nice.")

    # -- 2. the three component harnesses on YOUR tree ----------------------
    tot, rows = _run_suite(HERE, echo=echo)
    _scoreboard("-- your scoreboard --", rows, tot)
    your_fail, your_todo = tot[1], tot[2]

    # -- 3. solution parity (the oracle) ------------------------------------
    sol = HERE.parent / "solution"
    oracle_ok = None
    if not args.no_solution and HERE.name != "solution" and sol.is_dir():
        print(
            f"\n{'=' * 74}\n>>> solution parity (oracle): the same harnesses against "
            f"solution/\n{'=' * 74}"
        )
        sol_tot, sol_rows = _run_suite(sol, echo=False)
        _scoreboard("-- solution scoreboard --", sol_rows, sol_tot)
        oracle_ok = sol_tot[1] == 0 and sol_tot[2] == 0  # no FAIL, no TODO
        if oracle_ok:
            print(
                f"  the reference passes all {sol_tot[0]} runnable checks here "
                "-> the harness is trustworthy."
            )
        else:
            print(
                "  (!) the solution did NOT come back clean on this machine "
                f"(FAIL {sol_tot[1]}, TODO {sol_tot[2]})."
            )
            print(
                "      that points at the environment or the harness, not your "
                "code - investigate before trusting the verdict."
            )

    # -- verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    remaining_ids = todos.remaining(HERE)
    if your_fail:
        print(
            f"  X  {your_fail} check(s) FAIL. That is code that RAN but did the "
            "wrong thing -"
        )
        print(
            "     not a missing TODO. Read the FAIL hint above and fix it before "
            "moving on."
        )
        code = 1
    elif remaining_ids:
        print(
            f"  OK so far: 0 failures. What's left is exactly {len(remaining_ids)} "
            "TODO(s):"
        )
        print("     " + todos.compact(remaining_ids))
        print(
            "     Implement them (diff against solution/ when stuck) and re-run. "
            "Nothing"
        )
        print("     you've written so far conflicts with the solution's behavior.")
        code = 1 if args.strict else 0
    else:
        print("  DONE: every check passes and no TODOs remain - your project matches")
        print(
            "     the solution's behavior end to end. Take it to the rig:  python app.py"
        )
        code = 0
    if oracle_ok is False:
        code = code or 1
    print("=" * 74)
    return code


if __name__ == "__main__":
    sys.exit(main())
