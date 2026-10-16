# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop README's Instructor's corner, run as written.

The commands are read from ``workshop/README.md``, so a README edit that
breaks one fails here. ``diff -r project solution`` exits 1 whenever the
trees differ, so its line is checked against its comment instead: on the
tracked files (the README calls ``_artifacts/``, ``logs/`` and
``__pycache__/`` disposable), every difference lies inside a TODO block, and
the differing blocks are as many as the comment says.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from notebook_harness import PIPE_ROOT
from twin_ros_rig import join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.docs, pytest.mark.timeout(600)]

WORKSHOP = PIPE_ROOT / "workshop"
README = WORKSHOP / "README.md"
# common.todos' block: '# >>> TODO <id> ...' up to the matching '# <<< TODO <id>'
_BLOCK = re.compile(r"#\s*>>>\s*TODO\s+(?P<id>\d+\.\d+)\b.*?#\s*<<<\s*TODO\s+(?P=id)\b", re.DOTALL)
_CHANGE = re.compile(r"^(\d+)(?:,(\d+))?([acd])(\d+)(?:,(\d+))?$")


def _corner() -> list[str]:
    """The Instructor's corner commands, one per line (continuations joined)."""
    m = re.search(r"## Instructor's corner\s+```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert m, "workshop/README.md has no Instructor's corner bash block"
    return [line.strip() for line in m.group(1).replace("\\\n", " ").splitlines() if line.strip()]


def _env() -> dict:
    """This setup's environment, with the venv's python first on PATH (README: activate .venv)."""
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def test_instructor_commands_run():
    join_test_domain()
    commands = [c for c in _corner() if not c.startswith("diff ")]
    assert commands, "the Instructor's corner lists no command besides the diff"
    for command in commands:
        done = subprocess.run(command, shell=True, cwd=WORKSHOP, env=_env(), capture_output=True,
                              text=True, timeout=540, executable="/bin/bash")
        assert done.returncode == 0, (
            f"README command failed ({done.returncode}): {command}\n"
            f"--- stdout\n{done.stdout[-3000:]}\n--- stderr\n{done.stderr[-3000:]}"
        )


def _blocks(path: Path) -> list[tuple[str, int, int]]:
    """``(id, first line, last line)`` of each TODO block in ``path`` (1-based, marker lines)."""
    text = path.read_text(encoding="utf-8")
    return [
        (m.group("id"), text.count("\n", 0, m.start()) + 1, text.count("\n", 0, m.end()) + 1)
        for m in _BLOCK.finditer(text)
    ]


def _block_of(blocks, first: int, last: int, anchor: bool) -> str | None:
    """The block holding lines ``first..last`` strictly between its markers (an
    insertion or deletion ``anchor`` is the line it follows, so a marker may hold it)."""
    for todo_id, start, end in blocks:
        if (start <= first and last < end) if anchor else (start < first and last < end):
            return todo_id
    return None


def _tracked_copy(dest: Path) -> None:
    """The tracked files of ``workshop/project`` and ``workshop/solution``, as they are now."""
    listed = subprocess.run(["git", "ls-files", "-z", "project", "solution"], cwd=WORKSHOP,
                            capture_output=True, text=True, check=True).stdout.split("\0")
    for rel in filter(None, listed):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSHOP / rel, target)


def test_project_differs_from_solution_only_in_todo_bodies(tmp_path):
    diff = next((c for c in _corner() if c.startswith("diff ")), None)
    assert diff, "the Instructor's corner has no diff line"
    command, _, claim = diff.partition("#")
    m = re.search(r"exactly the (\d+) TODO bodies", claim)
    assert m, f"the diff line's comment no longer says how many TODO bodies differ: {claim!r}"
    want = int(m.group(1))

    _tracked_copy(tmp_path)
    done = subprocess.run(command.strip(), shell=True, cwd=tmp_path, capture_output=True, text=True,
                          executable="/bin/bash")
    assert done.returncode in (0, 1), f"{command.strip()} failed: {done.stderr}"

    problems, changed = [], set()
    left = right = None
    for line in done.stdout.splitlines():
        if line.startswith("Only in "):
            problems.append(line)
        elif line.startswith("diff "):
            _, _, a, b = line.split()
            left, right = _blocks(tmp_path / a), _blocks(tmp_path / b)
            where = b
        elif (c := _CHANGE.match(line)) and left is not None:
            l1, l2, kind, r1, r2 = c.groups()
            l1, r1 = int(l1), int(r1)
            l2, r2 = int(l2 or l1), int(r2 or r1)
            ids = (_block_of(left, l1, l2, anchor=kind == "a"), _block_of(right, r1, r2, anchor=kind == "d"))
            if None in ids or ids[0] != ids[1]:
                problems.append(f"{where}: {line} is outside a TODO body")
            else:
                changed.add(ids[0])
    assert not problems, "project/ and solution/ differ outside the TODO bodies:\n" + "\n".join(problems)
    project_ids = {todo_id for f in (tmp_path / "project").rglob("*.py") for todo_id, _, _ in _blocks(f)}
    assert len(project_ids) == want, f"the README says {want} TODOs; project/ has {len(project_ids)}"
    assert changed == project_ids, f"TODO bodies identical in project/ and solution/: {sorted(project_ids - changed)}"
