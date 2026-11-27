# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Self-test reporting (PROVIDED).

Both harnesses (``models/selftest.py`` and ``ros/selftest.py``) funnel their
checks through this module so every run prints the same per-TODO status
table with targeted hints, and exits nonzero only on real failures.

Statuses:
- ``PASS``  - your code ran and produced the expected result.
- ``FAIL``  - your code ran but did the wrong thing; the hint names the
  integration point to look at.
- ``TODO``  - the stub still raises NotImplementedError; go to its notebook.
- ``SKIP``  - this machine can't run the check (missing hardware/deps); the
  detail says what would unlock it.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

# The workshop's TODO map: id -> (where, what, notebook that teaches it).
TODO_INDEX = {
    "1.1": (
        "models/npu.py",
        "npu_available",
        "notebooks/01_ai_models/01_npu_yolo.ipynb",
    ),
    "1.2": (
        "models/npu.py",
        "build_npu_session",
        "notebooks/01_ai_models/01_npu_yolo.ipynb",
    ),
    "1.3": (
        "models/npu.py",
        "active_provider",
        "notebooks/01_ai_models/01_npu_yolo.ipynb",
    ),
    "1.4": (
        "models/hands.py",
        "create_hand_tracker",
        "notebooks/01_ai_models/02_cpu_mediapipe.ipynb",
    ),
    "1.5": (
        "models/llm.py",
        "llama_server_command",
        "notebooks/02_ros2/03_intent_node.ipynb",
    ),
    "1.6": (
        "models/llm.py",
        "parse_intent",
        "notebooks/02_ros2/03_intent_node.ipynb",
    ),
    "2.1": (
        "ros/camera_node.py",
        "CameraNode._setup_ros",
        "notebooks/02_ros2/01_cameras.ipynb",
    ),
    "2.2": (
        "ros/camera_node.py",
        "CameraNode._publish",
        "notebooks/02_ros2/01_cameras.ipynb",
    ),
    "2.3": (
        "ros/camera_client.py",
        "Ros2Camera._setup_ros",
        "notebooks/02_ros2/01_cameras.ipynb",
    ),
    "2.4": (
        "ros/arm_node.py",
        "So101ServerNode._setup_ros",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.5": (
        "ros/arm_node.py",
        "So101ServerNode._on_command",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.6": (
        "ros/arm_node.py",
        "So101ServerNode._publish_state",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.7": (
        "ros/arm_client.py",
        "Ros2ArmClient._setup_ros",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.8": (
        "ros/arm_client.py",
        "Ros2ArmClient.send_joints",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.9": (
        "ros/arm_client.py",
        "Ros2ArmClient._on_state",
        "notebooks/02_ros2/02_robot.ipynb",
    ),
    "2.10": (
        "ros/intent_node.py",
        "IntentNode._setup_ros",
        "notebooks/02_ros2/03_intent_node.ipynb",
    ),
    "2.11": (
        "ros/intent_node.py",
        "IntentNode._on_transcript",
        "notebooks/02_ros2/03_intent_node.ipynb",
    ),
    "3.1": (
        "integration/dispatch.py",
        "build_registry",
        "notebooks/03_integration/01_behavior.ipynb",
    ),
    "3.2": (
        "integration/dispatch.py",
        "handle_transcript",
        "notebooks/03_integration/01_behavior.ipynb",
    ),
    "3.3": (
        "integration/mimic.py",
        "mimic_tick",
        "notebooks/03_integration/01_behavior.ipynb",
    ),
    "3.4": (
        "integration/behaviors.py",
        "run_dance",
        "notebooks/03_integration/01_behavior.ipynb",
    ),
}


@dataclass
class CheckResult:
    """One check's outcome: which TODO it covers, its name, status, and detail."""

    todo: str  # TODO id ("1.2") or "-" for infrastructure checks
    name: str
    status: str  # PASS / FAIL / TODO / SKIP
    detail: str = ""


class Reporter:
    """Prints and accumulates PASS/FAIL/TODO/SKIP results for one self-test run."""

    def __init__(self, title: str):
        self.title = title
        self.results: list = []
        self.artifacts: list = []
        print(f"\n=== {title} ===")

    # ------------------------------------------------------------------

    def add(self, todo: str, name: str, status: str, detail: str = "") -> None:
        """Record one result and print its status line."""
        self.results.append(CheckResult(todo, name, status, detail))
        mark = {"PASS": "[ok]", "FAIL": "[x]", "TODO": "…", "SKIP": "-"}[status]
        line = f"  [{status:4s}] {mark} {name}"
        if todo != "-" and todo in TODO_INDEX:
            where, what, _nb = TODO_INDEX[todo]
            line += f"   (TODO {todo}: {what} in {where})"
        print(line)
        if detail:
            for row in detail.splitlines():
                print(f"          {row}")

    def artifact(self, path) -> None:
        """Record and print the path to a generated artifact (e.g. a saved image)."""
        self.artifacts.append(str(path))
        print(f"          artifact: {path}")

    # ------------------------------------------------------------------

    def run(self, todo: str, name: str, fn, hint: str = ""):
        """Run one check. Returns ``(ok, value)``.

        - NotImplementedError  → TODO (with the notebook to read)
        - AssertionError       → FAIL, the assertion message IS the feedback
        - other exceptions     → FAIL with hint + traceback tail
        """
        try:
            value = fn()
        except NotImplementedError as e:
            nb = TODO_INDEX.get(todo, ("", "", ""))[2]
            self.add(todo, name, "TODO", f"not implemented yet → open {nb}\n({e})")
            return False, None
        except AssertionError as e:
            self.add(todo, name, "FAIL", str(e) or "assertion failed")
            return False, None
        except Exception as e:
            tail = traceback.format_exc().strip().splitlines()[-1]
            detail = f"{tail}"
            if hint:
                detail += f"\nhint: {hint}"
            self.add(todo, name, "FAIL", detail)
            return False, None
        self.add(todo, name, "PASS")
        return True, value

    # ------------------------------------------------------------------

    def summary(self) -> int:
        """Print the PASS/FAIL/TODO/SKIP tally and the next TODOs; return the exit code."""
        counts = {s: 0 for s in ("PASS", "FAIL", "TODO", "SKIP")}
        for r in self.results:
            counts[r.status] += 1
        print(
            f"\n--- {self.title}: "
            + "  ".join(f"{k} {v}" for k, v in counts.items())
            + " ---"
        )
        todos = sorted(
            {r.todo for r in self.results if r.status == "TODO"},
            key=lambda s: [int(x) for x in s.split(".")] if s[0].isdigit() else [99],
        )
        if todos:
            print("Next up:")
            for t in todos:
                where, what, nb = TODO_INDEX[t]
                print(f"  TODO {t}: {what}  →  {where}   (read {nb})")
        if counts["FAIL"] == 0 and counts["TODO"] == 0 and counts["PASS"] > 0:
            print("ALL CHECKS PASSED [ok]")
        return 1 if counts["FAIL"] else 0
