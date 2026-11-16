#!/usr/bin/env python3
"""Functional and performance gate for labs on the public repo.

A lab opts in with labs/<name>/ci/manifest.yml:

    functional: python ci/functional.py
    performance: python ci/performance.py
    budget_ms: 80

functional must exit 0. For a vision demo that means the fixture's bounding
boxes match the golden file and the output image is within tolerance.

performance must exit 0 and print one line: PERF_MS=<milliseconds>
The run fails when that number is above budget_ms. The budget is for the
strix-halo runner only.

Workshops are not scanned. labs/reachymini-lab is grandfathered until it has
a headless fixture. Any other lab a pull request touches must have a manifest.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "labs"
GRANDFATHERED = {"reachymini-lab"}
PERF_RE = re.compile(r"^PERF_MS=([0-9]+(?:\.[0-9]+)?)\s*$", re.M)


def lab_dirs() -> list[Path]:
    if not LABS.is_dir():
        return []
    return sorted(
        p for p in LABS.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def changed_lab_names(base: str) -> set[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=ROOT,
        text=True,
    )
    names: set[str] = set()
    for line in out.splitlines():
        parts = Path(line).parts
        if len(parts) >= 2 and parts[0] == "labs":
            names.add(parts[1])
    return names


def load_manifest(lab: Path) -> dict | None:
    path = lab / "ci" / "manifest.yml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("functional", "performance", "budget_ms") if key not in data]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    return data


def select(mode: str, base: str) -> list[tuple[Path, dict]]:
    if mode == "pr":
        names = changed_lab_names(base)
    else:
        names = {p.name for p in lab_dirs()}

    chosen: list[tuple[Path, dict]] = []
    for name in sorted(names):
        lab = LABS / name
        if not lab.is_dir():
            continue
        manifest = load_manifest(lab)
        if manifest is not None:
            chosen.append((lab, manifest))
            continue
        if name in GRANDFATHERED:
            print(f"::warning::{name} has no ci/manifest.yml yet; hardware checks skipped")
            continue
        if mode == "pr":
            raise SystemExit(
                f"{name} changed and has no labs/{name}/ci/manifest.yml with "
                "functional, performance, and budget_ms"
            )
    return chosen


def run_command(lab: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=lab,
        text=True,
        capture_output=True,
        check=False,
    )


def execute(pairs: list[tuple[Path, dict]]) -> None:
    failed = False
    for lab, manifest in pairs:
        print(f"== {lab.name}: functional ==")
        proc = run_command(lab, str(manifest["functional"]))
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"::error::{lab.name} functional check failed")
            failed = True
            continue

        print(f"== {lab.name}: performance ==")
        proc = run_command(lab, str(manifest["performance"]))
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"::error::{lab.name} performance command failed")
            failed = True
            continue
        match = PERF_RE.search(proc.stdout)
        if match is None:
            print(f"::error::{lab.name} performance output must include PERF_MS=<number>")
            failed = True
            continue
        measured = float(match.group(1))
        budget = float(manifest["budget_ms"])
        print(f"{lab.name}: {measured:.1f} ms (budget {budget:.1f} ms)")
        if measured > budget:
            print(f"::error::{lab.name} {measured:.1f} ms exceeds budget {budget:.1f} ms")
            failed = True
    if failed:
        raise SystemExit(1)


def write_output(has_hardware: bool) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as handle:
        handle.write(f"has_hardware={'true' if has_hardware else 'false'}\n")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "list"
    if command not in {"list", "run", "all"}:
        raise SystemExit("usage: demo_checks.py list|run|all")
    base = os.environ.get("PR_BASE_SHA", "")
    mode = "all" if command == "all" or (command == "list" and not base) else "pr"
    pairs = select(mode, base)
    names = [lab.name for lab, _ in pairs]
    print("labs:", ", ".join(names) if names else "(none)")
    if command == "list":
        write_output(bool(names))
        return
    if not pairs:
        print("No lab manifests to run.")
        return
    execute(pairs)


if __name__ == "__main__":
    main()
