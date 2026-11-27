#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# The one entry point for automated digital-twin testing. Every run takes one
# machine-wide lock (flock, the kernel's file lock), so runs never overlap: a
# second invocation prints who holds the lock and waits in the kernel until
# that run exits, then starts. Inside a run the stages go one after another.
#
#   ./digital_twin_auto_testing.sh                    # sim, then notebooks
#   ./digital_twin_auto_testing.sh sim [args]         # ./run_sim_tests.sh [args]
#   ./digital_twin_auto_testing.sh notebooks [args]   # workshop/run_notebooks.sh --test [args]
#   ./digital_twin_auto_testing.sh pytest <args>      # one pytest run in the test environment,
#                                                     #   e.g. pytest tests/moveit/test_behaviors.py -k mimic
#   ./digital_twin_auto_testing.sh --no-wait ...      # exit 75 at once if another run holds the lock
#
# Each run writes logs/twin-tests/<run id>/: <stage>.log (also shown live),
# <stage>.exit with that stage's exit status, <stage>.aie.txt with how many
# NPU AI Engines were reserved and busy while that stage ran, and summary.txt.
# The script exits with the first failing stage's status, or 0.
#
# The tests use their own ROS domains (90-110, notebooks 93), not the domain of
# a twin you run yourself. The runner refuses to start while a workshop Jupyter
# kernel is alive: that kernel owns NPU sessions that tests must not disturb.
set -euo pipefail

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

WAIT=1
if [[ "${1:-}" == "--no-wait" ]]; then
  WAIT=0
  shift
fi
MODE="${1:-all}"
[[ $# -gt 0 ]] && shift
case "$MODE" in
  all|sim|notebooks) ;;
  pytest) [[ $# -gt 0 ]] || die "pytest needs arguments, e.g. pytest tests/moveit/test_behaviors.py" ;;
  -h|--help) sed -n '/^# The one entry point/,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
  *) die "unknown mode '$MODE' (all, sim, notebooks, pytest; see --help)" ;;
esac

LIVE_KERNELS="$(
  ps -eo pid=,args= |
    awk -v root="$REPO_ROOT" 'index($0, root "/.venv/bin/python") && index($0, "ipykernel_launcher")'
)"
if [[ -n "$LIVE_KERNELS" ]]; then
  printf 'ERROR: not starting twin tests while this workshop Jupyter kernel owns the NPU:\n%s\n' \
    "$LIVE_KERNELS" >&2
  exit 75
fi

LOCK="${XDG_RUNTIME_DIR:-/tmp}/vvla-digital-twin-tests.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  say "Another twin test run holds ${LOCK}:"
  cat "${LOCK}.owner" 2>/dev/null || echo "  (no owner record)"
  [[ "$WAIT" -eq 1 ]] || { echo "--no-wait: not starting" >&2; exit 75; }
  say "Waiting for it to finish"
  flock 9
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="${REPO_ROOT}/logs/twin-tests/${RUN_ID}"
mkdir -p "$RUN_DIR"
printf '  run %s, pid %s, started %s\n  %s %s\n  logs %s\n' \
  "$RUN_ID" "$$" "$(date '+%F %T')" "$0" "$MODE${*:+ $*}" "$RUN_DIR" >"${LOCK}.owner"
AIE_PID=""
cleanup() {
  if [[ -n "${AIE_PID}" ]]; then
    exec 8>&- 2>/dev/null || true
    wait "$AIE_PID" 2>/dev/null || true
    AIE_PID=""
  fi
  rm -f "${LOCK}.owner"
}
trap cleanup EXIT
say "Run ${RUN_ID}: logs in ${RUN_DIR}"

# xrt-smi is sampled until this stage's pipe closes. Closing fd 8 is that
# event. The samples land in <stage>.aie.jsonl and the readable count in
# <stage>.aie.txt. A missing tool is written there and does not fail the stage.
start_aie() {
  local name="$1"
  AIE_PID=""
  local py="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "$py" ]]; then
    printf '%-10s %s\n' aie "no venv python, AI engines not measured" >>"${RUN_DIR}/summary.txt"
    return
  fi
  exec 8> >("$py" -m vla_pipeline.utils.aie_engines \
    --stop-on-stdin \
    --out "${RUN_DIR}/${name}.aie.jsonl" \
    --summary "${RUN_DIR}/${name}.aie.txt" 9>&-)
  AIE_PID=$!
}

stop_aie() {
  local name="$1"
  if [[ -z "${AIE_PID}" ]]; then
    return
  fi
  exec 8>&- 2>/dev/null || true
  wait "$AIE_PID" 2>/dev/null || true
  AIE_PID=""
  if [[ -f "${RUN_DIR}/${name}.aie.txt" ]]; then
    while IFS= read -r line; do
      printf '%-10s %s\n' aie "$line" >>"${RUN_DIR}/summary.txt"
    done <"${RUN_DIR}/${name}.aie.txt"
  else
    printf '%-10s %s\n' aie "no report for ${name}" >>"${RUN_DIR}/summary.txt"
  fi
}

RC=0
# A stage runs with the lock's descriptor closed, so a process it leaves behind
# cannot keep the lock after this run exits.
stage() {
  local name="$1"
  shift
  say "Stage ${name}: $*"
  start_aie "$name"
  local status
  set +e
  "$@" 9>&- 2>&1 | tee "${RUN_DIR}/${name}.log"
  status="${PIPESTATUS[0]}"
  set -e
  stop_aie "$name"
  echo "$status" >"${RUN_DIR}/${name}.exit"
  printf '%-10s exit %s\n' "$name" "$status" >>"${RUN_DIR}/summary.txt"
  say "Stage ${name} exited ${status}"
  [[ "$RC" -ne 0 || "$status" -eq 0 ]] || RC="$status"
}

run_pytest() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/test_env.sh"
  "$PY" -m pytest --rootdir "$REPO_ROOT" "$@"
}

case "$MODE" in
  all)
    stage sim ./run_sim_tests.sh
    stage notebooks ./workshop/run_notebooks.sh --test
    ;;
  sim) stage sim ./run_sim_tests.sh "$@" ;;
  notebooks) stage notebooks ./workshop/run_notebooks.sh --test "$@" ;;
  pytest) stage pytest run_pytest "$@" ;;
esac

say "Run ${RUN_ID} done:"
cat "${RUN_DIR}/summary.txt"
exit "$RC"
