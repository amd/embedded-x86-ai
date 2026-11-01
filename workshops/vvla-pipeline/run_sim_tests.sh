#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Twin and NPU tests. Notebooks are a different command:
#   cd workshop && ./run_notebooks.sh --test
#
#   ./run_sim_tests.sh                              # NPU load order, then the twin stacks
#   ./run_sim_tests.sh -k toothbrush                # args go to the twin stacks only
#   ./run_sim_tests.sh --sim-yolo-device cpu
#   ./run_sim_tests.sh --sim-video failed
#   ./run_sim_tests.sh --sim-eyes seg
#   PROJECT=project ./run_sim_tests.sh -k workshop  # students' workshop code
set -euo pipefail

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "run_sim_tests.sh${*:+ $*}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || die "no venv at ${REPO_ROOT}/.venv - run ./bootstrap.sh first"

say "Sourcing scripts/ryzen_ai_env.sh"
# shellcheck disable=SC1091
VIRTUAL_ENV="${REPO_ROOT}/.venv" source "${REPO_ROOT}/scripts/ryzen_ai_env.sh" \
  || die "ryzen_ai_env.sh failed"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
[[ -f "$ROS_SETUP" ]] || die "ROS 2 is required and is not at $ROS_SETUP - run ./bootstrap.sh"
say "Sourcing ROS 2 ($ROS_SETUP)"
set +u +e
# shellcheck disable=SC1090
source "$ROS_SETUP" || die "ROS 2 setup failed"
set -e -u

say "Installing so101_sim from this clone"
"$PY" -m pip install --no-deps --upgrade --force-reinstall -e "${REPO_ROOT}/so101_sim"
"$PY" -m so101_sim.assets || die "could not fetch the SO-101 meshes"

"$PY" -c 'import pytest' || "$PY" -m pip install pytest
"$PY" -c 'import pytest_timeout' || "$PY" -m pip install pytest-timeout

export VVLA_TEST_STRICT=1
RC=0
say "NPU load order"
"$PY" -m pytest tests/npu || RC=$?
say "Stacks on the twin"
"$PY" -m pytest tests/sim -m "sim and not docs" "$@" || RC=$?
exit "$RC"
