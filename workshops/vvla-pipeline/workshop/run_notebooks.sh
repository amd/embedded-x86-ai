#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# =============================================================================
# run_notebooks.sh - one command to open the workshop notebooks.
#
# What it does, in order:
#   1. Uses the repo venv created by ./bootstrap.sh (../.venv). That venv
#      carries the Ryzen AI (VitisAI EP) + ROCm stack and, being built with
#      --system-site-packages, sees ROS 2's rclpy. If it doesn't exist the
#      script stops and tells you to run bootstrap.sh first.
#   2. Installs ONLY the dependencies that are missing (never clobbers an
#      existing Ryzen AI onnxruntime; pins the versions the notebooks need,
#      notably mediapipe 0.10.21 and numpy<2).
#   3. Registers the "vvla-workshop" Jupyter kernel for this clone on every
#      launch (scripts/install_kernel.sh --register-only). That kernel.json is
#      shared by every clone on this machine, so a launch must rewrite it.
#   4. Sources ROS 2 (required) so the notebook kernel gets rclpy + DDS.
#   5. scripts/ensure_voice.sh (run above the Jupyter launch) checks the
#      Gemma 4 files and the built llama-server. A notebook that needs Gemma
#      starts the server itself.
#   6. Opens the always-on-top resource HUD (CPU% / GPU% / NPU inf/s) so you
#      can watch the devices light up while a cell runs. It closes with Jupyter.
#   7. Launches Jupyter Lab in workshop/ (browse the notebooks/ folder).
#
# Usage:
#   ./run_notebooks.sh                 # set up + launch Jupyter Lab (+ HUD)
#   ./run_notebooks.sh --force-kernel  # uninstall the kernel, then register it
#   ./run_notebooks.sh --no-launch     # set everything up, don't start Jupyter
#   ./run_notebooks.sh --classic       # use `jupyter notebook` instead of lab
#   ./run_notebooks.sh --no-monitor    # don't open the resource HUD
#   ./run_notebooks.sh --test [args]   # steps 1-4, then the notebooks instead of Jupyter
#                                      # (args go to those tests). The twin and the NPU
#                                      # tests are ../run_sim_tests.sh
#
# Env overrides:
#   KERNEL_NAME=vvla-workshop     kernelspec name to register/select
#   JUPYTER_PORT=8888             port to serve on
# =============================================================================
set -euo pipefail

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "run_notebooks.sh${*:+ $*}"

WORKSHOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSHOP_DIR}/.." && pwd)"
KERNEL_NAME="${KERNEL_NAME:-vvla-workshop}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (VVLA workshop)}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

FORCE_KERNEL=0; NO_LAUNCH=0; LAUNCHER="lab"; WITH_MONITOR=1; TEST_MODE=0
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-kernel) FORCE_KERNEL=1 ;;
    --no-launch)    NO_LAUNCH=1 ;;
    --classic)      LAUNCHER="notebook" ;;
    --no-monitor)   WITH_MONITOR=0 ;;
    --test)         TEST_MODE=1; NO_LAUNCH=1; WITH_MONITOR=0; shift; PYTEST_ARGS=("$@"); break ;;
    -h|--help)      sed -n '/^# run_notebooks.sh/,/^# =====/p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# --- 1. Use the repo venv created by bootstrap.sh ----------------------------
RYZEN_ENV_SCRIPT="${REPO_ROOT}/scripts/ryzen_ai_env.sh"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  die "no venv at ${VENV_DIR} - run ./bootstrap.sh from the repo root first.
       The workshop kernel uses that venv (Ryzen AI stack + rclpy)."
fi
say "Using the repo venv from bootstrap.sh: ${VENV_DIR}"

PY="${VENV_DIR}/bin/python"
say "Python: $("$PY" -c 'import sys; print(sys.version.split()[0], "@", sys.executable)')"

# Source the Ryzen AI env so the kernel inherits the VitisAI EP native libs
# and NPU firmware path - else vision silently falls back to CPU.
if [[ -f "$RYZEN_ENV_SCRIPT" ]]; then
  say "Sourcing scripts/ryzen_ai_env.sh (NPU EP libs + firmware for the kernel)"
  # shellcheck disable=SC1090
  VIRTUAL_ENV="$VENV_DIR" source "$RYZEN_ENV_SCRIPT" || warn "ryzen_ai_env.sh returned nonzero - continuing"
fi

# Source ROS 2 (required) so the notebook kernel gets rclpy + DDS.
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
[[ -f "$ROS_SETUP" ]] || die "ROS 2 is required and is not at $ROS_SETUP - run ./bootstrap.sh from the repo root"
say "Sourcing ROS 2 ($ROS_SETUP)"
# ROS's setup.bash dereferences unbound vars (AMENT_TRACE_SETUP_FILES, ...) and
# can return nonzero on benign conditions. Relax nounset/errexit across the
# source, then restore our strict mode.
set +u +e
# shellcheck disable=SC1090
source "$ROS_SETUP" || warn "ROS 2 source returned nonzero - continuing"
set -e -u
# Packages, the venv path, and the controller plugin. A pull does not re-run
# bootstrap, and without the plugin /joint_states never arrives.
VENV_DIR="$VENV_DIR" "${REPO_ROOT}/scripts/ensure_ros2_control.sh"
# The digital-twin notebook's microphone needs the voice models and llama-server.
VENV_DIR="$VENV_DIR" "${REPO_ROOT}/scripts/ensure_voice.sh"

# --- 2. Ensure pip, then install ONLY what's missing ------------------------
"$PY" -m pip --version >/dev/null 2>&1 || {
  say "Bootstrapping pip in the venv"
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || die "no pip and ensurepip failed"
}

pip_install() { say "pip install $*"; "$PY" -m pip install --quiet "$@"; }

# numpy first and pinned: onnxruntime + the AMD wheels are built against 1.x.
if ! "$PY" -c 'import numpy, sys; sys.exit(0 if numpy.__version__[0]=="1" else 1)' 2>/dev/null; then
  "$PY" -c 'import numpy' 2>/dev/null && warn "numpy 2.x present - pinning numpy<2 for ORT/AMD wheels"
  pip_install "numpy<2"
fi

# ONNX Runtime is deliberately absent from this list. The NPU distribution is
# named onnxruntime-vitisai but shares the same import tree as stock
# onnxruntime; installing the stock wheel here would corrupt the NPU runtime.
# bootstrap.sh installs exactly one ORT flavor.
CORE=(
  "yaml:pyyaml>=6.0"
  "requests:requests>=2.31"
  "cv2:opencv-python>=4.10"
  "onnx:onnx>=1.16,<1.18"
  "matplotlib:matplotlib>=3.7"
  "ipywidgets:ipywidgets>=8.1"
  "jupyter_ui_poll:jupyter-ui-poll>=1.1,<2"  # common/ui_poll.py builds on its 1.x internals
  "mujoco:mujoco>=3.1.3"
  "OpenGL:PyOpenGL>=3.1"
)
for entry in "${CORE[@]}"; do
  mod="${entry%%:*}"; spec="${entry#*:}"
  if "$PY" -c "import ${mod}" 2>/dev/null; then
    printf '    have %-13s\n' "$mod"
  else
    pip_install "$spec"
  fi
done

# The SO-101 digital twin (03_integration/02_digital_twin.ipynb). Always this
# clone: the venv has --system-site-packages, and so101-sim stays at 0.1.0, so
# an import check or a plain `pip install -e` leaves an older tree in place
# (no so101_sim.record). --no-deps keeps pip off the pinned numpy.
say "Installing so101_sim from this clone"
"$PY" -m pip install --quiet --no-deps --upgrade --force-reinstall -e "${REPO_ROOT}/so101_sim"
"$PY" -m so101_sim.assets || warn "could not fetch the SO-101 meshes - the digital twin will not load"
# A pull does not re-run bootstrap. Without this, the kernel cannot import vla_pipeline.
say "Installing vla_pipeline from this clone"
"$PY" -m pip install --quiet --no-deps --upgrade --force-reinstall -e "${REPO_ROOT}"

# ipykernel is handled separately (NOT in the presence-only loop above): it must
# stay on the 6.x line. ipykernel 7.x removed Kernel.do_one_iteration(), which
# jupyter_ui_poll.poll() calls to service widget events during a busy cell loop -
# without it the notebooks' "stop" buttons silently do nothing. A plain "is it
# importable" check can't catch an already-installed 7.x, so check the major
# version and downgrade if needed. 6.29 is fully compatible with JupyterLab 4 +
# ipywidgets 8.
if "$PY" -c 'import sys, ipykernel; sys.exit(0 if int(ipykernel.__version__.split(".")[0]) == 6 else 1)' 2>/dev/null; then
  printf '    have %-13s (6.x, ok)\n' "ipykernel"
else
  warn "ipykernel missing or >=7 (7.x breaks the notebooks' stop buttons) - pinning >=6.29,<7"
  pip_install "ipykernel>=6.29,<7"
fi

# mediapipe needs the legacy mp.solutions API (removed in 0.10.30+), pin 0.10.21;
# and it breaks under protobuf 5.x (GetPrototype removed).
MP_RC=0
"$PY" - <<'PYEOF' || MP_RC=$?
import sys
try:
    import mediapipe as mp
    mp.solutions.hands
    sys.exit(0)
except ImportError:
    sys.exit(1)
except AttributeError:
    sys.exit(3)
except Exception as e:
    sys.exit(4 if "GetPrototype" in f"{e}" else 5)
PYEOF
case "$MP_RC" in
  0) printf '    have %-13s\n' "mediapipe" ;;
  1) pip_install "mediapipe==0.10.21" ;;
  3) warn "mediapipe too new (no mp.solutions) - pinning 0.10.21"; pip_install "mediapipe==0.10.21" ;;
  4) warn "protobuf too new for mediapipe - pinning protobuf==4.25.8"; pip_install "protobuf==4.25.8" ;;
  *) warn "mediapipe import failed (rc=$MP_RC) - the hand-tracking cells will not run" ;;
esac

# The launcher itself.
if ! "$PY" -m jupyter --version >/dev/null 2>&1; then
  pip_install "jupyterlab>=4.0"
elif [[ "$LAUNCHER" == "lab" ]] && ! "$PY" -c 'import jupyterlab' 2>/dev/null; then
  pip_install "jupyterlab>=4.0"
fi

# --- 3. Register this clone's kernel on every launch ------------------------
# scripts/install_kernel.sh owns kernel.json. --register-only rewrites the
# shared spec from this tree and does not start the voice server or the NPU
# preflight, which the steps above already ran. ipykernel install resets the
# launcher command, so the bringup prefix is applied after that returns.
KERNEL_ARGS=(--register-only --no-vscode)
if [[ "$FORCE_KERNEL" -eq 1 ]]; then
  KERNEL_ARGS+=(--force)
fi
"${REPO_ROOT}/scripts/install_kernel.sh" "${KERNEL_ARGS[@]}"
# Both entry scripts run python -m so101_sim.bringup. This kernel does it
# before any cell, so a Vitis session cannot start first.
KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}/kernel.json"
"$PY" - "$KERNEL_JSON" <<'PY'
import json, sys
path = sys.argv[1]
spec = json.load(open(path))
argv = spec["argv"]
i = argv.index("-m")
rest = argv[i + 2:]
if rest[:1] == ["kernel"]:
    rest = rest[1:]
spec["argv"] = [*argv[: i + 1], "so101_sim.bringup", "kernel", *rest]
with open(path, "w") as fh:
    json.dump(spec, fh, indent=1)
    fh.write("\n")
PY

# --- 4. Quick sanity line: what will this kernel actually see? --------------
"$PY" - <<'PYEOF'
def _v(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "-"
try:
    import onnxruntime as ort
    npu = "VitisAIExecutionProvider" in ort.get_available_providers()
except Exception:
    npu = False
print("==> kernel sees: numpy %s | onnxruntime %s | mediapipe %s | NPU(VitisAI) %s"
      % (_v("numpy"), _v("onnxruntime"), _v("mediapipe"), "YES" if npu else "no (CPU fallback)"))
PYEOF

MONITOR_PID=""
LAUNCH_PIDS=()
cleanup() {
  # The same stop as run_pipeline.sh: SIGINT to each so101_sim.stack group, then wait.
  trap '' INT
  local pid
  for pid in "${LAUNCH_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${LAUNCH_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The digital twin. Same two Python launches as ./run_pipeline.sh --digital-twin,
# before the kernel's python -m so101_sim.bringup kernel.
start_digital_twin() {
  command -v ros2 >/dev/null || die "ROS 2 not found - install ROS 2 Jazzy (/opt/ros/jazzy)."
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  cd "$REPO_ROOT"
  python -m so101_sim.stack --arm false --cameras true --digital-twin true --dry-run false &
  LAUNCH_PIDS+=("$!")
  python -m so101_sim.stack --arm true --cameras false --digital-twin true --dry-run false &
  LAUNCH_PIDS+=("$!")
}

# --test: the notebooks, in this environment. Twin and NPU tests are ../run_sim_tests.sh.
# The docs tests start their own twin on ROS domain 93. This shell's
# ROS_DOMAIN_ID is often 93 too, so starting the launch here makes those
# tests see a twin that is already running.
if [[ "$TEST_MODE" -eq 1 ]]; then
  for entry in "pytest:pytest" "pytest_timeout:pytest-timeout" "nbclient:nbclient"; do
    "$PY" -c "import ${entry%%:*}" 2>/dev/null || pip_install "${entry#*:}"
  done
  cd "$REPO_ROOT"
  say "Notebooks"
  "$PY" -m pytest tests/sim -m docs "${PYTEST_ARGS[@]}" || exit $?
  exit 0
fi

# --- 6. Pop the always-on-top resource HUD (CPU/GPU/NPU), torn down on exit --
# Only when we're really launching Jupyter and a display is available. It reuses
# launch_monitor.sh, logs to workshop/logs/monitor.log, and the cleanup trap
# kills it when Jupyter exits.
if [[ "$WITH_MONITOR" -eq 1 && "$NO_LAUNCH" -eq 0 ]]; then
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    mkdir -p "${WORKSHOP_DIR}/logs"
    say "Opening the resource HUD (CPU/GPU/NPU; --no-monitor to skip)"
    "${WORKSHOP_DIR}/launch_monitor.sh" >>"${WORKSHOP_DIR}/logs/monitor.log" 2>&1 &
    MONITOR_PID=$!
    say "  resource HUD PID ${MONITOR_PID} (closes when you quit Jupyter)"
  else
    warn "No display (DISPLAY/WAYLAND_DISPLAY unset) - skipping the resource HUD."
  fi
fi

# --- 7. Launch Jupyter (or stop here with --no-launch) ----------------------
if [[ "$NO_LAUNCH" -eq 1 ]]; then
  say "Setup complete (--no-launch). Start it yourself with:"
  echo "    cd ${WORKSHOP_DIR} && ${PY} -m jupyter ${LAUNCHER} notebooks"
  exit 0
fi

# A live camera cell holds /dev/video0 in its kernel. Jupyter leaves that
# kernel running when the notebook tab is closed, so the next notebook cannot
# open the AMD ISP camera until this whole script is restarted. Shut the
# kernel down with the notebook instead.
LAB_OVERRIDES="${VENV_DIR}/share/jupyter/lab/settings/overrides.json"
mkdir -p "$(dirname "$LAB_OVERRIDES")"
"$PY" - "$LAB_OVERRIDES" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as fh:
        data = json.load(fh)
except FileNotFoundError:
    data = {}
data.setdefault("@jupyterlab/notebook-extension:tracker", {})["kernelShutdown"] = True
with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY

start_digital_twin
say "Launching Jupyter ${LAUNCHER} - the workshop kernel is auto-selected"
cd "$WORKSHOP_DIR"
# Not exec'd: keep this shell alive so the trap stops the digital twin
# after Jupyter exits.
"$PY" -m jupyter "$LAUNCHER" notebooks --port "$JUPYTER_PORT"
