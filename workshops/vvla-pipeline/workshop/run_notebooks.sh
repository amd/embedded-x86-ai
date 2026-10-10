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
#   3. Registers the "vvla-workshop" Jupyter kernel if it doesn't exist
#      (the notebooks select it automatically).
#   4. Sources ROS 2 (if installed) so the notebook kernel gets real
#      rclpy + DDS. The ROS 2 notebooks need this; they tell you how to
#      install ROS 2 if it's missing.
#   5. Starts the llama-server on the iGPU (if the binary and the GGUF
#      model are present and nothing is already listening on the port).
#      It logs to workshop/logs/llama-server.log and shuts down when you
#      quit Jupyter.
#   6. Opens the always-on-top resource HUD (CPU% / GPU% / NPU inf/s) so you
#      can watch the devices light up while a cell runs. It closes with Jupyter.
#   7. Launches Jupyter Lab in workshop/ (browse the notebooks/ folder).
#
# Usage:
#   ./run_notebooks.sh                 # set up + launch Jupyter Lab (+ llama server + HUD)
#   ./run_notebooks.sh --force-kernel  # re-register the kernel
#   ./run_notebooks.sh --no-launch     # set everything up, don't start Jupyter
#   ./run_notebooks.sh --classic       # use `jupyter notebook` instead of lab
#   ./run_notebooks.sh --no-llm        # don't start the llama server
#   ./run_notebooks.sh --no-monitor    # don't open the resource HUD
#
# Env overrides:
#   KERNEL_NAME=vvla-workshop     kernelspec name to register/select
#   JUPYTER_PORT=8888             port to serve on
# =============================================================================
set -euo pipefail

WORKSHOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSHOP_DIR}/.." && pwd)"
KERNEL_NAME="${KERNEL_NAME:-vvla-workshop}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (VVLA workshop)}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

FORCE_KERNEL=0; NO_LAUNCH=0; LAUNCHER="lab"; WITH_LLM=1; WITH_MONITOR=1
for arg in "$@"; do
  case "$arg" in
    --force-kernel) FORCE_KERNEL=1 ;;
    --no-launch)    NO_LAUNCH=1 ;;
    --classic)      LAUNCHER="notebook" ;;
    --no-llm)       WITH_LLM=0 ;;
    --no-monitor)   WITH_MONITOR=0 ;;
    -h|--help)      sed -n '2,/^# =====/p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

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

# Source ROS 2 (if installed) so the notebook kernel gets real rclpy + DDS.
# The ROS 2 notebooks require it; without it they stop and print install
# instructions instead of running.
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  say "Sourcing ROS 2 ($ROS_SETUP)"
  # ROS's setup.bash dereferences unbound vars (AMENT_TRACE_SETUP_FILES, ...) and
  # can return nonzero on benign conditions. Relax nounset/errexit across the
  # source, then restore our strict mode.
  set +u +e
  # shellcheck disable=SC1090
  source "$ROS_SETUP" || warn "ROS 2 source returned nonzero - continuing"
  set -e -u
else
  warn "ROS 2 not found at $ROS_SETUP - the 02_ros2 notebooks will not run."
  warn "Install ROS 2 Jazzy, then relaunch this script so the kernel sees rclpy."
fi

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
  "jupyter_ui_poll:jupyter-ui-poll>=0.2"
)
for entry in "${CORE[@]}"; do
  mod="${entry%%:*}"; spec="${entry#*:}"
  if "$PY" -c "import ${mod}" 2>/dev/null; then
    printf '    have %-13s\n' "$mod"
  else
    pip_install "$spec"
  fi
done

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

# --- 3. Register the kernel if it doesn't already exist ---------------------
_KERNEL_CHECK='import sys
try:
    from jupyter_client.kernelspec import KernelSpecManager
    sys.exit(0 if sys.argv[1] in KernelSpecManager().find_kernel_specs() else 1)
except Exception:
    sys.exit(1)'
kernel_exists() { "$PY" -c "$_KERNEL_CHECK" "$KERNEL_NAME" 2>/dev/null; }

if [[ "$FORCE_KERNEL" -eq 1 ]]; then
  "$PY" -m jupyter kernelspec uninstall -y "$KERNEL_NAME" >/dev/null 2>&1 || true
fi
if [[ "$FORCE_KERNEL" -eq 0 ]] && kernel_exists; then
  say "Kernel '${KERNEL_NAME}' already registered"
else
  say "Registering kernel '${KERNEL_NAME}' (${KERNEL_DISPLAY})"
  "$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"
fi

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

# --- 5. Start the llama server (component 1's brain, torn down on exit) -----
LLAMA_BIN="${REPO_ROOT}/third_party/llama.cpp/build/bin/llama-server"
LLAMA_GGUF="${REPO_ROOT}/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8081}"
LLAMA_PID=""
MONITOR_PID=""
cleanup() {
  [[ -n "$LLAMA_PID" ]] && kill -INT "$LLAMA_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

llama_alive() {
  "$PY" - "$LLAMA_HOST" "$LLAMA_PORT" <<'PYEOF'
import sys, urllib.request
try:
    r = urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=1)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

if [[ "$WITH_LLM" -eq 1 ]]; then
  if llama_alive; then
    say "llama-server already running at http://${LLAMA_HOST}:${LLAMA_PORT} - using it"
  elif [[ -x "$LLAMA_BIN" && -f "$LLAMA_GGUF" ]]; then
    mkdir -p "${WORKSHOP_DIR}/logs"
    say "Starting llama-server on the iGPU (port ${LLAMA_PORT})"
    say "  log: ${WORKSHOP_DIR}/logs/llama-server.log"
    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}" \
      "$LLAMA_BIN" --model "$LLAMA_GGUF" \
      --host "$LLAMA_HOST" --port "$LLAMA_PORT" \
      --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --no-warmup \
      >>"${WORKSHOP_DIR}/logs/llama-server.log" 2>&1 &
    LLAMA_PID=$!
    say "  llama-server PID ${LLAMA_PID} (auto-stops when you quit Jupyter;"
    say "  first load takes 10-30 s while the model streams into iGPU memory)"
  else
    warn "llama-server binary or GGUF model not found - skipping."
    warn "  expected binary: ${LLAMA_BIN}"
    warn "  expected model:  ${LLAMA_GGUF}"
    warn "  (run the repo bootstrap to build/download them; the 03_igpu_llama"
    warn "   notebook shows the exact command to start the server yourself)"
  fi
fi

# --- 6. Pop the always-on-top resource HUD (CPU/GPU/NPU), torn down on exit --
# Only when we're really launching Jupyter and a display is available. It reuses
# launch_monitor.sh, logs to workshop/logs/monitor.log, and the cleanup trap
# kills it (alongside the llama server) when Jupyter exits.
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

say "Launching Jupyter ${LAUNCHER} - the workshop kernel is auto-selected"
cd "$WORKSHOP_DIR"
# Not exec'd: keep this shell alive so the trap tears the llama server down
# after Jupyter exits.
"$PY" -m jupyter "$LAUNCHER" notebooks --port "$JUPYTER_PORT"
