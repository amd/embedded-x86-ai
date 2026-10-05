#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# =============================================================================
# install_kernel.sh - register the repo's .venv as a Jupyter kernel and
# bring up llama-server (the iGPU brain).
#
# Why this exists: running `jupyter notebook` from inside the venv is not
# enough. Jupyter (and VS Code, and JupyterLab) launch notebooks with whatever
# *kernelspec* the notebook's metadata names - here "vvla-workshop". If that
# exact kernel was never registered, you silently get a system Python without
# the Ryzen AI onnxruntime (VitisAI EP) or ROCm torch, and every model falls
# back to CPU/simulation. (The old default name here was "strix-vvla", which no
# notebook references - so VS Code found no kernel at all. Now fixed to match.)
#
# It ALSO bakes the accelerator + ROS 2 environment into the kernelspec's
# kernel.json (LD_LIBRARY_PATH for the VitisAI EP native libs, the NPU firmware
# xclbin, HSA_OVERRIDE_GFX_VERSION, and rclpy paths). run_notebooks.sh sources
# that env before launching Jupyter, but VS Code / JupyterLab launch the kernel
# directly and source nothing - so without baking, the same notebook opened in
# VS Code loses the NPU and rclpy. Baking makes every launcher behave the same.
#
# It also starts llama-server with the exact flags and environment the
# pipeline itself uses (vla_pipeline/llm/llama_intent.py), reading
# config/pipeline.yaml - so the notebooks' chat box and IntentRouter find a
# live model on the iGPU instead of falling back to stubs.
#
# Usage:
#   ./scripts/install_kernel.sh              # kernel + env bake + llama-server
#   ./scripts/install_kernel.sh --force      # reinstall the kernelspec too
#   ./scripts/install_kernel.sh --no-server  # skip starting llama-server
#   ./scripts/install_kernel.sh --no-vscode  # don't write .vscode/settings.json
#   WITH_ROS_ENV=0 ./scripts/install_kernel.sh   # don't bake ROS 2 into kernel
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PY="${VENV_DIR}/bin/python"
# These MUST match the kernelspec the notebooks request in their metadata
# (workshop/notebooks/**/*.ipynb -> name "vvla-workshop") and what
# workshop/run_notebooks.sh registers. If they don't match, opening a notebook
# directly in VS Code - which selects a kernel by the name in the notebook -
# finds nothing and you silently get no accelerator stack.
KERNEL_NAME="${KERNEL_NAME:-vvla-workshop}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (VVLA workshop)}"
# ROS 2 setup sourced (if present) so its rclpy/DDS env can be baked into the
# kernel too; mirrors workshop/run_notebooks.sh. Set WITH_ROS_ENV=0 to skip.
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
WITH_ROS_ENV="${WITH_ROS_ENV:-1}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. The venv must exist (bootstrap.sh creates it).
[ -x "$PY" ] || die ".venv not found at ${VENV_DIR} - run ./bootstrap.sh first
       (or set VENV_DIR=/path/to/venv and re-run this script)."

say "Using venv python: $PY"

# 2. Make sure ipykernel + ipywidgets live INSIDE the venv.
#    (uv-created venvs may lack pip; fall back to uv or ensurepip.)
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    say "pip missing in venv - bootstrapping it"
    "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || {
        command -v uv >/dev/null 2>&1 || die "neither pip nor uv available"
        say "installing with uv instead"
        uv pip install --python "$PY" 'ipykernel>=6.29,<7' ipywidgets
    }
fi
if "$PY" -m pip --version >/dev/null 2>&1; then
    # ipywidgets: presence is enough.
    "$PY" -c "import ipywidgets" 2>/dev/null || {
        say "installing ipywidgets into the venv"
        "$PY" -m pip install --quiet "ipywidgets"
    }
    # ipykernel MUST stay on the 6.x line. ipykernel 7.x removed
    # Kernel.do_one_iteration(), which jupyter_ui_poll.poll() calls to service
    # widget events during a busy cell loop - without it the notebooks' "stop"
    # buttons silently do nothing. A plain "is it importable" check can't catch
    # an already-installed 7.x, so check the major version and downgrade if
    # needed. 6.29 is fully compatible with JupyterLab 4 + ipywidgets 8.
    "$PY" -c 'import sys, ipykernel; sys.exit(0 if int(ipykernel.__version__.split(".")[0]) == 6 else 1)' 2>/dev/null || {
        say "pinning ipykernel>=6.29,<7 (7.x breaks the notebooks' stop buttons)"
        "$PY" -m pip install --quiet "ipykernel>=6.29,<7"
    }
fi

# 2b. protobuf <-> mediapipe compatibility. Newer protobuf (5.x+) removed
#     MessageFactory.GetPrototype, which mediapipe still calls - the symptom is
#     "'MessageFactory' object has no attribute 'GetPrototype'" in the vision
#     cells, silently dropping hand/pose tracking to the synthetic fallback.
#     Something (often a later pip install) bumps protobuf past mediapipe.
set +e
"$PY" - <<'PYEOF'
import sys
try:
    import mediapipe  # noqa: F401
    sys.exit(0)                       # imports fine
except Exception as e:
    msg = "%s: %s" % (type(e).__name__, e)
    sys.exit(2 if "GetPrototype" in msg else 1)   # 1 = not installed / other
PYEOF
MP_RC=$?
set -e
if [ "$MP_RC" -eq 2 ]; then
    say "mediapipe is broken by a too-new protobuf (GetPrototype removed in 5.x) - pinning protobuf 4.25.8"
    "$PY" -m pip install --quiet "protobuf==4.25.8"
    if "$PY" -c "import mediapipe" 2>/dev/null; then
        say "fixed: mediapipe imports cleanly again"
    else
        say "WARNING: mediapipe still fails after the pin - try: $PY -m pip install --upgrade mediapipe"
    fi
fi

# 3. Register the venv as a user kernelspec (idempotent; --force reinstalls).
if [[ " $* " == *" --force "* ]]; then
    "$PY" -m jupyter kernelspec uninstall -y "$KERNEL_NAME" >/dev/null 2>&1 || true
fi
say "Registering kernel '${KERNEL_NAME}' (${KERNEL_DISPLAY})"
"$PY" -m ipykernel install --user --name "$KERNEL_NAME" \
      --display-name "$KERNEL_DISPLAY"

# 3b. Bake the accelerator (and ROS 2) environment INTO the kernelspec.
#     run_notebooks.sh sources scripts/ryzen_ai_env.sh (+ ROS 2) *before* it
#     launches Jupyter, so its kernel inherits LD_LIBRARY_PATH, the VitisAI EP
#     native libs, XLNX_VART_FIRMWARE and HSA_OVERRIDE_GFX_VERSION. VS Code and
#     JupyterLab launch the kernel straight from kernel.json and DO NOT source
#     anything - so without this the very same notebook opened in VS Code gets a
#     kernel with no EP libs on the loader path (silent CPU fallback) and no
#     rclpy for the 02_ros2 notebooks. Writing the env into kernel.json makes
#     every launcher behave like run_notebooks.sh.
KERNEL_DIR="$("$PY" - "$KERNEL_NAME" <<'PYEOF'
import sys
from jupyter_client.kernelspec import KernelSpecManager
print(KernelSpecManager().get_kernel_spec(sys.argv[1]).resource_dir)
PYEOF
)"
KERNEL_JSON="${KERNEL_DIR}/kernel.json"
say "Baking runtime env into ${KERNEL_JSON}"

# Capture the env in a subshell that sources the same scripts run_notebooks.sh
# does, then emit only the vars that matter as JSON.
KERNEL_ENV_JSON="$(
    set +e +u
    export VIRTUAL_ENV="$VENV_DIR"
    if [ "$WITH_ROS_ENV" = "1" ] && [ -f "$ROS_SETUP" ]; then
        # shellcheck disable=SC1090
        source "$ROS_SETUP" >/dev/null 2>&1 || true
    fi
    if [ -f "${REPO_ROOT}/scripts/ryzen_ai_env.sh" ]; then
        # shellcheck disable=SC1090
        source "${REPO_ROOT}/scripts/ryzen_ai_env.sh" >/dev/null 2>&1 || true
    fi
    "$PY" - <<'PYEOF'
import json, os
# Only the variables that make the accelerator / ROS 2 stack resolve at the
# moment the kernel process starts (kernelspec "env" replaces, so keep it tight).
keys = [
    "LD_LIBRARY_PATH", "XLNX_VART_FIRMWARE", "HSA_OVERRIDE_GFX_VERSION",
    "VIRTUAL_ENV",
    # ROS 2 - present only when a setup.bash was actually sourced above.
    "AMENT_PREFIX_PATH", "AMENT_CURRENT_PREFIX", "COLCON_PREFIX_PATH",
    "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "PYTHONPATH",
    "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION",
    "ROS_LOCALHOST_ONLY", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
]
print(json.dumps({k: os.environ[k] for k in keys if os.environ.get(k)}))
PYEOF
)"

if [ -n "${KERNEL_ENV_JSON:-}" ]; then
    "$PY" - "$KERNEL_JSON" "$KERNEL_ENV_JSON" <<'PYEOF'
import json, sys
path, env = sys.argv[1], json.loads(sys.argv[2])
spec = json.load(open(path))
spec.setdefault("env", {}).update(env)
with open(path, "w") as f:
    json.dump(spec, f, indent=1)
print("    baked %d env vars: %s" % (len(env), ", ".join(sorted(env))))
PYEOF
else
    say "    (nothing to bake - ryzen_ai_env.sh / ROS produced no vars)"
fi

# 3c. Point VS Code's Python extension at the venv and stop it from filtering
#     out --user kernelspecs. The named+baked kernel above is what really lets
#     VS Code find the kernel; this is belt-and-suspenders. --no-vscode skips it.
if [[ " $* " != *" --no-vscode "* ]]; then
    VSCODE_DIR="${REPO_ROOT}/.vscode"
    mkdir -p "$VSCODE_DIR"
    "$PY" - "${VSCODE_DIR}/settings.json" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
want = {
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "jupyter.kernels.filter": [],
}
if os.path.exists(path):
    try:
        data = json.load(open(path))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except Exception as e:
        print("    left existing .vscode/settings.json untouched (not plain JSON: %s)" % e)
        print("    add these keys yourself: %s" % json.dumps(want))
        sys.exit(0)
else:
    data = {}
for k, v in want.items():
    data.setdefault(k, v)   # never clobber a key the user already set
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print("    wrote %s" % path)
PYEOF
fi

# 4. Bring up llama-server (the iGPU brain) unless asked not to.
#    Mirrors vla_pipeline/llm/llama_intent.py: same flags, same
#    HSA_OVERRIDE_GFX_VERSION handling, same log file.
LLM_STATUS="skipped (--no-server)"
if [[ " $* " != *" --no-server "* ]]; then
    say "Checking llama-server"
    # Read the llm: section of config/pipeline.yaml with the venv's python.
    LLM_ENV="$("$PY" - "$REPO_ROOT" <<'PYEOF'
import shlex, sys
from pathlib import Path
root = Path(sys.argv[1])
try:
    import yaml
    l = yaml.safe_load(open(root / "config" / "pipeline.yaml"))["llm"]
except Exception as e:
    print("# config parse failed: %s" % e, file=sys.stderr)
    sys.exit(1)
r = lambda p: str(p if Path(p).is_absolute() else root / p)
print("LLM_BIN=%s" % shlex.quote(r(l["server_bin"])))
print("LLM_MODEL=%s" % shlex.quote(r(l["model_gguf"])))
print("LLM_HOST=%s" % shlex.quote(str(l.get("host", "127.0.0.1"))))
print("LLM_PORT=%s" % shlex.quote(str(l.get("port", 8081))))
print("LLM_NGL=%s" % shlex.quote(str(l.get("n_gpu_layers", 99))))
print("LLM_CTX=%s" % shlex.quote(str(l.get("ctx_size", 2048))))
print("LLM_PAR=%s" % shlex.quote(str(l.get("parallel", 1))))
print("LLM_NOWARMUP=%s" % shlex.quote("1" if l.get("no_warmup") else ""))
print("LLM_GFX=%s" % shlex.quote(str(l.get("hsa_override_gfx_version") or "")))
PYEOF
)" || die "could not read llm: section of config/pipeline.yaml"
    eval "$LLM_ENV"

    llm_alive() {
        "$PY" - "$LLM_HOST" "$LLM_PORT" <<'PYEOF'
import sys, urllib.request
try:
    r = urllib.request.urlopen(
        "http://%s:%s/health" % (sys.argv[1], sys.argv[2]), timeout=1)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PYEOF
    }

    if llm_alive; then
        LLM_STATUS="already running at http://${LLM_HOST}:${LLM_PORT}"
        say "llama-server ${LLM_STATUS}"
    elif [ ! -x "$LLM_BIN" ]; then
        LLM_STATUS="NOT STARTED - ${LLM_BIN} missing (run ./bootstrap.sh, llama.cpp build step)"
        say "$LLM_STATUS"
    elif [ ! -f "$LLM_MODEL" ]; then
        LLM_STATUS="NOT STARTED - ${LLM_MODEL} missing (run ./bootstrap.sh, model download step)"
        say "$LLM_STATUS"
    else
        mkdir -p "${REPO_ROOT}/logs"
        LLM_LOG="${REPO_ROOT}/logs/llama-server.log"
        CMD=("$LLM_BIN" --model "$LLM_MODEL" --host "$LLM_HOST" --port "$LLM_PORT"
             --n-gpu-layers "$LLM_NGL" --ctx-size "$LLM_CTX" --parallel "$LLM_PAR")
        [ -n "$LLM_NOWARMUP" ] && CMD+=(--no-warmup)
        # Deterministic GPU init on Strix (see llama_intent.py): present the
        # iGPU as its nearest supported relative unless already overridden.
        if [ -n "$LLM_GFX" ] && [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
            export HSA_OVERRIDE_GFX_VERSION="$LLM_GFX"
        fi
        say "Starting llama-server on the iGPU (log: ${LLM_LOG})"
        nohup "${CMD[@]}" >"$LLM_LOG" 2>&1 &
        LLM_PID=$!
        disown "$LLM_PID" 2>/dev/null || true
        deadline=$((SECONDS + 120))
        while [ $SECONDS -lt $deadline ]; do
            if llm_alive; then break; fi
            if ! kill -0 "$LLM_PID" 2>/dev/null; then
                printf '\033[1;31mllama-server exited during startup. Last log lines:\033[0m\n'
                tail -n 20 "$LLM_LOG" || true
                die "see full log: ${LLM_LOG}"
            fi
            sleep 0.5
        done
        if llm_alive; then
            LLM_STATUS="running (pid ${LLM_PID}) at http://${LLM_HOST}:${LLM_PORT}"
            say "llama-server ready - ${LLM_STATUS}"
        else
            LLM_STATUS="NOT READY after 120 s - check ${LLM_LOG}"
            say "$LLM_STATUS"
        fi
    fi
fi

# 5. Sanity report: does THIS kernel actually see the accelerator stack?
#    Reflect the same env we baked into kernel.json so the report matches what
#    VS Code / JupyterLab will actually see (not a bare venv).
if [ -f "${REPO_ROOT}/scripts/ryzen_ai_env.sh" ]; then
    set +u +e
    VIRTUAL_ENV="$VENV_DIR" source "${REPO_ROOT}/scripts/ryzen_ai_env.sh" >/dev/null 2>&1 || true
    set -u -e
fi
say "Sanity check (what the '${KERNEL_NAME}' kernel will see):"
"$PY" - <<'PYEOF'
import shutil, sys
print("  python        : %s" % sys.executable)
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    npu = "VitisAIExecutionProvider" in provs
    print("  onnxruntime   : %s   providers: %s" % (ort.__version__, provs))
    if npu:
        print("  NPU (VitisAI) : YES - vision will run on the NPU")
    else:
        print("  NPU (VitisAI) : NO - Ryzen AI wheel not in this venv; vision "
              "falls back to CPU (see bootstrap.sh section 4c / RYZEN_AI_WHEELS)")
except ImportError:
    print("  onnxruntime   : NOT INSTALLED - vision falls back to CPU/simulation")
try:
    import torch
    ok = torch.cuda.is_available()
    print("  torch ROCm    : %s" % ("YES" if ok else
          "no (iGPU telemetry simulated; llama-server uses the iGPU separately)"))
except ImportError:
    print("  torch         : not installed (fine - only llama-server uses the iGPU)")
try:
    from google.protobuf import __version__ as pbv
    try:
        import mediapipe
        print("  protobuf      : %s   mediapipe: %s (imports OK)"
              % (pbv, getattr(mediapipe, "__version__", "?")))
    except Exception as e:
        hint = (" - protobuf too new for mediapipe; pin protobuf==4.25.8"
                if "GetPrototype" in str(e) else "")
        print("  protobuf      : %s   mediapipe: BROKEN (%s: %s)%s"
              % (pbv, type(e).__name__, e, hint))
except ImportError:
    print("  protobuf      : not installed (mediapipe hands/pose will be simulated)")
print("  xrt-smi       : %s" % (shutil.which("xrt-smi") or
      "not on PATH (NPU telemetry will be simulated in the resource view)"))
PYEOF

cat <<EOF

Done.
  kernel       : ${KERNEL_NAME} ("${KERNEL_DISPLAY}")
  kernel.json  : ${KERNEL_JSON}
  llama-server : ${LLM_STATUS}

In Jupyter (jupyter notebook / lab):
  1. Restart Jupyter:            cd ${REPO_ROOT}/workshop && jupyter notebook
  2. The notebooks auto-select "${KERNEL_DISPLAY}"
     (or Kernel -> Change Kernel -> "${KERNEL_DISPLAY}").

In VS Code:
  1. Install the "Python" and "Jupyter" extensions (Microsoft).
  2. Open this repo folder (${REPO_ROOT}) and reload the window:
     Command Palette -> "Developer: Reload Window" (so it re-scans kernels).
  3. Open a notebook - it should auto-pick "${KERNEL_DISPLAY}". If not, click
     "Select Kernel" (top right) -> "Jupyter Kernel..." -> "${KERNEL_DISPLAY}".
  Because the NPU/ROS env is baked into kernel.json, ws.detect_capabilities()
  shows "NPU (VitisAI execution provider) [on]" in VS Code too - not only when
  launched via run_notebooks.sh.

To stop the server later:  pkill -f llama-server
EOF
