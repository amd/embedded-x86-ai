#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# =============================================================================
# bootstrap.sh - one-shot setup for the Ryzen AI VVLA workshop
#
# Target system: AMD Ryzen AI APU (arch auto-detected - e.g. Strix Point /
#                Radeon 890M = gfx1150, Strix Halo = gfx1151) · Ubuntu 24.04
#                ROCm 7.2.4 (installed if absent) · Ryzen AI SW 1.7.1 (XDNA2
#                NPU) · ROS 2 Jazzy (installed if absent)
#
# What it does:
#   1. apt build deps (skippable: --skip-apt)
#      1b. Installs ROCm ${ROCM_VERSION} from repo.radeon.com if /opt/rocm is absent
#      1c. Auto-detects the GPU arch (GFX_ARCH) via amdgpu-arch / rocminfo
#      1d. Installs ROS 2 Jazzy from packages.ros.org if /opt/ros/jazzy is absent
#   2. Installs uv (if missing) and creates a uv venv at ./.venv
#      (--system-site-packages so the ROS 2 Jazzy python stack stays importable)
#   3. Installs Python deps into the venv:
#        - PyTorch ROCm wheels for the DETECTED arch (AMD repo)
#        - LeRobot [feetech] from source, MediaPipe, Jupyter kernel deps, etc.
#        - Ryzen AI onnxruntime (VitisAI EP) from $RYZEN_AI_WHEELS - installed
#          LAST (section 4c) so nothing clobbers it; ROCm is never used for
#          onnxruntime (llama.cpp + torch only)
#   4. Builds llama.cpp with HIP/ROCm for the DETECTED arch into third_party/
#   5. Downloads the Llama-3.2-3B-Instruct GGUF Q4_K_M (hf download). The YOLO
#      ONNX export is a separate README step (scripts/export_yolo26s_*.py).
#   5b. Compiles the exported YOLO models for the NPU into cache/
#   6. Verifies the environment and prints next steps
#
# Flags:
#   --skip-apt      don't run apt (no sudo needed)
#   --skip-llama    don't clone/build llama.cpp
#   --skip-models   don't download/export models
#   --skip-compile  don't compile NPU models (yolo) even if SDK venv present
#   --cpu-only      no ROCm torch / no NPU wheels (dev machines; everything
#                   still runs with --device cpu and --dry-run)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
THIRD_PARTY="${REPO_ROOT}/third_party"
MODELS_DIR="${REPO_ROOT}/models"
LLAMA_DIR="${THIRD_PARTY}/llama.cpp"

# --- pinned versions / sources -----------------------------------------------
# ROCm release installed by section 1b if /opt/rocm is absent (override with
# ROCM_VERSION=...). The GPU arch (GFX_ARCH) is auto-detected in section 1c
# AFTER ROCm is present - do NOT hardcode it: Strix Point / Radeon 890M
# (Ryzen AI 9 HX 370/375) is gfx1150, Strix Halo (Ryzen AI Max) is gfx1151,
# and building llama.cpp for the wrong one loads but segfaults on the first
# kernel dispatch. Override detection with GFX_ARCH=... if ever needed.
ROCM_VERSION="${ROCM_VERSION:-7.2.4}"
UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-noble}")"
TORCH_SPEC='torch==2.11.0+rocm7.13.0'
TORCHVISION_SPEC='torchvision==0.26.0+rocm7.13.0'
TORCHAUDIO_SPEC='torchaudio==2.11.0+rocm7.13.0'
LLAMA_GGUF_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
LLAMA_GGUF_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_OUT_DIR="${MODELS_DIR}/llama-3.2-3b"
# Ryzen AI SW 1.7.1: directory containing (or whose subtree contains) the Linux
# onnxruntime-vitisai wheels. If unset, auto-detects ./ryzen_ai* in the repo root.
# Set before running, e.g.:  export RYZEN_AI_WHEELS=/opt/ryzen_ai-1.7.1/wheels
RYZEN_AI_WHEELS="${RYZEN_AI_WHEELS:-}"
if [[ -z "$RYZEN_AI_WHEELS" ]]; then
  for cand in "${REPO_ROOT}"/ryzen_ai*; do
    [[ -d "$cand" ]] && RYZEN_AI_WHEELS="$cand" && break
  done
fi
# Resolve to an absolute path - the script changes directories later.
if [[ -n "$RYZEN_AI_WHEELS" && -d "$RYZEN_AI_WHEELS" ]]; then
  RYZEN_AI_WHEELS="$(cd "$RYZEN_AI_WHEELS" && pwd)"
fi

SKIP_APT=0; SKIP_LLAMA=0; SKIP_MODELS=0; CPU_ONLY=0; SKIP_COMPILE=0
for arg in "$@"; do
  case "$arg" in
    --skip-apt)    SKIP_APT=1 ;;
    --skip-llama)  SKIP_LLAMA=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    --cpu-only)    CPU_ONLY=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# Full Ryzen AI SDK venv used to COMPILE NPU models (yolo pose/detect). The
# deployment .venv can only RUN precompiled models. This venv is installed at
# the repo root (./ryzenai-compile) by bootstrap's compile step, using the
# install_ryzen_ai.sh shipped in the SDK wheel directory (./ryzen_ai*).
RYZEN_AI_COMPILE_VENV="${RYZEN_AI_COMPILE_VENV:-${REPO_ROOT}/ryzenai-compile}"
# Locate the SDK wheel dir (contains install_ryzen_ai.sh) in the repo root.
RYZEN_AI_SDK_DIR="${RYZEN_AI_SDK_DIR:-}"
if [[ -z "$RYZEN_AI_SDK_DIR" ]]; then
  for cand in "${REPO_ROOT}"/ryzen_ai*; do
    [[ -f "${cand}/install_ryzen_ai.sh" ]] && RYZEN_AI_SDK_DIR="$cand" && break
  done
fi

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# Print the iGPU/dGPU gfx arch (e.g. gfx1150). Tries the LLVM helper, then
# rocminfo, then both again under sudo - a user freshly added to the
# render/video groups can't open the KFD nodes until re-login, but root can.
detect_gfx_arch() {
  local out
  if [[ -x /opt/rocm/llvm/bin/amdgpu-arch ]]; then
    out="$(/opt/rocm/llvm/bin/amdgpu-arch 2>/dev/null | head -n1)" && [[ -n "$out" ]] && { echo "$out"; return 0; }
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    out="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+')" && [[ -n "$out" ]] && { echo "$out"; return 0; }
  fi
  if [[ -x /opt/rocm/llvm/bin/amdgpu-arch ]]; then
    out="$(sudo /opt/rocm/llvm/bin/amdgpu-arch 2>/dev/null | head -n1)" && [[ -n "$out" ]] && { echo "$out"; return 0; }
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    out="$(sudo rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+')" && [[ -n "$out" ]] && { echo "$out"; return 0; }
  fi
  return 1
}

[[ "${EUID}" -eq 0 ]] && die "Run bootstrap.sh as your normal user (it sudo's only for apt)."

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
log "Environment checks"
if [[ "$CPU_ONLY" -eq 0 ]]; then
  if [[ ! -e /dev/accel/accel0 ]]; then
    warn "NPU device /dev/accel/accel0 not found - XDNA driver / Ryzen AI SW 1.7.1 may not be installed. NPU components will fall back to CPU."
  fi
  if [[ -z "$RYZEN_AI_WHEELS" ]]; then
    warn "RYZEN_AI_WHEELS not set - the VitisAI onnxruntime wheel will NOT be installed; stock onnxruntime (CPU EP) is used instead. Set RYZEN_AI_WHEELS to the Ryzen AI 1.7.1 wheel directory and rerun to enable the NPU."
  fi
fi
# (ROS 2 Jazzy presence is handled in section 1d - installed if absent.)

# -----------------------------------------------------------------------------
# 1. apt dependencies
# -----------------------------------------------------------------------------
if [[ "$SKIP_APT" -eq 0 ]]; then
  log "Installing apt dependencies (sudo)"
  sudo apt update
  # libgl1/libglib2.0-0 cover headless OpenCV; the rest are the X11/xcb libs the
  # GUI OpenCV wheel's bundled Qt plugin needs so cv2.imshow() can open a window
  # (yolo_pose_npu.py + the mediapipe demo). Without them you get the Qt "xcb
  # platform plugin could not be loaded" error even with a non-headless build.
  sudo apt install -y \
    git curl gnupg cmake build-essential pkg-config python3-dev \
    ffmpeg libportaudio2 portaudio19-dev v4l-utils \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 libxcb1 libxcb-xinerama0 \
    libxcb-cursor0 libxkbcommon-x11-0
else
  log "Skipping apt step (--skip-apt)"
fi

# -----------------------------------------------------------------------------
# 1b. ROCm (installed from repo.radeon.com if /opt/rocm is absent)
# -----------------------------------------------------------------------------
if [[ "$CPU_ONLY" -eq 0 && ! -d /opt/rocm ]]; then
  if [[ "$SKIP_APT" -eq 1 ]]; then
    warn "/opt/rocm missing but --skip-apt given - cannot install ROCm. GPU steps will be skipped; rerun without --skip-apt or install ROCm ${ROCM_VERSION} manually."
  else
    log "Installing ROCm ${ROCM_VERSION} (sudo; Ubuntu ${UBUNTU_CODENAME})"
    sudo mkdir -p --mode=0755 /etc/apt/keyrings
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null
    # ROCm 7.x serves the kernel/graphics driver packages under
    # graphics/<rocm-ver>/ubuntu. The old amdgpu/<rocm-ver>/ubuntu path 404s on
    # its Release file - that directory is now keyed by the Radeon Software
    # driver version (e.g. amdgpu/30.30/ubuntu), NOT the ROCm version. Both the
    # rocm and graphics repos go in one rocm.list, matching AMD's native-install
    # docs for Ubuntu 24.04.
    sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} ${UBUNTU_CODENAME} main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${ROCM_VERSION}/ubuntu ${UBUNTU_CODENAME} main
EOF
    # Drop the stale/incorrect amdgpu.list an older bootstrap may have written;
    # it points at the non-existent amdgpu/${ROCM_VERSION}/ubuntu and otherwise
    # breaks every subsequent `apt update`.
    sudo rm -f /etc/apt/sources.list.d/amdgpu.list
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
      | sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
    sudo apt update
    sudo apt install -y rocm \
      || die "ROCm ${ROCM_VERSION} install failed - check repo.radeon.com availability for ${UBUNTU_CODENAME}."
    # GPU compute device access for the current user (takes effect on next
    # login; detect_gfx_arch falls back to sudo until then).
    sudo usermod -aG render,video "$USER"
    sudo ldconfig
    warn "Added $USER to the render/video groups - log out/in (or reboot) before the first real GPU run."
  fi
fi

# -----------------------------------------------------------------------------
# 1c. GPU architecture (auto-detected; drives the torch index + llama.cpp build)
# -----------------------------------------------------------------------------
GFX_ARCH="${GFX_ARCH:-}"
if [[ "$CPU_ONLY" -eq 0 ]]; then
  if [[ -z "$GFX_ARCH" ]]; then
    GFX_ARCH="$(detect_gfx_arch || true)"
  fi
  if [[ -z "$GFX_ARCH" ]]; then
    die "Could not detect the GPU architecture (amdgpu-arch/rocminfo gave nothing).
     If ROCm was just installed, log out/in so the render/video groups apply,
     or pass it explicitly:  GFX_ARCH=gfx1150 ./bootstrap.sh ..."
  fi
  log "GPU architecture: ${GFX_ARCH}"

  # --- llama.cpp HIP build target -------------------------------------------
  # ROCm 7.x's NATIVE gfx1150/gfx1151 (Strix Point / Strix Halo) HIP kernels
  # segfault on the first kernel dispatch in current llama.cpp (llama-server
  # exits -11 right after "initializing slots"). The known-good workaround is to
  # build llama.cpp for gfx1100 (RDNA3) and present the iGPU to the HSA runtime
  # as gfx1100 via HSA_OVERRIDE_GFX_VERSION=11.0.0. We do this ONLY for the
  # llama.cpp build - torch keeps the *real* arch (${GFX_ARCH}) for its wheel
  # index below. Override the target with LLAMA_GFX_ARCH=... if a future ROCm
  # fixes native gfx115x (then unset the override too).
  LLAMA_GFX_ARCH="${LLAMA_GFX_ARCH:-$GFX_ARCH}"
  case "$GFX_ARCH" in
    gfx1150|gfx1151)
      [[ "$LLAMA_GFX_ARCH" == "$GFX_ARCH" ]] && LLAMA_GFX_ARCH="gfx1100"
      ;;
  esac
  HSA_OVERRIDE_FOR_LLAMA=""
  if [[ "$LLAMA_GFX_ARCH" == "gfx1100" && "$GFX_ARCH" != "gfx1100" ]]; then
    HSA_OVERRIDE_FOR_LLAMA="11.0.0"
  fi
  if [[ "$LLAMA_GFX_ARCH" != "$GFX_ARCH" ]]; then
    log "llama.cpp will be built for ${LLAMA_GFX_ARCH} (device is ${GFX_ARCH}; native ${GFX_ARCH} segfaults under ROCm ${ROCM_VERSION})"
  fi
fi

# Persist the HSA override so llama-server (started by workshop/run_notebooks.sh)
# finds it in every shell. It is appended to scripts/ryzen_ai_env.sh - the env file
# the workshop already sources - guarded so re-running bootstrap never duplicates it.
# Scope note: HSA_OVERRIDE_GFX_VERSION only affects the ROCm/HIP GPU runtime
# (llama.cpp, and torch-on-GPU which the workshop does not use at inference
# time). The XDNA2 NPU (VitisAI EP) and all ONNX paths are unaffected.
if [[ "$CPU_ONLY" -eq 0 && -n "${HSA_OVERRIDE_FOR_LLAMA:-}" ]]; then
  ENV_SH="${REPO_ROOT}/scripts/ryzen_ai_env.sh"
  if [[ -f "$ENV_SH" ]]; then
    if ! grep -q 'HSA_OVERRIDE_GFX_VERSION' "$ENV_SH"; then
      log "Recording HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_FOR_LLAMA} in scripts/ryzen_ai_env.sh"
      {
        echo ""
        echo "# llama.cpp is built for gfx1100 on Strix (gfx1150/gfx1151); present the"
        echo "# iGPU as gfx1100 so its HIP kernels load instead of segfaulting. Affects"
        echo "# only the ROCm/HIP GPU runtime - the XDNA2 NPU (VitisAI) is unaffected."
        echo "export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_FOR_LLAMA}"
      } >> "$ENV_SH"
    fi
  else
    warn "scripts/ryzen_ai_env.sh not found - add 'export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_FOR_LLAMA}' to your shell init so llama-server runs on the iGPU."
  fi
fi
ROCM_WHL_INDEX="https://repo.amd.com/rocm/whl/${GFX_ARCH}/"

# -----------------------------------------------------------------------------
# 1d. ROS 2 Jazzy (installed from packages.ros.org if absent)
# -----------------------------------------------------------------------------
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  if [[ "$SKIP_APT" -eq 1 ]]; then
    warn "ROS 2 Jazzy not found at /opt/ros/jazzy but --skip-apt given - cannot install it.
       The ROS 2 transport will be unavailable (set robot.use_ros2: false), or
       rerun without --skip-apt to install ros-jazzy-desktop."
  else
    log "Installing ROS 2 Jazzy (sudo; Ubuntu ${UBUNTU_CODENAME})"
    # ROS 2 requires a UTF-8 locale.
    if ! locale 2>/dev/null | grep -qi 'utf-\?8'; then
      sudo apt install -y locales
      sudo locale-gen en_US en_US.UTF-8
      sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
      export LANG=en_US.UTF-8
    fi
    # The universe component hosts ROS dependencies (enabled by default on
    # desktop installs, but not on every server/minimal image).
    sudo apt install -y software-properties-common curl
    sudo add-apt-repository -y universe
    # Install the ros2-apt-source package: it ships BOTH the apt source entry
    # and the signing key, and apt keeps the key current when upstream rotates
    # it (the old "curl ros.key into a keyring" method goes stale).
    ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
      | grep -F '"tag_name"' | awk -F'"' '{print $4}')" \
      || die "Could not query the latest ros2-apt-source release from api.github.com."
    [[ -n "$ROS_APT_SOURCE_VERSION" ]] || die "Empty ros2-apt-source version - check api.github.com output."
    curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb" \
      || die "Could not download ros2-apt-source for '${UBUNTU_CODENAME}' - Jazzy supports Ubuntu 24.04 (noble)."
    sudo apt install -y /tmp/ros2-apt-source.deb
    rm -f /tmp/ros2-apt-source.deb
    sudo apt update
    sudo apt install -y ros-jazzy-desktop \
      || die "ROS 2 Jazzy install failed - check packages.ros.org availability for ${UBUNTU_CODENAME}."
  fi
fi

# -----------------------------------------------------------------------------
# 2. uv + venv (in repo root, layered over system/ROS python)
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

log "Creating uv venv at ${VENV_DIR}"
# Pin to the SYSTEM Python 3.12 (Ubuntu 24.04 default). Two reasons:
#   - lerobot 0.5.2 requires Python >= 3.12 (uv's managed 3.11 fails resolution)
#   - --system-site-packages only exposes ROS 2 Jazzy's rclpy if the venv
#     interpreter is ABI-identical to the system python it was built against.
SYS_PY="$(command -v /usr/bin/python3.12 || command -v python3.12 || true)"
if [[ -z "$SYS_PY" ]]; then
  die "python3.12 not found - install it first (apt install python3.12 python3.12-dev) or check your Ubuntu 24.04 setup."
fi
# --allow-existing keeps an existing venv's interpreter and IGNORES --python,
# so a stale venv (e.g. one uv created with its managed 3.11) must be removed.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_VER="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [[ "$VENV_VER" != "3.12" ]]; then
    warn "Existing venv uses Python ${VENV_VER} (need 3.12) - recreating it."
    rm -rf "$VENV_DIR"
  fi
fi
uv venv --python "$SYS_PY" --system-site-packages --allow-existing "$VENV_DIR"
# Keep colcon (if this repo lands in a ROS workspace) out of the venv.
touch "${VENV_DIR}/COLCON_IGNORE" "${VENV_DIR}/AMENT_IGNORE"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# -----------------------------------------------------------------------------
# 3. Python dependencies
# -----------------------------------------------------------------------------
log "Installing Python dependencies into the venv"
uv pip install -r "${REPO_ROOT}/requirements.txt"

if [[ "$CPU_ONLY" -eq 0 ]]; then
  log "Installing PyTorch ROCm wheels for ${GFX_ARCH} (AMD repo)"
  uv pip install --index-url "$ROCM_WHL_INDEX" \
    "$TORCH_SPEC" "$TORCHVISION_SPEC" "$TORCHAUDIO_SPEC" \
    || warn "ROCm torch install failed - check ${ROCM_WHL_INDEX} availability; falling back to whatever torch resolves."
  # NOTE: the Ryzen AI onnxruntime (VitisAI EP) wheels are installed in
  # section 4b, AFTER requirements/torch/lerobot, so no later dependency
  # resolution can clobber them with a stock or ROCm onnxruntime.
else
  log "--cpu-only: installing CPU torch"
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

log "Installing LeRobot (feetech extras) from source"
mkdir -p "$THIRD_PARTY"
if [[ ! -d "${THIRD_PARTY}/lerobot/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "${THIRD_PARTY}/lerobot"
fi
uv pip install -e "${THIRD_PARTY}/lerobot[feetech]"

# -----------------------------------------------------------------------------
# 4. llama.cpp with HIP/ROCm
# -----------------------------------------------------------------------------
if [[ "$SKIP_LLAMA" -eq 0 ]]; then
  log "Building llama.cpp (llama-server) with HIP for ${LLAMA_GFX_ARCH}"
  if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  fi
  pushd "$LLAMA_DIR" >/dev/null
  # Use the SYSTEM cmake: a pip-installed cmake inside the venv shadows it on
  # PATH and breaks if the venv ever moves (absolute shebang paths).
  CMAKE_BIN="$(command -v /usr/bin/cmake || command -v cmake)"
  if [[ "$CPU_ONLY" -eq 0 && -d /opt/rocm ]]; then
    # IMPORTANT: pin everything to the SYSTEM ROCm at /opt/rocm. The AMD torch
    # wheels install a pip ROCm runtime (_rocm_sdk_core) whose hipconfig shim
    # shadows the system one while the venv is active; it has no HIP compiler
    # or hip-lang CMake package, so HIP detection fails if cmake picks it up.
    # AMDGPU_TARGETS / GPU_TARGETS use ${LLAMA_GFX_ARCH} (gfx1100 on Strix) - see
    # the down-targeting note in section 1c.
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    HIPCXX=/opt/rocm/llvm/bin/clang++ \
    "$CMAKE_BIN" -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS="$LLAMA_GFX_ARCH" \
      -DGPU_TARGETS="$LLAMA_GFX_ARCH" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_BUILD_TYPE=Release
  else
    warn "Building llama.cpp CPU-only (no /opt/rocm or --cpu-only)."
    "$CMAKE_BIN" -S . -B build -DCMAKE_BUILD_TYPE=Release
  fi
  "$CMAKE_BIN" --build build --config Release -j"$(nproc)" --target llama-server llama-cli
  popd >/dev/null
else
  log "Skipping llama.cpp build (--skip-llama)"
fi

# -----------------------------------------------------------------------------
# 4b. Enforce numpy < 2 and huggingface-hub < 1.0
# -----------------------------------------------------------------------------
# onnxruntime and the AMD/Ryzen wheels are compiled against the numpy 1.x ABI;
# torch / ultralytics may have pulled numpy 2.x as a transitive upgrade, which
# crashes every `import onnxruntime`. transformers requires huggingface-hub <1.0
# but lerobot / optimum can drag in 1.x. Re-pin both as the last dependency
# action so they can't be clobbered by an earlier step.
NPVER="$(python -c 'import numpy,sys; print(numpy.__version__)' 2>/dev/null || echo none)"
case "$NPVER" in
  2.*|none)
    log "Re-pinning numpy < 2 (found ${NPVER}; onnxruntime needs the 1.x ABI)"
    uv pip install 'numpy<2' \
      || warn "Could not downgrade numpy below 2.0 - onnxruntime imports may fail."
    ;;
  *)
    log "numpy ${NPVER} OK (1.x ABI)"
    ;;
esac

HUBVER="$(python -c 'import huggingface_hub as h; print(h.__version__)' 2>/dev/null || echo none)"
case "$HUBVER" in
  1.*|none)
    log "Re-pinning huggingface-hub < 1.0 (found ${HUBVER}; transformers requires <1.0)"
    uv pip install 'huggingface_hub[cli]>=0.34,<1.0' \
      || warn "Could not pin huggingface-hub below 1.0 - transformers imports may fail."
    ;;
  *)
    log "huggingface-hub ${HUBVER} OK (<1.0)"
    ;;
esac

# onnx 1.18+ writes model IR version 13, which the VitisAI ORT runtime (max IR
# 11) cannot load. Cap it so exported/probe models stay loadable on the NPU.
ONNXVER="$(python -c 'import onnx; print(onnx.__version__)' 2>/dev/null || echo none)"
case "$ONNXVER" in
  none) : ;;
  *)
    if python -c 'import onnx,sys; from packaging.version import Version; sys.exit(0 if Version(onnx.__version__) >= Version("1.18") else 1)' 2>/dev/null; then
      log "Re-pinning onnx < 1.18 (found ${ONNXVER}; VitisAI ORT needs IR <= 11)"
      uv pip install 'onnx>=1.16,<1.18' \
        || warn "Could not cap onnx below 1.18 - NPU model loads may fail with IR-version errors."
    else
      log "onnx ${ONNXVER} OK (IR <= 11)"
    fi
    ;;
esac

# -----------------------------------------------------------------------------
# 4c. Enforce the onnxruntime flavor (LAST dependency action)
# -----------------------------------------------------------------------------
# Policy: ROCm is used ONLY by llama.cpp (and the torch wheels). onnxruntime
# must be either the Ryzen AI VitisAI build (NPU) or stock CPU - never
# onnxruntime-rocm, which transitive deps (requirements / lerobot / AMD index
# resolution) can drag in. All onnxruntime variants unpack into the same
# site-packages/onnxruntime directory, so a leftover ROCm/stock build shadows
# or corrupts the VitisAI one. This runs after every other pip action so
# nothing can clobber the result.
if [[ "$CPU_ONLY" -eq 0 && -n "$RYZEN_AI_WHEELS" && -d "$RYZEN_AI_WHEELS" ]]; then
  log "Installing Ryzen AI onnxruntime (VitisAI EP) from ${RYZEN_AI_WHEELS}"
  # Wheels may sit anywhere inside the SDK folder - search the whole subtree.
  mapfile -t RAI_WHLS < <(find "$RYZEN_AI_WHEELS" -name "*.whl" \
    \( -iname "*onnxruntime*vitisai*" -o -iname "*voe*" -o -iname "*vitis*" \) | sort -u)
  if [[ ${#RAI_WHLS[@]} -eq 0 ]]; then
    # Fall back to every wheel found in the subtree.
    mapfile -t RAI_WHLS < <(find "$RYZEN_AI_WHEELS" -name "*.whl" | sort -u)
  fi
  if [[ ${#RAI_WHLS[@]} -gt 0 ]]; then
    printf '  %s\n' "${RAI_WHLS[@]}"
    uv pip uninstall onnxruntime onnxruntime-rocm onnxruntime-gpu onnxruntime-vitisai voe || true
    uv pip install "${RAI_WHLS[@]}" \
      || warn "Could not install Ryzen AI wheels from ${RYZEN_AI_WHEELS}"
  else
    warn "No .whl files found anywhere under ${RYZEN_AI_WHEELS} - VitisAI EP not installed."
  fi
else
  # No SDK (or --cpu-only): make sure the ROCm build isn't what's installed.
  if python -c 'import onnxruntime' 2>/dev/null && \
     uv pip list 2>/dev/null | grep -qi 'onnxruntime-rocm'; then
    log "Replacing onnxruntime-rocm with stock CPU onnxruntime (ROCm is for llama.cpp only)"
    uv pip uninstall onnxruntime onnxruntime-rocm onnxruntime-gpu || true
    uv pip install onnxruntime \
      || warn "Could not install stock onnxruntime - imports may fail."
  fi
fi

# -----------------------------------------------------------------------------
# 4d. Enforce GUI-enabled OpenCV (LAST pip action, like onnxruntime)
# -----------------------------------------------------------------------------
# yolo_pose_npu.py and the mediapipe demo open windows via cv2.imshow(), which
# the *-headless* OpenCV wheels omit - at runtime they raise "The function is
# not implemented. Rebuild the library with ... GTK+/Qt support". mediapipe,
# ultralytics and optimum can each pull opencv-python-headless transitively, and
# every OpenCV variant unpacks into the SAME site-packages/cv2 directory (last
# install wins), so a headless wheel installed later silently disables HighGUI.
# Strip all variants and reinstall the GUI build LAST. opencv-contrib-python is
# GUI-enabled, is the variant mediapipe declares as its dependency, and is a
# superset of opencv-python (so ultralytics is satisfied at runtime too). numpy<2
# is pinned in the same resolution so this step can't drag the 2.x ABI back in.
log "Pinning GUI-enabled OpenCV (cv2.imshow) - removing any headless build"
uv pip uninstall opencv-python opencv-python-headless \
                 opencv-contrib-python opencv-contrib-python-headless || true
uv pip install 'opencv-contrib-python>=4.10' 'numpy<2' \
  || warn "GUI OpenCV install failed - cv2.imshow windows (yolo/mediapipe demos) may not open."

# -----------------------------------------------------------------------------
# 5. Models
# -----------------------------------------------------------------------------
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  mkdir -p "$MODELS_DIR" "$LLAMA_OUT_DIR" "${REPO_ROOT}/cache" "${REPO_ROOT}/logs"

  log "YOLO ONNX export is a manual setup step — see README"

  log "Downloading Llama-3.2-3B-Instruct GGUF (Q4_K_M)"
  if [[ -f "${LLAMA_OUT_DIR}/${LLAMA_GGUF_FILE}" ]]; then
    echo "Already present: ${LLAMA_OUT_DIR}/${LLAMA_GGUF_FILE}"
  else
    hf download "$LLAMA_GGUF_REPO" "$LLAMA_GGUF_FILE" --local-dir "$LLAMA_OUT_DIR" \
      || warn "GGUF download failed. Llama 3.2 is a gated/licensed model - log in with 'hf auth login' (accept the Llama 3.2 license on huggingface.co) or place ${LLAMA_GGUF_FILE} in ${LLAMA_OUT_DIR} manually."
  fi
else
  log "Skipping model download/export (--skip-models)"
fi

# -----------------------------------------------------------------------------
# 5b. Compile NPU models (yolo pose/detect) using the full Ryzen AI SDK venv
# -----------------------------------------------------------------------------
# The deployment .venv can RUN precompiled NPU models but cannot COMPILE them.
# Compilation needs the full SDK venv. bootstrap installs it at the repo root
# (./ryzenai-compile) from the SDK wheel dir (./ryzen_ai*), then compiles every
# exported model into cache/. If the SDK wheels aren't present it prints the
# steps and continues (llama still runs on the iGPU and the YOLO models can
# fall back to device: cpu).
if [[ "$SKIP_COMPILE" -eq 0 && "$CPU_ONLY" -eq 0 ]]; then
  # 1) Install the full SDK venv into the repo root if it isn't there yet.
  if [[ ! -x "${RYZEN_AI_COMPILE_VENV}/bin/python" ]]; then
    if [[ -n "$RYZEN_AI_SDK_DIR" && -f "${RYZEN_AI_SDK_DIR}/install_ryzen_ai.sh" ]]; then
      log "Installing full Ryzen AI SDK (compiler) into ${RYZEN_AI_COMPILE_VENV}"
      # install_ryzen_ai.sh must run from the wheel dir; it refuses an existing
      # target, so we point it at our (absent) repo-root path.
      ( cd "$RYZEN_AI_SDK_DIR" && \
        bash ./install_ryzen_ai.sh -a yes -p "$RYZEN_AI_COMPILE_VENV" -n ryzenai-compile ) \
        || warn "SDK install failed - see output above. NPU compile will be skipped."
    else
      warn "Ryzen AI SDK wheel dir (./ryzen_ai*) not found - cannot install the compiler.
       Place the SDK (with install_ryzen_ai.sh) in the repo root, then re-run
       ./bootstrap.sh --skip-apt --skip-llama --skip-models"
    fi
  fi

  # 2) Compile every exported model into cache/ using the SDK venv's python,
  #    run against THIS repo so cache_dir/cache_key/configs match the runtime.
  # Ensure the deployment venv is the active one (the SDK installer activates
  # its own venv in a subshell; re-source ours so uv pip targets .venv).
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  if [[ -x "${RYZEN_AI_COMPILE_VENV}/bin/python" ]]; then
    log "Compiling NPU models (yolo pose + yolo detect) with ${RYZEN_AI_COMPILE_VENV}"
    # compile_npu_models.py self-sets LD_LIBRARY_PATH from its own venv and
    # re-execs, so the VitisAI EP loads correctly and a CPU fallback is treated
    # as a hard error (no silent no-op "compile").
    if PYTHONPATH="${REPO_ROOT}" "${RYZEN_AI_COMPILE_VENV}/bin/python" \
         "${REPO_ROOT}/scripts/compile_npu_models.py"; then
      log "NPU compile complete - artifacts in ${REPO_ROOT}/cache"
      # The deployment .venv needs flexmlrt at runtime to load compiled models.
      if [[ -n "$RYZEN_AI_SDK_DIR" ]]; then
        FLEXMLRT_WHL=$(ls "${RYZEN_AI_SDK_DIR}"/flexmlrt*.whl 2>/dev/null | head -1 || true)
        if [[ -n "$FLEXMLRT_WHL" ]]; then
          log "Installing flexmlrt runtime into deployment .venv"
          uv pip install "$FLEXMLRT_WHL" || warn "flexmlrt install failed - NPU models may not load at runtime."
        fi
      fi
    else
      warn "NPU compile failed. Retry with:
       source ${RYZEN_AI_COMPILE_VENV}/bin/activate
       cd ${REPO_ROOT} && python scripts/compile_npu_models.py"
    fi
  else
    warn "No compiler venv at ${RYZEN_AI_COMPILE_VENV} - skipping NPU model compilation.
       Until compiled, set yolo_pose.device / yolo_detect.device to 'cpu' in workshop.yaml to run."
  fi
else
  [[ "$SKIP_COMPILE" -eq 1 ]] && log "Skipping NPU model compilation (--skip-compile)"
fi

# -----------------------------------------------------------------------------
# 6. Verify
# -----------------------------------------------------------------------------
log "Verifying environment"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/ryzen_ai_env.sh"
python - <<'PY'
import importlib, sys

ok = True
for mod in ("numpy", "cv2", "mediapipe", "onnxruntime", "yaml", "requests"):
    try:
        importlib.import_module(mod)
        print(f"  [ok] {mod}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {mod}: {e}")

try:
    import cv2
    info = cv2.getBuildInformation()
    gui_line = next((l for l in info.splitlines() if l.strip().startswith("GUI")), "")
    if "NONE" in gui_line.upper() or not gui_line:
        ok = False
        print(f"  [MISSING] OpenCV HighGUI - headless build detected ({gui_line.strip() or 'no GUI line'}); cv2.imshow will fail.")
    else:
        print(f"  [ok] OpenCV HighGUI ({gui_line.split(':',1)[-1].strip()}) - cv2.imshow available")
except Exception as e:
    print(f"  [warn] could not probe OpenCV GUI support: {e}")

try:
    import torch
    print(f"  [ok] torch {torch.__version__} | GPU available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  [MISSING] torch: {e}")

import onnxruntime as ort
eps = ort.get_available_providers()
print(f"  onnxruntime EPs: {eps}")
if "VitisAIExecutionProvider" not in eps:
    print("  [warn] VitisAI EP absent - NPU components will run on CPU (set RYZEN_AI_WHEELS and rerun).")
else:
    # Registration != loadable. The native libs (voe/lib) must be on
    # LD_LIBRARY_PATH or the EP fails at session creation and silently
    # falls back to CPU. Probe with a tiny model to confirm it really loads.
    import os, tempfile, numpy as np
    try:
        from onnx import helper, TensorProto
        import onnx
        X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        node = helper.make_node("Identity", ["x"], ["y"])
        g = helper.make_graph([node], "probe", [X], [Y])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        # VitisAI ORT supports up to IR v11; newer onnx defaults to v13 and the
        # probe fails to load. Pin a compatible IR version.
        m.ir_version = 10
        f = os.path.join(tempfile.mkdtemp(), "probe.onnx")
        onnx.save(m, f)
        sess = ort.InferenceSession(f, providers=["VitisAIExecutionProvider", "CPUExecutionProvider"])
        used = sess.get_providers()
        if "VitisAIExecutionProvider" in used:
            print("  [ok] VitisAI EP loads (NPU runtime libs resolved)")
        else:
            print(f"  [warn] VitisAI registered but did not load; active: {used}")
    except Exception as e:
        print(f"  [warn] VitisAI EP failed to initialize: {e}")
        print("         → NPU libs likely missing from LD_LIBRARY_PATH "
              "(scripts/ryzen_ai_env.sh should add voe/lib).")

try:
    import lerobot
    print(f"  [ok] lerobot {getattr(lerobot, '__version__', '?')}")
except Exception as e:
    print(f"  [MISSING] lerobot: {e}")

sys.exit(0 if ok else 1)
PY

cat <<EOF

=============================================================================
 Bootstrap complete.
=============================================================================
 If you have not exported the YOLO models yet, do that now and rerun
 bootstrap to compile them for the NPU (see README, "Installation").

 Before the first hardware run:
   * Calibrate the arm: see calibration/ and arm_calibrator/README.md
   * Set motor_port / robot_id / camera devices in
     workshop/{project,solution}/config/workshop.yaml
   * Serial permissions: sudo chmod 666 /dev/ttyACM0   (resets on replug)

 Start the workshop (registers the kernel, starts llama-server, opens Jupyter):

     cd workshop
     ./run_notebooks.sh
=============================================================================
EOF
