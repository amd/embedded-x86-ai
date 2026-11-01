#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Ryzen AI NPU runtime environment.
#
# The VitisAI ONNX Runtime EP needs its native libraries (libxcompiler-core-*,
# libonnxruntime_vitisai_ep.so, etc.) on LD_LIBRARY_PATH. The `voe` and
# `onnxruntime-vitisai` wheels install them into the venv but do NOT add their
# directories to the loader path, so the EP fails with
#   "libxcompiler-core-without-symbol.so: cannot open shared object file"
# and onnxruntime silently falls back to the CPU EP.
#
# Source this AFTER activating the venv:
#     source scripts/ryzen_ai_env.sh
#
# Every entry point sources it, so it also carries two settings for the whole
# stack (at the end): llama.cpp's GPU target and the ROS 2 transport.
#
# Safe to source repeatedly.

# Resolve the active venv (VIRTUAL_ENV is set by `source .venv/bin/activate`).
_RAI_VENV="${VIRTUAL_ENV:-}"
if [[ -z "$_RAI_VENV" ]]; then
  _RAI_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _RAI_VENV="${_RAI_SELF}/.venv"
fi

_RAI_PYVER="$("${_RAI_VENV}/bin/python" -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo python3.12)"
_RAI_SP="${_RAI_VENV}/lib/${_RAI_PYVER}/site-packages"

# Directories inside the venv that hold the EP's native .so files.
_RAI_CANDIDATES=(
  "${_RAI_SP}/voe/lib"             # libxcompiler-core-*, flexml-runner, etc.
  "${_RAI_SP}/flexmlrt/lib"        # libflexmlrt.so - needed to RUN compiled VAIML models
  "${_RAI_SP}/onnxruntime/capi"    # libonnxruntime_vitisai_ep.so + providers
)

for _d in "${_RAI_CANDIDATES[@]}"; do
  if [[ -d "$_d" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$_d:"*) ;;                                  # already present
      *) export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done

if [[ -z "${XILINX_XRT:-}" && -f /opt/xilinx/xrt/setup.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/xilinx/xrt/setup.sh >/dev/null
fi

# Strix (STX) NPU firmware xclbin ships inside the flexml wheel. The deployment
# venv installs voe and flexmlrt, not that wheel, and install_ryzen_ai.sh is
# what unpacks it into an SDK venv. XLNX_VART_FIRMWARE must point at a specific
# .xclbin FILE (not the directory), else the EP errors with "xclbin is set to
# a directory".
_RAI_XCLBIN_REL="lib/${_RAI_PYVER}/site-packages/flexml/flexml_extras/data/ryzen-ai/stx"
_RAI_XCLBIN_DIR="${_RAI_SP}/flexml/flexml_extras/data/ryzen-ai/stx"
_RAI_XCLBIN_CANDIDATES=()
[[ -n "${RYZEN_AI_VENV:-}" ]] && _RAI_XCLBIN_CANDIDATES+=("${RYZEN_AI_VENV}/${_RAI_XCLBIN_REL}")
if [[ -n "${RYZEN_AI_WHEELS:-}" ]]; then
  _RAI_XCLBIN_CANDIDATES+=("${RYZEN_AI_WHEELS}/venv/${_RAI_XCLBIN_REL}" "${RYZEN_AI_WHEELS}/.venv/${_RAI_XCLBIN_REL}")
fi
for _d in "$HOME"/ryzen_ai*/venv "$HOME"/ryzen_ai*/.venv /opt/ryzen_ai*/venv /opt/ryzen_ai*/.venv; do
  [[ -d "$_d" ]] && _RAI_XCLBIN_CANDIDATES+=("${_d}/${_RAI_XCLBIN_REL}")
done
# The deployment venv does not install the flexml wheel. Pull the small stx
# firmware out of the wheel that matches the installed voe, then fall back to
# an SDK venv (venv or .venv) if that wheel is not present.
if ! compgen -G "${_RAI_XCLBIN_DIR}/*.xclbin" >/dev/null; then
  _RAI_VOE_VER=""
  _d="$(ls -d "${_RAI_SP}"/voe-*.dist-info 2>/dev/null | head -1 || true)"
  [[ -n "$_d" ]] && _RAI_VOE_VER="$(basename "$_d" | sed -n 's/^voe-\(.*\)\.dist-info$/\1/p')"
  _RAI_WHL=""
  _RAI_WHEEL_DIRS=()
  [[ -n "${RYZEN_AI_WHEELS:-}" ]] && _RAI_WHEEL_DIRS+=("$RYZEN_AI_WHEELS")
  for _d in "$HOME"/ryzen_ai* /opt/ryzen_ai*; do
    [[ -d "$_d" ]] && _RAI_WHEEL_DIRS+=("$_d")
  done
  if [[ -n "$_RAI_VOE_VER" ]]; then
    for _d in "${_RAI_WHEEL_DIRS[@]}"; do
      _x="$(ls "$_d"/flexml-"${_RAI_VOE_VER}"-*.whl 2>/dev/null | head -1 || true)"
      [[ -n "$_x" ]] && _RAI_WHL="$_x" && break
    done
  fi
  if [[ -n "$_RAI_WHL" ]]; then
    mkdir -p "$_RAI_XCLBIN_DIR"
    unzip -qo -j "$_RAI_WHL" 'flexml/flexml_extras/data/ryzen-ai/stx/*.xclbin' -d "$_RAI_XCLBIN_DIR"
  fi
fi
if ! compgen -G "${_RAI_XCLBIN_DIR}/*.xclbin" >/dev/null; then
  for _d in "${_RAI_XCLBIN_CANDIDATES[@]}"; do
    if compgen -G "${_d}/*.xclbin" >/dev/null; then
      _RAI_XCLBIN_DIR="$_d"
      break
    fi
  done
fi
if [[ -z "${XLNX_VART_FIRMWARE:-}" && -d "$_RAI_XCLBIN_DIR" ]]; then
  if [[ -n "${RAI_XCLBIN:-}" && -f "$RAI_XCLBIN" ]]; then
    export XLNX_VART_FIRMWARE="$RAI_XCLBIN"
  else
    # Strix Halo uses the 2x4x4 overlay; fall back to 4x4 then anything.
    for _x in unified-2x4x4.xclbin unified-4x4.xclbin; do
      if [[ -f "${_RAI_XCLBIN_DIR}/${_x}" ]]; then
        export XLNX_VART_FIRMWARE="${_RAI_XCLBIN_DIR}/${_x}"
        break
      fi
    done
  fi
  if [[ -z "${XLNX_VART_FIRMWARE:-}" ]]; then
    _x=$(ls "${_RAI_XCLBIN_DIR}"/*.xclbin 2>/dev/null | head -1 || true)
    [[ -n "$_x" ]] && export XLNX_VART_FIRMWARE="$_x"
  fi
fi

if [[ ! -e "${_RAI_SP}/voe/lib/libxcompiler-core-without-symbol.so" ]]; then
  echo "[ryzen_ai_env] warning: VitisAI native libs not found in venv voe/lib;" >&2
  echo "[ryzen_ai_env] the NPU EP will fall back to CPU. Ensure the 'voe' wheel installed." >&2
fi

unset _RAI_VENV _RAI_SELF _RAI_PYVER _RAI_SP _RAI_CANDIDATES _d _RAI_XCLBIN_REL _RAI_XCLBIN_CANDIDATES _RAI_XCLBIN_DIR _RAI_VOE_VER _RAI_WHL _RAI_WHEEL_DIRS _x

# llama.cpp is built for gfx1100 on Strix (gfx1150/gfx1151); present the
# iGPU as gfx1100 so its HIP kernels load instead of segfaulting. Affects
# only the ROCm/HIP GPU runtime - the XDNA2 NPU (VitisAI) is unaffected.
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# ROS 2 camera images (640x480 bgr8, 0.9 MB each, 30 fps) between processes on
# this machine. On Fast DDS's default transports, frames were lost for up to
# 1.7 s at a time while the pipeline ran on the twin (a pick then said the arm
# camera was offline); with its LARGE_DATA transports the longest gap was
# 0.33 s. Every ROS process of the stack sources this file, so they all use it.
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA}"
