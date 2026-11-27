#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Everything the voice front end in config/pipeline.yaml (voice_intent) needs.
# gemma4-e4b-it needs the Gemma 4 GGUF and its audio projector, the Silero VAD
# model, and a llama-server built with Gemma 4 audio. A machine set up before
# Gemma, or with bootstrap.sh --skip-models, has none of the models, and the
# microphone stays off without them. The pipeline and notebook launchers call
# this, like ensure_ros2_control.sh. It downloads what is missing, does nothing
# when everything is present, and fails with the reason when something cannot
# be put in place.
#
#   scripts/ensure_voice.sh                # the models, then the llama-server check
#   scripts/ensure_voice.sh --models-only  # bootstrap.sh: it builds llama.cpp itself
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PY="${VENV_DIR}/bin/python"
HF="${VENV_DIR}/bin/hf"
LLAMA_DIR="${ROOT}/third_party/llama.cpp"
GEMMA_GGUF_REPO="ggml-org/gemma-4-E4B-it-GGUF"
SILERO_URL="https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
# The llama.cpp commit that added the Gemma 4 audio encoder (mtmd, PR #21421).
LLAMA_GEMMA4_AUDIO="547765a93"

fail() { echo "ensure_voice: $*" >&2; exit 1; }

MODELS_ONLY=0
case "${1:-}" in
  "") ;;
  --models-only) MODELS_ONLY=1 ;;
  *) echo "usage: $0 [--models-only]" >&2; exit 2 ;;
esac

[[ -x "$PY" ]] || fail "no venv at ${VENV_DIR} - run ./bootstrap.sh first."

# config/pipeline.yaml is the authority for the voice front end and its paths.
# Run from ROOT: python - puts the working directory first on sys.path, so a
# caller standing in another clone would otherwise read that clone's config.
VOICE_ENV="$(cd "$ROOT" && "$PY" - <<'PY'
import shlex
from vla_pipeline.utils.config import load_config, resolve
cfg = load_config()
print("VOICE_INTENT=%s" % shlex.quote(str(cfg.get("voice_intent"))))
g = cfg.get("gemma4-e4b-it") or {}
for key, name in (("model_gguf", "GEMMA_GGUF"), ("mmproj", "GEMMA_MMPROJ"),
                  ("vad_model", "SILERO_OUT"), ("server_bin", "GEMMA_SERVER")):
    print("%s=%s" % (name, shlex.quote(str(resolve(g[key])) if g.get(key) else "")))
PY
)" || fail "could not read config/pipeline.yaml"
eval "$VOICE_ENV"

if [[ "$VOICE_INTENT" != "gemma4-e4b-it" ]]; then
  exit 0
fi
for name in GEMMA_GGUF GEMMA_MMPROJ SILERO_OUT GEMMA_SERVER; do
  [[ -n "${!name}" ]] || fail "config/pipeline.yaml gemma4-e4b-it has no ${name} path."
done

for file in "$GEMMA_GGUF" "$GEMMA_MMPROJ"; do
  [[ -f "$file" ]] && continue
  [[ -x "$HF" ]] || fail "${file} is missing and there is no hf CLI in ${VENV_DIR} - run ./bootstrap.sh."
  echo "ensure_voice: downloading $(basename "$file") from ${GEMMA_GGUF_REPO}"
  mkdir -p "$(dirname "$file")"
  "$HF" download "$GEMMA_GGUF_REPO" "$(basename "$file")" --local-dir "$(dirname "$file")" \
    || fail "could not download $(basename "$file") from ${GEMMA_GGUF_REPO}."
  [[ -f "$file" ]] || fail "hf download finished but ${file} is not there."
done

if ! { [[ -f "$SILERO_OUT" ]] && echo "${SILERO_SHA256}  ${SILERO_OUT}" | sha256sum -c --status; }; then
  echo "ensure_voice: downloading the Silero VAD model (v6.2.3)"
  mkdir -p "$(dirname "$SILERO_OUT")"
  curl --fail --location --show-error -o "${SILERO_OUT}.part" "$SILERO_URL" \
    || fail "could not download ${SILERO_URL}."
  echo "${SILERO_SHA256}  ${SILERO_OUT}.part" | sha256sum -c --status \
    || fail "the Silero VAD download does not match sha256 ${SILERO_SHA256}."
  mv "${SILERO_OUT}.part" "$SILERO_OUT"
fi

[[ "$MODELS_ONLY" -eq 1 ]] && exit 0

REBUILD="./bootstrap.sh --skip-apt --skip-models --skip-compile"
[[ -x "$GEMMA_SERVER" ]] || fail "${GEMMA_SERVER} is missing - run ${REBUILD} (it builds llama.cpp)."
# The binary reports the commit it was built from, e.g.
# "version: 0.6.0-dev (build 11548, commit 781dbc5ac)".
BUILT="$("$GEMMA_SERVER" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -n1)"
[[ -n "$BUILT" ]] || fail "${GEMMA_SERVER} --version does not name its commit."
[[ -d "${LLAMA_DIR}/.git" ]] || fail "${LLAMA_DIR} is not a git clone, so llama-server ${BUILT} cannot be checked for Gemma 4 audio."
if ! git -C "$LLAMA_DIR" merge-base --is-ancestor "$LLAMA_GEMMA4_AUDIO" "$BUILT" 2>/dev/null; then
  fail "llama-server was built from llama.cpp ${BUILT}, which predates Gemma 4 audio (${LLAMA_GEMMA4_AUDIO}).
       Update and rebuild it: git -C third_party/llama.cpp pull --ff-only && ${REBUILD}"
fi
