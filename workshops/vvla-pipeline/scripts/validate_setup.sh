#!/usr/bin/env bash
# Validate the VVLA setup fixes.
#
#   ./scripts/validate_setup.sh
#   ./scripts/validate_setup.sh --full
#   ./scripts/validate_setup.sh --full --wav /path/to/speech.wav
#
# The default run checks scripts, requirements, the external SDK environment,
# the VAIML caches already in cache/, and an existing deployment .venv.
# It never compiles. --full repeats bootstrap with --skip-compile so those
# caches stay in place. --wav runs Whisper on the NPU against the existing cache.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${RYZEN_AI_WHEELS:-${HOME}/ryzen_ai-1.7.1}"
SDK_VENV="${RYZEN_AI_VENV:-${SDK_ROOT}/venv}"
FULL=0
WAV="${WHISPER_WAV:-}"
PASSES=0
FAILURES=0

usage() {
  cat <<EOF
Usage: ./scripts/validate_setup.sh [--full] [--wav FILE]

  default   static checks, SDK VitisAI registration, existing VAIML caches, and .venv
  --full    also repeat bootstrap with --skip-compile
  --wav     transcribe FILE with Whisper on the NPU, using cache/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1 ;;
    --wav)
      shift
      WAV="${1:-}"
      [[ -n "$WAV" ]] || { echo "--wav needs a file" >&2; exit 2; }
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

pass() { printf 'PASS  %s\n' "$1"; PASSES=$((PASSES + 1)); }
fail() { printf 'FAIL  %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }
note() { printf 'INFO  %s\n' "$*"; }

run_check() {
  local name="$1"
  shift
  if "$@"; then
    pass "$name"
  else
    fail "$name"
  fi
}

note "workshop: ${REPO_ROOT}"
note "SDK:      ${SDK_ROOT}"

run_check "git diff whitespace" git -C "$REPO_ROOT" diff --check

run_check "shell syntax" bash -n \
  "$REPO_ROOT/bootstrap.sh" \
  "$REPO_ROOT/run_pipeline.sh" \
  "$REPO_ROOT/activate_venv.sh" \
  "$REPO_ROOT/scripts/compile_npu_models.sh" \
  "$REPO_ROOT/scripts/install_kernel.sh" \
  "$REPO_ROOT/scripts/ryzen_ai_env.sh" \
  "$REPO_ROOT/workshop/run_notebooks.sh"

run_check "python syntax" python3 -m py_compile \
  "$REPO_ROOT/scripts/compile_npu_models.py" \
  "$REPO_ROOT/scripts/verify_npu_stack.py" \
  "$REPO_ROOT/vla_pipeline/utils/npu_session.py" \
  "$REPO_ROOT/vla_pipeline/audio/whisper_npu.py" \
  "$REPO_ROOT/vla_pipeline/main.py"

run_check "scripts are executable" bash -c '
  set -euo pipefail
  root="$1"
  for path in bootstrap.sh run_pipeline.sh activate_venv.sh \
      scripts/compile_npu_models.sh scripts/verify_npu_stack.py; do
    [[ -x "$root/$path" ]] || { echo "not executable: $path" >&2; exit 1; }
  done
' bash "$REPO_ROOT"

run_check "requirements do not install stock onnxruntime" bash -c '
  set -euo pipefail
  awk "!/^[[:space:]]*(#|$)/" "$1/requirements.txt" |
    grep -Eiq "^onnxruntime([<>=[:space:]]|$)" && exit 1
  exit 0
' bash "$REPO_ROOT"

run_check "Ryzen AI wheels are discoverable through a symlink" bash -c '
  set -euo pipefail
  sdk="$1"
  link="/tmp/vvla-ryzen-ai-wheels-link"
  ln -sfn "$sdk" "$link"
  [[ "$(readlink -f "$link")" == "$(readlink -f "$sdk")" ]] || exit 1
  find -L "$link" -type f -iname "*onnxruntime*vitisai*.whl" | grep -q .
  find -L "$link" -type f -iname "voe-*.whl" | grep -q .
  find -L "$link" -type f -iname "flexmlrt-*.whl" | grep -q .
' bash "$SDK_ROOT"

run_check "requirements resolve without stock onnxruntime" bash -c '
  set -euo pipefail
  command -v uv >/dev/null || exit 1
  tmp="/tmp/vvla-requirements-check"
  log="/tmp/vvla-requirements-dry-run.txt"
  rm -rf "$tmp"
  uv venv --quiet --python 3.12 "$tmp"
  uv pip install --python "$tmp/bin/python" --dry-run -r "$1/requirements.txt" >"$log" 2>&1
  if grep -Eq "^[[:space:]]*[+~] onnxruntime(==|[<=>])" "$log"; then
    echo "dry-run proposed stock onnxruntime; see $log" >&2
    exit 1
  fi
' bash "$REPO_ROOT"

run_check "SDK venv registers VitisAI" bash -c '
  set -euo pipefail
  set +u
  # shellcheck disable=SC1090
  source "$1/venv/bin/activate"
  python - <<PY
import onnxruntime as ort
assert "VitisAIExecutionProvider" in ort.get_available_providers()
PY
' bash "$SDK_ROOT"

run_check "ROS 2 Jazzy, ros2_control and the topic bridge are installed" bash -c '
  set -euo pipefail
  root="$1"
  [[ -f /opt/ros/jazzy/setup.bash ]] || { echo "ROS 2 Jazzy is not at /opt/ros/jazzy (bootstrap section 1d)" >&2; exit 1; }
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
  for pkg in controller_manager joint_trajectory_controller forward_command_controller \
      joint_state_broadcaster robot_state_publisher xacro; do
    ros2 pkg prefix "$pkg" >/dev/null || { echo "missing ROS 2 package: $pkg" >&2; exit 1; }
  done
  bridge="$root/ros_ws/install/topic_based_ros2_control"
  [[ -d "$bridge" ]] || { echo "$bridge is missing (bootstrap section 4e builds it)" >&2; exit 1; }
' bash "$REPO_ROOT"

if [[ "$FULL" -eq 1 ]]; then
  note "repeating bootstrap (apt, llama, model downloads, and compile skipped)"
  run_check "repeat bootstrap preserves the deployment environment" bash -c '
    set -euo pipefail
    cd "$1"
    export RYZEN_AI_WHEELS="$2" RYZEN_AI_VENV="$3"
    ./bootstrap.sh --skip-apt --skip-llama --skip-models --skip-compile
  ' bash "$REPO_ROOT" "$SDK_ROOT" "$SDK_VENV"
fi

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  fail "deployment .venv exists at ${REPO_ROOT}/.venv (re-run with --full to create it)"
else
  run_check "deployment venv registers VitisAI and passes NPU preflight" bash -c '
    set -euo pipefail
    set +u
    cd "$1"
    export RYZEN_AI_WHEELS="$2" RYZEN_AI_VENV="$3"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # shellcheck disable=SC1091
    source scripts/ryzen_ai_env.sh
    python - <<PY
import importlib.metadata as metadata
import onnxruntime as ort
for name in ("onnxruntime-vitisai", "voe", "flexmlrt"):
    metadata.version(name)
providers = ort.get_available_providers()
assert "VitisAIExecutionProvider" in providers, providers
PY
    python scripts/verify_npu_stack.py --preflight
  ' bash "$REPO_ROOT" "$SDK_ROOT" "$SDK_VENV"
fi

VAIML_KEYS=(
  yolo26s_pose_fp32
  yolo26s_detect_fp32
)

run_check "existing VAIML caches are reused" bash -c '
  set -euo pipefail
  root="$1"
  shift
  for key in "$@"; do
    rai="$root/cache/$key/$key.rai"
    if [[ ! -f "$rai" ]]; then
      echo "missing $rai" >&2
      exit 1
    fi
    echo "reuse $rai"
  done
' bash "$REPO_ROOT" "${VAIML_KEYS[@]}"

if [[ -n "$WAV" ]]; then
  run_check "Whisper WAV uses VitisAI" bash -c '
    set -euo pipefail
    set +u
    cd "$1"
    export RYZEN_AI_WHEELS="$2" RYZEN_AI_VENV="$3"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # shellcheck disable=SC1091
    source scripts/ryzen_ai_env.sh
    log="$(mktemp)"
    python -m vla_pipeline.audio.whisper_npu --input "$4" --device npu >"$log" 2>&1
    status=$?
    cat "$log"
    [[ "$status" -eq 0 ]] || exit "$status"
    grep -q "Whisper active providers" "$log" || exit 1
    python - "$log" <<PY
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r"Whisper active providers.*", text)
if not match or "VitisAIExecutionProvider" not in match.group(0):
    sys.exit(1)
PY
  ' bash "$REPO_ROOT" "$SDK_ROOT" "$SDK_VENV" "$WAV"
fi

echo
printf 'Result: %d passed, %d failed\n' "$PASSES" "$FAILURES"
[[ "$FAILURES" -eq 0 ]]
