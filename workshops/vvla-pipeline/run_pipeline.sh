#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Single entry point for the Strix VLA pipeline.
#
#   ./run_pipeline.sh                     # voice control, full hardware
#   ./run_pipeline.sh --text-commands     # type commands instead of speaking
#   ./run_pipeline.sh --dry-run           # no robot hardware
#   ./run_pipeline.sh --digital-twin      # no robot hardware: MuJoCo twin of the arm,
#                                         #   shown in the twin console window
#   ./run_pipeline.sh --headless          # no preview windows
#
# Starts launch/strix_vla.launch.py in the background - the SO-101 server node
# (or the digital twin) under ros2_control when robot.use_ros2: true in
# config/pipeline.yaml, the camera nodes when a camera's use_ros2 is true - then
# runs the orchestrator. Its arm client waits for the controllers to come up.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ROS 2 underlay (optional - only needed for the ROS 2 transport)
[[ -f /opt/ros/jazzy/setup.bash ]] && source /opt/ros/jazzy/setup.bash

[[ -f .venv/bin/activate ]] || { echo "No .venv - run ./bootstrap.sh first." >&2; exit 1; }
source .venv/bin/activate

# Ryzen AI NPU runtime libs (LD_LIBRARY_PATH for the VitisAI EP).
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/ryzen_ai_env.sh"

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # SmolVLA/torch attention on ROCm
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"  # gfx1151 ROCm dispatch fix

NPU_CONFIGURED="$(python - <<'PY'
from vla_pipeline.utils.config import load_config
cfg = load_config()
sections = ("whisper", "yolo_pose", "yolo_detect")
print("yes" if any(str((cfg.get(s) or {}).get("device", "cpu")).lower() == "npu"
                   for s in sections) else "no")
PY
)"
if [[ "$NPU_CONFIGURED" == "yes" ]]; then
  python scripts/verify_npu_stack.py --preflight || {
    echo "WARNING: NPU preflight failed; models configured for the NPU will run on CPU." >&2
  }
fi

USE_ROS2="$(python - <<'PY'
from vla_pipeline.utils.config import load_config
cfg = load_config()
robot = str(cfg["robot"].get("use_ros2", False)).lower()
cams = cfg.get("cameras") or {}
cam_ros2 = any(str((cams.get(r) or {}).get("use_ros2", False)).lower() == "true"
               for r in ("mount", "arm"))
print(f"{robot} {str(cam_ros2).lower()}")
PY
)"
ROBOT_ROS2="${USE_ROS2%% *}"
CAMERA_ROS2="${USE_ROS2##* }"

LAUNCH_PID=""
cleanup() {
  # The launch runs in a session of its own: SIGINT to its process group reaches
  # every node, so the arm server parks and each node exits. The launch itself,
  # started as a background job with SIGINT ignored, ends when they have.
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

DRY=""
SIM=""
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY="--dry-run"
  [[ "$a" == "--digital-twin" ]] && SIM="--digital-twin"
done

# ---------------------------------------------------------------------------
# One-time interactive calibration. The server node runs backgrounded (no TTY),
# so a missing calibration file can't be created there - LeRobot's interactive
# calibrate() would hit EOF on input(). Detect it here and calibrate in the
# FOREGROUND (this terminal) before starting any background processes.
# ---------------------------------------------------------------------------
if [[ -z "$DRY" && -z "$SIM" ]]; then
  NEED_CAL="$(python - <<'PY'
from vla_pipeline.utils.config import load_config
from vla_pipeline.robot.arm_interface import calibration_exists
r = load_config()["robot"]
need = r.get("enable_motors", True) and not calibration_exists(r["robot_id"])
print("yes" if need else "no")
PY
)"
  if [[ "$NEED_CAL" == "yes" ]]; then
    echo "=================================================================="
    echo " No calibration found for this arm - starting calibration."
    echo " Move the arm as prompted and press ENTER. This happens once."
    echo "=================================================================="
    python -m vla_pipeline.robot.arm_interface --calibrate || {
      echo "Calibration failed or was cancelled - aborting launch." >&2
      exit 1
    }
  fi
fi

if [[ "$ROBOT_ROS2" == "true" || "$CAMERA_ROS2" == "true" ]]; then
  command -v ros2 >/dev/null || { echo "ROS 2 not found - install ROS 2 Jazzy (/opt/ros/jazzy)." >&2; exit 1; }
  # The twin hosts the arm server AND the arm camera (they share the physics).
  TWIN=false
  [[ -n "$SIM" && -z "$DRY" && "$ROBOT_ROS2" == "true" ]] && TWIN=true
  DRY_ARM=false
  [[ -n "$DRY" ]] && DRY_ARM=true
  echo "Starting the ROS 2 nodes (arm: $ROBOT_ROS2, cameras: $CAMERA_ROS2, digital twin: $TWIN)..."
  setsid python "$(command -v ros2)" launch launch/strix_vla.launch.py \
    arm:="$ROBOT_ROS2" cameras:="$CAMERA_ROS2" digital_twin:="$TWIN" dry_run:="$DRY_ARM" &
  LAUNCH_PID=$!
fi

python -m vla_pipeline.main "$@"
