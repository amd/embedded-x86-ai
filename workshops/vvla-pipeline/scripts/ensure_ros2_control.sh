#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Everything ros2_control_arm.py needs, done by the setup scripts:
# the apt packages (control_msgs is not in ros-jazzy-desktop), a check that
# librcl_action.so loads, the venv .pth, and the topic_based_ros2_control plugin.
# A machine that only git-pulls never re-runs bootstrap, so the notebook and
# pipeline launchers call this. It does nothing when those pieces are already
# present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
ROS_WS="${ROOT}/ros_ws"
BRIDGE_SRC="${ROS_WS}/src/topic_based_ros2_control"
BRIDGE_PREFIX="${ROS_WS}/install/topic_based_ros2_control"
TOPIC_BRIDGE_REPO="${TOPIC_BRIDGE_REPO:-https://github.com/PickNikRobotics/topic_based_ros2_control.git}"
TOPIC_BRIDGE_REF="${TOPIC_BRIDGE_REF:-0.3.0}"
PACKAGES=(
  ros-jazzy-ros2-control
  ros-jazzy-ros2-controllers
  ros-jazzy-control-msgs
  ros-jazzy-controller-manager-msgs
  ros-jazzy-xacro
  ros-dev-tools
)

missing=()
for pkg in "${PACKAGES[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  if [[ "${SKIP_APT:-0}" == "1" ]]; then
    echo "ensure_ros2_control: missing ${missing[*]}. Re-run ./bootstrap.sh without --skip-apt." >&2
    exit 1
  fi
  sudo apt install -y "${missing[@]}"
fi

# Every process this repo starts sources ROS 2 first, which puts
# /opt/ros/jazzy/lib on LD_LIBRARY_PATH. That is what makes librcl_action.so
# load. ldconfig's cache is a second copy of the same fact, and writing it
# needs root, which this account does not have.
if [[ ! -f /opt/ros/jazzy/lib/librcl_action.so ]]; then
  echo "ensure_ros2_control: /opt/ros/jazzy/lib/librcl_action.so is missing." >&2
  exit 1
fi
if ! LD_LIBRARY_PATH="/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    /usr/bin/python3 -c 'import ctypes; ctypes.CDLL("librcl_action.so")'; then
  echo "ensure_ros2_control: librcl_action.so does not load from /opt/ros/jazzy/lib." >&2
  exit 1
fi

VENV_DIR="$VENV_DIR" "${ROOT}/scripts/link_ros_python.sh"

if [[ -d "$BRIDGE_PREFIX" ]]; then
  exit 0
fi

mkdir -p "${ROS_WS}/src"
if [[ ! -d "${BRIDGE_SRC}/.git" ]]; then
  git clone "$TOPIC_BRIDGE_REPO" "$BRIDGE_SRC"
fi
git -C "$BRIDGE_SRC" fetch --tags --quiet
git -C "$BRIDGE_SRC" checkout --quiet --force "$TOPIC_BRIDGE_REF"
for patch in "${ROOT}"/patches/topic_based_ros2_control/*.patch; do
  if git -C "$BRIDGE_SRC" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$BRIDGE_SRC" apply "$patch"
  elif git -C "$BRIDGE_SRC" apply --reverse --check "$patch" >/dev/null 2>&1; then
    :
  else
    echo "ensure_ros2_control: cannot apply $(basename "$patch")" >&2
    exit 1
  fi
done

# colcon must use the system Python ROS was built with, not the venv.
build_path="$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv/bin$' | paste -sd: -)"
(
  unset VIRTUAL_ENV
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
  cd "$ROS_WS"
  PATH="$build_path" colcon build --packages-select topic_based_ros2_control \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
)
[[ -d "$BRIDGE_PREFIX" ]] || {
  echo "ensure_ros2_control: ${BRIDGE_PREFIX} was not produced. See ${ROS_WS}/log/latest_build." >&2
  exit 1
}
echo "ensure_ros2_control: ${BRIDGE_PREFIX}"
