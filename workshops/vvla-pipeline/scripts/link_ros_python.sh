#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Make this repo's venv able to import ROS 2 Jazzy's Python modules.
#
# control_msgs and rclpy are not in /usr/lib. --system-site-packages does not
# add /opt/ros. Sourcing setup.bash only changes the shell that sourced it.
# Jupyter starts the venv and does not source it. A .pth file in the venv's
# own site-packages is read on every start, including a kernel whose
# kernel.json has no PYTHONPATH.
#
# The path is discovered from the installed ros-jazzy-control-msgs package.
# A hardcoded python3.12 path is skipped by site.py when that directory does
# not exist, and the import error stays exactly the same.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_DIR:-${ROOT}/.venv}"
PY="${VENV}/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "link_ros_python: no venv python at ${PY}. Run ./bootstrap.sh first." >&2
  exit 1
fi

init=""
if dpkg -L ros-jazzy-control-msgs >/dev/null 2>&1; then
  init="$(dpkg -L ros-jazzy-control-msgs | awk '/site-packages\/control_msgs\/__init__\.py$/{print; exit}')"
fi
if [[ -z "$init" ]]; then
  echo "link_ros_python: ros-jazzy-control-msgs is not installed, so control_msgs cannot be imported." >&2
  echo "  sudo apt install ros-jazzy-control-msgs ros-jazzy-controller-manager-msgs ros-jazzy-ros2-controllers" >&2
  exit 1
fi

ros_py="$(dirname "$(dirname "$init")")"
purelib="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
mkdir -p "$purelib"
printf '%s\n' "$ros_py" > "${purelib}/ros_jazzy.pth"

if ! env -u PYTHONPATH "$PY" -c 'import xacro; from control_msgs.action import FollowJointTrajectory'; then
  echo "link_ros_python: wrote ${purelib}/ros_jazzy.pth -> ${ros_py}" >&2
  echo "but this venv still cannot import control_msgs. Its sys.path is:" >&2
  env -u PYTHONPATH "$PY" -c 'import sys; print("\n".join(sys.path))' >&2
  exit 1
fi
echo "link_ros_python: ${purelib}/ros_jazzy.pth -> ${ros_py}"
