#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# ---------------------------------------------------------------------------
# launch_monitor.sh -- open the always-on-top consolidated resource HUD.
#
# Shows only:  CPU %  (burgundy)  ·  GPU %  (orange)  ·  NPU inf/s | Idle (cyan)
#
# Run it from a terminal or double-click it in a file manager. Close the HUD
# from its  ✕ , the Esc key, or a right-click. Run again to re-open it.
# From a notebook, use instead:  from common.monitor import open_monitor
# ---------------------------------------------------------------------------
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/common/resource_hud.py" "$@"
