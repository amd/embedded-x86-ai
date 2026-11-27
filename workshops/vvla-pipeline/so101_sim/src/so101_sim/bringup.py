# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""DDS participant before any Vitis AI session.

``run_pipeline.sh`` and ``run_notebooks.sh`` both start their Python process
with this module. ``FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`` does not discover
peers when a Vitis AI session is opened first, and ``rclpy.init`` alone does
not create a participant. This module creates one, then runs the pipeline or
the notebook kernel.

The notebook kernel and the pipeline both create the participant on the
process's current ROS domain. Neither pins the private twin domain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROFILE = Path(__file__).resolve().parents[3] / "workshop" / "common" / "twin_localhost.xml"
_participant = None  # kept so the participant outlives the call
DEFAULT_DOMAIN_ID = 87  # override with TWIN_ROS_DOMAIN_ID


def twin_domain_id() -> int:
    """The ROS domain the notebook kernel and the twin use: ``TWIN_ROS_DOMAIN_ID``, else 87."""
    return int(os.environ.get("TWIN_ROS_DOMAIN_ID", DEFAULT_DOMAIN_ID))


def ensure_dds_participant(domain_id: int | None = None) -> None:
    """Create this process's DDS participant.

    ``domain_id`` set: pin that ROS domain and the localhost discovery profile.
    ``domain_id`` None: keep the process domain and its discovery. A second call
    on the same domain does nothing. ROS 2 must already be on a different domain
    only when this process was started without this module; restart it.
    """
    global _participant
    try:
        import rclpy
    except ImportError:
        return
    if domain_id is None:
        if not rclpy.ok():
            rclpy.init()
    elif rclpy.ok():
        current = rclpy.get_default_context().get_domain_id()
        if current != domain_id:
            raise RuntimeError(
                f"ROS 2 already runs in this process on domain {current}; bring-up needs "
                f"domain {domain_id}. Restart the process."
            )
    else:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST":
            del os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(_PROFILE)
        rclpy.init()
    if _participant is None:
        _participant = rclpy.create_node("dds_participant")


def main() -> None:
    """``pipeline`` runs the orchestrator; ``kernel`` runs the notebook kernel."""
    # Imported as a module, not as __main__, so the participant lives in one place.
    from so101_sim.bringup import ensure_dds_participant as start

    if len(sys.argv) < 2 or sys.argv[1] not in ("pipeline", "kernel"):
        raise SystemExit("usage: python -m so101_sim.bringup pipeline|kernel [args...]")
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "kernel":
        start(None)
        sys.argv = ["ipykernel_launcher", *rest]
        from ipykernel.kernelapp import IPKernelApp

        IPKernelApp.launch_instance()
        return
    start(None)
    sys.argv = ["vla_pipeline.main", *rest]
    from vla_pipeline.main import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
