# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Spin a ROS 2 executor until Ctrl-C or SIGTERM shuts its context down."""

from __future__ import annotations

import signal


def _ignore_further_sigint() -> None:
    """A second SIGINT must not kill the process while it releases its device.

    The launch forwards its own SIGINT after the one that stopped the spin.
    With the default disposition restored, that signal kills the process
    before the camera is released or the arm server closes the bus.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def spin_until_shutdown(executor) -> None:
    """``executor.spin()`` until the process is told to stop.

    rclpy's signal handler shuts the context down the moment SIGINT or
    SIGTERM arrives. A timer due at that moment still runs, and its publish
    fails on the closed context (``RCLError: publisher's context is
    invalid``). That is the same shutdown, so it ends the spin as
    ``ExternalShutdownException`` does; an error while the context is still
    up is raised.
    """
    from rclpy.executors import ExternalShutdownException

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        _ignore_further_sigint()
    except Exception:
        if executor.context.ok():
            raise
        _ignore_further_sigint()
