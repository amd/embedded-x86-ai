# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One twin per ROS domain: an exclusive ``flock`` the twin node holds while it runs.

Two twins on one domain would both publish the arm's topics. The node takes
the domain's lock before it starts ROS and keeps it until the process exits,
when the kernel releases it, so a second twin fails at once and a check
needs no discovery window.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path


class DomainTaken(RuntimeError):
    def __init__(self, domain_id: int, pid: int):
        super().__init__(
            f"a twin already runs on ROS domain {domain_id} (pid {pid}). Close the notebook "
            f"that runs it, or find it with: ROS_DOMAIN_ID={domain_id} ros2 node list"
        )
        self.domain_id, self.pid = domain_id, pid


def lock_path(domain_id: int) -> Path:
    return Path(tempfile.gettempdir()) / f"so101_twin_domain_{int(domain_id)}.lock"


def _holder(f) -> int:
    f.seek(0)
    text = f.read().strip()
    return int(text) if text.isdigit() else 0


def acquire(domain_id: int):
    """Take the domain's lock for the rest of this process; returns the open lock file.

    Raises :class:`DomainTaken` when another process holds it.
    """
    f = open(lock_path(domain_id), "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pid = _holder(f)
        f.close()
        raise DomainTaken(int(domain_id), pid) from None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def owner(domain_id: int) -> int | None:
    """The pid of the twin holding the domain's lock (0 if it wrote none), or None."""
    path = lock_path(domain_id)
    if not path.exists():
        return None
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _holder(f)
        fcntl.flock(f, fcntl.LOCK_UN)
        return None
