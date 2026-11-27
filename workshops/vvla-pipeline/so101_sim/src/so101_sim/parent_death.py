# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A command whose processes die with the process that started it (Linux).

The notebook kernel that starts the twin and its controllers runs threads,
and ``Popen(preexec_fn=...)`` is unsafe in a threaded parent, so this
wrapper sets its own parent-death signal and runs the command as its child:

    python -m so101_sim.parent_death CMD ARGS...

The wrapper leads a process group of its own, which CMD and everything CMD
starts join. When the parent exits, the whole group gets SIGINT: a launch's
nodes as well as the launch, which loses a node it was still starting. Once
CMD has exited, whatever is left in the group is killed. CMD gets ``STOP_S``
to exit after that SIGINT; then it is killed with the rest, so the promise
does not rest on CMD stopping. A ``ros2 launch`` was seen never to exit after
SIGINT, with all its nodes gone (2026-11-25); ``so101_sim/ros_launch.py`` has
the cause and removes it for the launches this repository starts. SIGINT sent
to the group, how its owner stops it, is CMD's to act on; the wrapper waits
for CMD and exits with its status.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

# How long a stopped command gets to exit before it is killed. A launch stopped with SIGINT
# exited within 0.21 s (2026-11-25); one of its nodes that does not answer SIGINT gets
# SIGTERM from the launch after 5 s. Its owner's stop (control.stop_controllers) waits as long.
STOP_S = 10.0


def die_with_parent(sig: int = signal.SIGTERM) -> None:
    """Get ``sig`` when the process that started this one exits. It survives
    ``exec``; a parent that already exited ends this process at once."""
    import ctypes

    parent = os.getppid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, int(sig)) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    except AttributeError as exc:
        raise RuntimeError("PR_SET_PDEATHSIG is unavailable on this platform") from exc
    if os.getppid() != parent:  # the parent exited before prctl took effect
        os._exit(1)


def _group_members(pgid: int) -> list[int]:
    """The processes in process group ``pgid`` other than this one."""
    members = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = Path(entry.path, "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid:  # state, ppid, pgrp, ...
            members.append(int(entry.name))
    return members


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m so101_sim.parent_death CMD [ARGS...]")
    if os.getpgrp() != os.getpid():
        os.setpgid(0, 0)
    pgid = os.getpgrp()

    def kill_group(*_) -> None:
        for pid in _group_members(pgid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def parent_exited(*_) -> None:
        """Stop the group; when CMD has not exited STOP_S later, SIGALRM kills it."""
        os.killpg(pgid, signal.SIGINT)
        signal.setitimer(signal.ITIMER_REAL, STOP_S)

    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGALRM, kill_group)
    signal.signal(signal.SIGTERM, parent_exited)
    die_with_parent(signal.SIGTERM)
    code = subprocess.Popen(sys.argv[1:]).wait()
    signal.setitimer(signal.ITIMER_REAL, 0)  # CMD has exited: what is left is killed now
    kill_group()
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
