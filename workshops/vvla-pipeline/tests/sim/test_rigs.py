# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Rig plumbing that every scenario relies on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim


def test_a_behavior_crash_fails_the_scenario():
    from rigs import _Rig

    rig = SimpleNamespace(events=[(0.1, "say", {"text": "Released."}),
                                  (0.2, "behavior_crash", {"error": "MotionTimeout('servos still moving')"})])
    with pytest.raises(AssertionError, match="MotionTimeout"):
        _Rig._raise_crashes(rig)


def test_a_scenario_without_a_crash_passes():
    from rigs import _Rig

    _Rig._raise_crashes(SimpleNamespace(events=[(0.1, "handled", {"text": "drop it", "intent": "release"})]))
