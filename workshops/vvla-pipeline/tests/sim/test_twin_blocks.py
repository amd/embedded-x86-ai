# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pieces the notebook twin is built from, without ROS: capture stamps,
scene snapshots, and the console's camera fan-out."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.sim


@pytest.fixture
def world(workshop_cfg):
    from twin_ros_rig import _offline_world

    w = _offline_world(workshop_cfg)
    yield w
    w.close()


def test_frames_carry_the_time_their_scene_was_captured(world):
    from so101_sim.camera import SimCamera
    from so101_sim.render import RenderService

    svc = RenderService(world, require_hw=False)
    cam = SimCamera(world, "arm", renderer=svc)
    try:
        before = time.time()
        world.place("sports_ball", (0.25, 0.0, 0.015))
        ok, first = cam.read(fresh=True)
        assert ok
        assert cam.stamp_of(first) is not None and cam.stamp_of(first) >= before
        assert cam.last_stamp == cam.stamp_of(first)
        ok, second = cam.read(fresh=True)
        assert ok and cam.stamp_of(second) > cam.stamp_of(first)
        seq, frame = cam.view.latest()  # the unstamped API is unchanged
        assert isinstance(seq, int) and frame is not None
    finally:
        cam.release()
        svc.close()


def test_a_scene_snapshot_round_trips(world):
    world.place("toothbrush", (0.30, 0.10, 0.02), (0.92388, 0.0, 0.0, 0.38268))
    world.set_stand("stand_a", (0.25, 0.05), 0.12)
    snap = json.loads(json.dumps(world.snapshot_scene()))

    world.reset()
    arm_before = world.data.qpos[world._qadr].copy()
    world.data.qvel[:] = 0.3
    world.apply_scene(snap)

    for name, (qadr, dadr) in world._free.items():
        assert world.data.qpos[qadr : qadr + 7] == pytest.approx(snap["bodies"][name]), name
        assert world.data.qvel[dadr : dadr + 6] == pytest.approx(0.0), name
    mid = world.model.body("stand_a").mocapid[0]
    assert world.data.mocap_pos[mid] == pytest.approx(snap["mocap"]["stand_a"])
    assert world.data.qpos[world._qadr] == pytest.approx(arm_before), "apply_scene moved the arm"


class _Feed:
    """A camera whose frames the test hands over; stamped if ``stamped``."""

    def __init__(self, stamped: bool):
        self._cond = threading.Condition()
        self._queue: list = []
        if stamped:
            self.last_stamp = None

    def push(self, stamp=None) -> np.ndarray:
        frame = np.zeros((4, 4, 3), np.uint8)
        with self._cond:
            self._queue.append((frame, stamp))
            self._cond.notify_all()
        return frame

    def read(self):
        with self._cond:
            if not self._cond.wait_for(lambda: self._queue, timeout=0.05):
                return False, None
            frame, stamp = self._queue.pop(0)
        if hasattr(self, "last_stamp"):
            self.last_stamp = stamp
        return True, frame

    def close(self):
        pass


def test_fanout_judges_a_stamped_frame_by_its_capture_time():
    from common.twin_console import FanoutCamera

    feed = _Feed(stamped=True)
    fan = FanoutCamera(feed, timeout_s=1.0)
    try:
        settled_at = time.time()
        time.sleep(0.05)
        early = feed.push(stamp=settled_at - 0.1)  # rendered before the settle, delivered after
        late = threading.Timer(0.2, lambda: feed.push(stamp=time.time()))
        late.start()
        ok, frame = fan.read_after(settled_at)
        late.join()
        assert ok and frame is not early
    finally:
        fan.close()


def test_fanout_falls_back_to_arrival_time_without_a_stamp():
    from common.twin_console import FanoutCamera

    feed = _Feed(stamped=False)
    fan = FanoutCamera(feed, timeout_s=1.0)
    try:
        settled_at = time.time()
        time.sleep(0.05)
        frame = feed.push()
        ok, got = fan.read_after(settled_at)
        assert ok and got is frame
    finally:
        fan.close()
