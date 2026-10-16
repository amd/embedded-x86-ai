# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""SimWorld.reset is MuJoCo's mj_resetData, plus the arm pose it was asked to hold.

mj_resetData sets qpos to the compiled qpos0, restores mocap to the compiled
body poses, and clears every other state and control value. The test dirties
all of that, resets, and compares the result with a second MjData that was
only mj_resetData'd.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.sim


@pytest.fixture
def world():
    from so101_sim.world import SimWorld

    sim = SimWorld(start=False)
    yield sim
    sim.close()


def _state(model, data, sig: int) -> np.ndarray:
    import mujoco

    out = np.zeros(mujoco.mj_stateSize(model, sig))
    mujoco.mj_getState(model, data, out, sig)
    return out


def test_reset_matches_mj_resetData(world):
    """A disturbed simulation comes back to the compiled state."""
    import mujoco

    from so101_sim import MOTOR_NAMES

    model, data = world.model, world.data
    # Dirty every field mj_resetData documents: configuration, velocity,
    # mocap, applied forces, ctrl, then step so time, contacts and the
    # constraint warmstart are nonzero too.
    for name in world.objects:
        world.place(name, (0.35, 0.15, 0.25), (0.5, 0.5, 0.5, 0.5))
        _qadr, dadr = world._free[name]
        data.qvel[dadr : dadr + 6] = (0.4, -0.3, 0.2, 1.5, -0.8, 0.6)
    world.set_stand("stand_a", (0.05, 0.05), 0.25)
    world.set_stand("stand_b", (-0.05, -0.05), 0.25)
    # Hold the arm while the props fall. Driving every actuator to 0.4 rad
    # from the park pose throws the props clear of every contact, so the
    # dirty ctrl and applied forces are written after the step.
    data.ctrl[:] = data.qpos[world._qadr]
    world.advance(0.4)
    data.ctrl[:] = 0.4
    data.qfrc_applied[:] = 0.7
    data.xfrc_applied[:] = 0.3

    brush = world._free["toothbrush"][0]
    assert data.time > 0
    assert not np.allclose(data.qpos[brush : brush + 7], model.qpos0[brush : brush + 7])
    assert data.ncon > 0

    from so101_sim.kinematics import OPEN_GRIPPER

    pose = {**world.rest_pose, "gripper": OPEN_GRIPPER}
    world.reset(pose)

    ref = mujoco.MjData(model)
    mujoco.mj_resetData(model, ref)
    # The only qpos reset is allowed to change after mj_resetData is the arm.
    ref.qpos[world._qadr] = data.qpos[world._qadr]
    mujoco.mj_forward(model, ref)

    for name, (qadr, dadr) in world._free.items():
        assert data.qpos[qadr : qadr + 7] == pytest.approx(model.qpos0[qadr : qadr + 7]), name
        assert data.qvel[dadr : dadr + 6] == pytest.approx(0), name
        assert data.body(name).xpos == pytest.approx(ref.body(name).xpos), name
        assert data.body(name).xquat == pytest.approx(ref.body(name).xquat), name

    assert data.time == 0
    assert data.qvel == pytest.approx(0)
    assert data.qfrc_applied == pytest.approx(0)
    assert data.xfrc_applied == pytest.approx(0)
    if model.na:
        assert data.act == pytest.approx(0)
    assert data.mocap_pos == pytest.approx(ref.mocap_pos)
    assert data.mocap_quat == pytest.approx(ref.mocap_quat)

    cleared = (
        int(mujoco.mjtState.mjSTATE_TIME)
        | int(mujoco.mjtState.mjSTATE_QVEL)
        | int(mujoco.mjtState.mjSTATE_ACT)
        | int(mujoco.mjtState.mjSTATE_WARMSTART)
        | int(mujoco.mjtState.mjSTATE_QFRC_APPLIED)
        | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
        | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    )
    assert _state(model, data, cleared) == pytest.approx(_state(model, ref, cleared))

    for i, name in enumerate(MOTOR_NAMES):
        q = world.jmap.to_rad(name, pose[name])
        lo, hi = world._jrange[i]
        q = min(max(q, lo), hi)
        assert data.qpos[world._qadr[i]] == pytest.approx(q), name
        assert data.ctrl[world._act[i]] == pytest.approx(q), name
    assert world.servo.v_ref == pytest.approx(0)
    assert world.servo.goal == pytest.approx(data.qpos[world._qadr])

    # A cleared warmstart must not throw the props off qpos0 on the next steps.
    world.advance(0.5)
    for name, (qadr, _dadr) in world._free.items():
        assert data.qpos[qadr : qadr + 3] == pytest.approx(model.qpos0[qadr : qadr + 3], abs=1e-3), name
        quat = data.qpos[qadr + 3 : qadr + 7]
        home = model.qpos0[qadr + 3 : qadr + 7]
        assert abs(abs(float(np.dot(quat, home))) - 1.0) < 1e-3, name
