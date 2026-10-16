# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Test rigs: the real pipeline / workshop stacks wired to the digital twin.

A rig owns one scenario: it builds the stack's own context (arm transport,
cameras, models, LLM client, dispatcher) with only the sensors swapped for
the scripted sources, records joint/goal/current traces for assertions, and
films the run with :class:`so101_sim.SimRecorder`.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from scripted import (
    Clock,
    CountingDetector,
    ScriptedOperatorCamera,
    ScriptedTranscripts,
    SignalBoard,
    SimDetector,
    TimelineHands,
    TimelinePose,
)

logger = logging.getLogger(__name__)

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_GRIPPER = MOTORS.index("gripper")

HOLD_WATCH_S = 5.0  # how long a picked object is watched after the robot says "Got the ..."
SLIP_TOL_M = 0.002  # how far it may move in the gripper frame once it is off the table
LIFT_SLIP_TOL_M = 0.004  # how far while the fingers take its weight (the pads give a little)
CARRY_TOL_M = 0.01  # object rise vs hand rise; the wrist tilts a little on the way up
_LIFT_SETTLE_S = 0.3  # the object takes its weight on the fingers just after lift-off
_GOAL_TOL = 0.5  # gripper goal units; a carry keeps one goal from the lift to the end


@dataclass
class Sample:
    t: float
    joints: dict
    goal: dict
    current: float  # gripper Present_Current (raw counts)
    grasped: dict = field(default_factory=dict)
    heights: dict = field(default_factory=dict)
    in_hand: dict = field(default_factory=dict)  # body -> position in the gripper frame (m)
    hand_z: float = 0.0  # gripper body height (m)
    overload: bool = False  # the gripper's overload protection has cut its torque


class Trace:
    """Samples the twin at ``hz`` on a thread (scenario clock)."""

    def __init__(self, world, clock: Clock, hz: float = 50.0, watch=()):
        self.world, self.clock, self.hz = world, clock, hz
        self.watch = tuple(watch)  # object bodies to track
        self.samples: list[Sample] = []
        self._run = False
        self._thread = None

    def start(self) -> None:
        if self._run:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, name="trace", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._run = False
        if self._thread is not None:
            self._thread.join(2.0)

    def _loop(self) -> None:
        w = self.world
        while self._run:
            snap = w.snapshot()
            cur = snap["current_a"]["gripper"] * 1000.0 / w.current.ma_per_unit
            grasped = {b: w.is_grasped(b) for b in self.watch}
            heights = {b: float(w.body_pos(b)[2]) for b in self.watch}
            in_hand = {b: w.in_gripper(b) for b in self.watch}
            hand_z = float(w.body_pos("gripper")[2])
            overload = bool(w.protection.tripped[_GRIPPER])
            self.samples.append(
                Sample(self.clock(), snap["joints"], snap["goal"], cur, grasped, heights, in_hand, hand_z,
                       overload)
            )
            time.sleep(1.0 / self.hz)

    # -- queries ------------------------------------------------------------
    def window(self, t0: float = 0.0, t1: float = float("inf")) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]

    def series(self, joint: str, t0: float = 0.0, t1: float = float("inf"), goal: bool = False):
        win = self.window(t0, t1)
        key = "goal" if goal else "joints"
        return np.array([s.t for s in win]), np.array([getattr(s, key)[joint] for s in win])

    def held_current(self, hold_s: float = 0.3) -> float:
        """Highest gripper current held for ``hold_s``: a squeeze keeps it up,
        the jaw's own moves only spike it."""
        best = 0.0
        for i, s in enumerate(self.samples):
            win = [x.current for x in self.samples[i:] if x.t <= s.t + hold_s]
            if self.samples[-1].t - s.t >= hold_s:
                best = max(best, min(win))
        return best

    def heights(self, body: str) -> np.ndarray:
        return np.array([s.heights[body] for s in self.samples])

    def grasped_seconds(self, body: str) -> float:
        """Longest continuous stretch with ``body`` held between both jaws."""
        best = run = 0.0
        prev = None
        for s in self.samples:
            if s.grasped.get(body):
                run += 0.0 if prev is None else s.t - prev
                best = max(best, run)
            else:
                run = 0.0
            prev = s.t
        return best

    def lift_off(self, body: str, z0: float, dz: float = 0.005) -> float | None:
        """First time ``body`` is ``dz`` above its resting height ``z0``."""
        for s in self.samples:
            if s.heights[body] > z0 + dz:
                return s.t
        return None

    def grasp_start(self, body: str, at: float) -> float | None:
        """Start of the unbroken stretch, up to ``at``, with ``body`` between both fingers."""
        start = None
        for s in self.window(0.0, at):
            if s.grasped.get(body):
                start = s.t if start is None else start
            else:
                start = None
        return start

    def hand_rise(self, t0: float, t1: float, dz: float = 0.0005) -> float | None:
        """Last time in ``[t0, t1]`` that the hand was at its lowest: the lift
        starts right after it."""
        win = self.window(t0, t1)
        if not win:
            return None
        low = min(s.hand_z for s in win)
        return max(s.t for s in win if s.hand_z <= low + dz)

    def first_overload(self) -> float | None:
        """When the gripper's overload protection first cut its torque."""
        return next((s.t for s in self.samples if s.overload), None)

    def goal_span(self, joint: str, t0: float, t1: float = float("inf")) -> float:
        """How far the commanded ``joint`` goal ranged over the window."""
        goals = [s.goal[joint] for s in self.window(t0, t1)]
        return max(goals) - min(goals) if goals else 0.0

    def drift(self, body: str, t0: float, t1: float = float("inf")) -> float:
        """Largest move (m) of ``body`` in the gripper frame away from where it
        was at ``t0``. A held object that does not slip stays at ~0."""
        win = self.window(t0, t1)
        if not win:
            return float("inf")
        ref = win[0].in_hand[body]
        return max(float(np.linalg.norm(s.in_hand[body] - ref)) for s in win)

    def longest_release(self, body: str, t0: float, t1: float = float("inf")) -> float:
        """Longest stretch (s) in the window with ``body`` not between both fingers."""
        best = 0.0
        since = None
        for s in self.window(t0, t1):
            if s.grasped.get(body):
                since = None
            else:
                since = s.t if since is None else since
                best = max(best, s.t - since)
        return best


def hold_problem(rig, body: str, z0: float, watch_s: float = HOLD_WATCH_S,
                 slip_m: float = SLIP_TOL_M, lift_slip_m: float = LIFT_SLIP_TOL_M,
                 until: float | None = None) -> str | None:
    """Why ``body`` was not held still from the lift to the end of the run
    (or ``until``, e.g. a hand-over opening the jaws), or ``None``.

    Held still means: between both fingers the whole time; in the gripper
    frame, within ``lift_slip_m`` of where it sat when the hand started to
    rise until just after it left the table, and within ``slip_m`` of that
    spot from then on; a gripper goal that never moves while it is carried;
    and no overload trip. The run must go on ``watch_s`` past "Got the ...".
    """
    tr = rig.trace
    lift = tr.lift_off(body, z0)
    if lift is None:
        return f"{body} never left the table"
    end = tr.samples[-1].t if until is None else until
    got = [t for t, e, d in rig.events if e == "say" and "got the" in str(d.get("text", "")).lower()]
    if got and end - got[-1] < watch_s - 0.1:
        return f"the run ended {end - got[-1]:.1f} s after 'Got the ...', not {watch_s:.0f} s"
    grip = tr.grasp_start(body, lift)
    if grip is None:
        return f"{body} was not between both fingers when it left the table"
    rise = tr.hand_rise(grip, lift)
    t0 = lift + _LIFT_SETTLE_S
    seated = tr.drift(body, rise, t0)
    gap = tr.longest_release(body, rise, end)
    moved = tr.drift(body, t0, end)
    loosened = tr.goal_span("gripper", lift, end)
    logger.info(
        "hold check: %s moved %.2f mm taking its weight, then drifted %.2f mm; "
        "gripper goal ranged %.1f; longest release %.2f s, watched %.1f s after lift-off",
        body, 1000 * seated, 1000 * moved, loosened, gap, end - lift,
    )
    tripped = tr.first_overload()
    if tripped is not None and tripped <= end:
        return (
            f"the gripper's overload protection cut the grip to Protective_Torque "
            f"{tripped - lift:+.1f} s from lift-off"
        )
    if gap > 0.1:
        return f"{body} was out of the fingers for {gap:.2f} s after the lift started"
    if loosened > _GOAL_TOL:
        return f"the gripper goal moved {loosened:.1f} while it carried the {body}"
    if seated > lift_slip_m:
        return (
            f"{body} slid {1000 * seated:.1f} mm in the gripper while it was lifted "
            f"(tolerance {1000 * lift_slip_m:.1f} mm)"
        )
    if moved > slip_m:
        return (
            f"{body} slipped {1000 * moved:.1f} mm in the gripper after lift-off "
            f"(tolerance {1000 * slip_m:.1f} mm)"
        )
    return None


def carry_problem(rig, body: str, z0: float, tol: float = CARRY_TOL_M) -> str | None:
    """Why ``body`` did not rise as far as the hand that picked it up, or ``None``.

    Measured from where the hand gripped it, so a tall bottle held high up
    its body is judged like a ball.
    """
    tr = rig.trace
    lift = tr.lift_off(body, z0)
    if lift is None:
        return f"{body} never left the table"
    grip = tr.grasp_start(body, lift)
    if grip is None:
        return f"{body} was not between both fingers when it left the table"
    win = tr.window(tr.hand_rise(grip, lift))
    hand = max(s.hand_z for s in win) - win[0].hand_z
    obj = max(s.heights[body] for s in win) - z0
    logger.info("carry check: %s rose %.1f mm, the hand %.1f mm after picking it up",
                body, 1000 * obj, 1000 * hand)
    if abs(obj - hand) > tol:
        return f"{body} rose {1000 * obj:.0f} mm, the hand {1000 * hand:.0f} mm after picking it up"
    return None


def assert_ordered_high_scan(caplog, pans, *, allow_near=False):
    """Logged pans are an ordered prefix of the list, and the reach follows them."""
    text = caplog.text
    at = []
    for pan in pans:
        token = f"scan pan {float(pan):.1f}"
        i = text.find(token)
        if i < 0:
            break
        at.append(i)
    assert at, "the arm never scanned"
    assert at == sorted(at), "scan pans were not visited in order"
    for pan in pans[len(at):]:
        assert f"scan pan {float(pan):.1f}" not in text
    solved = text.find("-> solved")
    assert solved > at[-1], "the reach started before the scan stopped"
    near = text.find("scan near pan")
    if allow_near:
        assert near < 0 or near > at[-1]
    else:
        assert near < 0, "the high sweep already stored a view"


class _Rig:
    """Common scenario plumbing; subclasses build the stack's context."""

    target = "?"

    def __init__(self, twin, cfg: dict, recorder_path=None, title: str = ""):
        self.world = twin.world
        self.renderer = twin.renderer
        self.cfg = copy.deepcopy(cfg)
        self.clock = Clock()
        self.board = SignalBoard()
        self.events: list[tuple[float, str, dict]] = []
        self.said: list[str] = []
        self.pose = TimelinePose(self.clock)
        self.hands = TimelineHands(self.clock, self._hand_obs_cls())
        self.operator = ScriptedOperatorCamera(self.pose, self.hands, self.board)
        self.world.reset()
        for obj in self.world.objects:
            self.world.hide(obj)
        self.trace = Trace(self.world, self.clock, watch=self.world.objects)
        self.recorder = None
        if recorder_path is not None:
            from so101_sim.record import SimRecorder

            self.recorder = SimRecorder(
                self.world, recorder_path, renderer=self.renderer,
                user_source=self.operator.signals_pane, user_label="scripted user signals",
                title=title, load_threshold=self._load_threshold(),
            ).start()
        self._closers: list = []
        self._placed: dict[str, np.ndarray] = {}

    # -- hooks ----------------------------------------------------------------
    def _hand_obs_cls(self):
        raise NotImplementedError

    def _load_threshold(self) -> float:
        return 120.0

    # -- events ---------------------------------------------------------------
    def on_event(self, event: str, **data) -> None:
        self.events.append((self.clock(), event, data))
        self.board.on_event(event, **data)
        if self.recorder is not None:
            self.recorder.event(event, **data)
            if event == "behavior_start":
                self.recorder.label(f"behavior: {data.get('intent')}")
            elif event in ("behavior_end", "handled"):
                self.recorder.label("idle")

    def say(self, message: str) -> None:
        self.said.append(message)
        logger.info("[say] %s", message)
        self.on_event("say", text=message)

    def event_times(self, name: str, **match) -> list[float]:
        return [t for t, e, d in self.events if e == name and all(d.get(k) == v for k, v in match.items())]

    def said_any(self, phrase: str) -> bool:
        return any(phrase.lower() in m.lower() for m in self.said)

    # -- scene ------------------------------------------------------------------
    def stage(self, obj: str, pose: dict, stand: str = "stand_a"):
        p = self.world.stage_object(obj, {k: v for k, v in pose.items() if k != "gripper"}, stand=stand)
        self._placed[obj] = np.array(p, float)
        self.on_event("stage", object=obj, at=[round(float(v), 3) for v in p])
        return p

    def default_scene(self) -> None:
        """Every prop back at its XML spot - the scene the notebook console shows."""
        for obj in self.world.objects:
            self.put(obj)

    def _eyes(self, detector, detection_cls, resolve_class) -> None:
        """Arm camera + detector: the real YOLO on the rendered wrist image when
        ``detector`` is given (as in the notebook), else ground-truth boxes from
        the twin's segmentation."""
        from so101_sim.camera import SimCamera
        from so101_sim.detect import SegDetector

        self.seg_camera = SimCamera(
            self.world, "arm", renderer=self.renderer, segmentation=detector is None
        )
        self._closers.append(self.seg_camera.release)
        if detector is None:
            self.detector = SimDetector(SegDetector(self.seg_camera), detection_cls, resolve_class)
        else:
            self.detector = CountingDetector(detector, resolve_class)

    def put(self, obj: str, pos=None) -> np.ndarray:
        """Put a hidden object back on the table: at its XML spot, or at ``pos``."""
        p = np.array(self.world.model.body(obj).pos if pos is None else pos, float)
        self.world.place(obj, p)
        self._placed[obj] = p.copy()
        self.on_event("stage", object=obj, at=[round(float(v), 3) for v in p])
        return p

    def disturbed(self, *handled: str, tol: float = 0.01) -> dict[str, float]:
        """Props on the table that moved more than ``tol`` (m) from where they
        were put, ``handled`` (picked up or dropped on purpose) excepted."""
        moved = {
            obj: float(np.linalg.norm(self.world.body_pos(obj) - p))
            for obj, p in self._placed.items() if obj not in handled
        }
        return {obj: round(d, 3) for obj, d in moved.items() if d > tol}

    def wait_until(self, pred, timeout: float, poll: float = 0.02) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(poll)
        return False

    def joints(self) -> dict:
        return self.world.joint_cal()

    def near(self, pose: dict, tol: float = 3.0, joints=None) -> bool:
        cur = self.world.joint_cal()
        keys = joints or [k for k in pose if k != "gripper"]
        return all(abs(cur[k] - pose[k]) <= tol for k in keys)

    def when_near(self, pose: dict, action, tol: float = 3.0, delay: float = 0.0, timeout: float = 20.0):
        """Run ``action()`` on a thread once the arm reaches ``pose`` (+ ``delay``)."""

        def run():
            if self.wait_until(lambda: self.near(pose, tol), timeout):
                time.sleep(delay)
                action()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def when(self, pred, action, timeout: float = 60.0):
        """Run ``action()`` on a thread once ``pred()`` holds."""

        def run():
            if self.wait_until(pred, timeout):
                action()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def when_lifted(self, obj: str, z0: float, action, dz: float = 0.03, timeout: float = 60.0):
        """Run ``action()`` on a thread once ``obj`` is ``dz`` above ``z0``."""
        return self.when(lambda: self.world.body_pos(obj)[2] > z0 + dz, action, timeout)

    def slicken(self, obj: str, mu: float = 0.01) -> None:
        """Make ``obj`` slide out of any grip; :meth:`finish` restores it.

        MuJoCo takes the larger friction of two touching geoms unless one has
        the higher priority, so the object's geoms get that too.
        """
        w = self.world
        body = w.model.body(obj)
        geoms = range(int(body.geomadr[0]), int(body.geomadr[0] + body.geomnum[0]))
        with w.lock:
            saved = [(g, w.model.geom_friction[g].copy(), int(w.model.geom_priority[g])) for g in geoms]
            for g, _, _ in saved:
                w.model.geom_friction[g, 0] = mu
                w.model.geom_priority[g] = 10

        def restore():
            with w.lock:
                for g, friction, priority in saved:
                    w.model.geom_friction[g] = friction
                    w.model.geom_priority[g] = priority

        self._closers.append(restore)
        self.on_event("slicken", object=obj, mu=mu)

    # -- lifecycle --------------------------------------------------------------
    def finish(self, status: str, message: str = "", keep: bool = True):
        self.trace.stop()
        path = None
        if self.recorder is not None:
            self.recorder.stamp(status, message)
            path = self.recorder.stop(keep=keep)
        for close in reversed(self._closers):
            try:
                close()
            except Exception as e:  # pragma: no cover
                logger.debug("rig close: %s", e)
        return path


def _llm_host_port(base: str) -> tuple[str, int]:
    host, port = base.rsplit("//", 1)[1].split(":")
    return host, int(port)


# ---------------------------------------------------------------------------
# pipeline (vla_pipeline)
# ---------------------------------------------------------------------------
class PipelineRig(_Rig):
    """``vla_pipeline``: SimArmClient + real Dispatcher/registry/LLM client."""

    target = "pipeline"

    def _hand_obs_cls(self):
        from vla_pipeline.vision.mediapipe_hands import HandObservation

        return HandObservation

    def _load_threshold(self) -> float:
        return float(self.cfg["behaviors"]["grip"].get("load_threshold", 120.0))

    def __init__(self, twin, cfg: dict, llm_base: str, recorder_path=None, title: str = "",
                 arm=None, arm_camera=None, detector=None):
        super().__init__(twin, cfg, recorder_path, title)
        from vla_pipeline.llm.llama_intent import LlamaIntentParser
        from vla_pipeline.main import PipelineContext
        from vla_pipeline.robot.sim_arm import SimArmClient
        from vla_pipeline.vision.yolo_detect_npu import Detection, resolve_class

        host, port = _llm_host_port(llm_base)
        self.cfg["llm"].update(host=host, port=port, autostart=False)
        self.cfg["robot"]["backend"] = "sim"
        self.arm = arm if arm is not None else SimArmClient(self.cfg, world=self.world)
        self._eyes(detector, Detection, resolve_class)
        self.arm_camera = arm_camera if arm_camera is not None else self.seg_camera
        self.parser = LlamaIntentParser(self.cfg)
        self._closers.append(self.parser.close)
        self.ctx = PipelineContext(
            cfg=self.cfg, arm=self.arm, pose=self.pose, detector=self.detector, hands=self.hands,
            operator_camera=self.operator, arm_camera=self.arm_camera, headless=True,
        )
        self.ctx.say = self.say
        self.dispatcher = None

    def run(self, script, timeout: float = 30.0, settle: float = 0.6, startup_grace_s: float = 0.0,
            until=None) -> "PipelineRig":
        """Play ``script`` through the real Dispatcher; return once idle (or ``until()``)."""
        from vla_pipeline.dispatch import Dispatcher
        from vla_pipeline.main import is_stop, make_noise_gate, registry

        self.clock.start()
        self.trace.start()
        d = Dispatcher(
            self.ctx, self.parser.parse, registry, is_stop=is_stop,
            is_noise=make_noise_gate(self.cfg), startup_grace_s=startup_grace_s, clock=self.clock,
        )
        d.listeners.append(self.on_event)
        self.dispatcher = d
        shutdown = threading.Event()
        loop = threading.Thread(target=d.run, args=(shutdown,), name="dispatch", daemon=True)
        loop.start()
        tx = ScriptedTranscripts(self.clock, script, d.on_transcript).start()
        self.transcripts = tx

        idle_since = None
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"scenario did not finish within {timeout}s")
                if until is not None and until():
                    break
                idle = tx.done.is_set() and d.transcripts.empty() and not d.busy.is_set()
                if idle:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= settle:
                        break
                else:
                    idle_since = None
                time.sleep(0.02)
        finally:
            self.ctx.stop_event.set()
            shutdown.set()
            loop.join(15.0)
            self.ctx.stop_event.clear()
        return self


# ---------------------------------------------------------------------------
# workshop (workshop/<project>)
# ---------------------------------------------------------------------------
class WorkshopRig(_Rig):
    """Workshop: common.motion.SimArmClient + integration.handle_transcript."""

    target = "workshop"

    def _hand_obs_cls(self):
        from models.hands import HandObservation

        return HandObservation

    def __init__(self, twin, cfg: dict, llm_base: str, recorder_path=None, title: str = "", arm=None,
                 detector=None):
        super().__init__(twin, cfg, recorder_path, title)
        from common.imaging import Detection, resolve_class
        from common.motion import SimArmClient, mapper_from_config
        from integration.dispatch import WorkshopContext, build_registry
        from models.llm import LlamaIntentParser

        host, port = _llm_host_port(llm_base)
        self.cfg["llm"].update(host=host, port=port, autostart=False)
        self.cfg["robot"]["backend"] = "sim"
        self.arm = arm if arm is not None else SimArmClient(self.cfg, world=self.world)
        self._eyes(detector, Detection, resolve_class)
        self.parser = LlamaIntentParser(self.cfg, autostart=False)
        self._closers.append(self.parser.close)
        self.ctx = WorkshopContext(
            cfg=self.cfg, arm=self.arm, pose=self.pose, hands=self.hands, camera=self.operator,
            mapper=mapper_from_config(self.cfg), registry=build_registry(), headless=True,
            detector=self.detector, arm_camera=self.seg_camera,
        )
        self.ctx.say = self.say

    def _deliver(self, text: str) -> None:
        from common.intents import Intent
        from integration.dispatch import handle_transcript

        self.on_event("heard", text=text)

        def parse(t):
            cmd = self.parser.parse(t)
            self.on_event("parsed", text=t, intent=cmd.intent.value, params=dict(cmd.params or {}))
            if cmd.intent not in (Intent.UNKNOWN, Intent.STOP) and self.ctx.registry.get(cmd.intent):
                self.on_event("behavior_start", intent=cmd.intent.value)
            return cmd

        try:
            cmd = handle_transcript(self.ctx, text, parse)
        except Exception as e:
            self.on_event("behavior_crash", error=repr(e))
            raise
        if cmd.intent is Intent.STOP:
            self.on_event("stop", text=text)
        self.on_event("handled", text=text, intent=cmd.intent.value)

    def run(self, script, timeout: float = 30.0, settle: float = 0.3) -> "WorkshopRig":
        self.clock.start()
        self.trace.start()
        tx = ScriptedTranscripts(self.clock, script, self._deliver, concurrent=True).start()
        self.transcripts = tx
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"scenario did not finish within {timeout}s")
                if tx.done.is_set() and not any(w.is_alive() for w in tx.workers):
                    break
                time.sleep(0.02)
            time.sleep(settle)
        finally:
            self.ctx.stop_event.set()
            tx.join(10.0)
            self.ctx.stop_event.clear()
        return self
