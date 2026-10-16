# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Scripted stand-ins for the user-facing signals of the real interface.

Each one replaces exactly one model/sensor *behind its real interface*, so
the behaviors, dispatcher and transports under test run unchanged:

``TimelinePose``        YOLO-pose ``infer`` / ``primary_person`` (keyframed people)
``TimelineHands``       MediaPipe ``HandTracker.process`` (keyframed pinch/roll/size)
``ScriptedOperatorCamera`` the operator camera: paced frames showing the scripted
                        person, plus the "signals" pane for the test video
``SimDetector``         YOLO ``ObjectDetector`` from the twin's segmentation
``ScriptedTranscripts`` Whisper: timed utterances into the dispatcher

All timelines share one clock (``Clock``) started when the scenario starts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np


class Clock:
    """Scenario time in seconds since ``start()`` (0 before)."""

    def __init__(self):
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.monotonic()

    def __call__(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0


def _interp(keys: list, t: float, blend):
    """Piecewise-linear over ``[(t, value), ...]``; ``blend(a, b, u)`` or step if None."""
    if not keys:
        return None
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            out = blend(a, b, (t - t0) / max(t1 - t0, 1e-9))
            return a if out is None else out
    return keys[-1][1]


# ---------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Person:
    """One scripted person, in frame fractions. ``scale`` ~ nearness (box size)."""

    cx: float = 0.5
    cy: float = 0.45
    scale: float = 1.0
    wrist: tuple = (0.65, 0.45)  # right wrist (the control signal)
    wrist_vis: float = 0.95

    def keypoints(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """``(box[4], kpts[17, 3])`` in pixels - the ScriptedPose body layout."""
        s = self.scale
        cx, cy = self.cx * w, self.cy * h
        k = np.zeros((17, 3), np.float32)

        def put(i, dx, dy, v=0.9):
            k[i] = (cx + dx * w * s, cy + dy * h * s, v)

        put(0, 0.0, -0.18)
        put(1, -0.02, -0.20)
        put(2, 0.02, -0.20)
        put(3, -0.045, -0.185)
        put(4, 0.045, -0.185)
        put(5, -0.09, -0.05)
        put(6, 0.09, -0.05)
        put(11, -0.07, 0.22)
        put(12, 0.07, 0.22)
        put(7, -0.14, 0.08)
        put(9, -0.18, 0.18, 0.4)
        put(13, -0.07, 0.45)
        put(14, 0.07, 0.45)
        put(15, -0.08, 0.66)
        put(16, 0.08, 0.66)
        wx, wy = self.wrist[0] * w, self.wrist[1] * h
        sx, sy = k[6, 0], k[6, 1]
        k[8] = ((sx + wx) / 2 + 0.03 * w * s, (sy + wy) / 2 + 0.06 * h * s, 0.9)  # right elbow
        k[10] = (wx, wy, self.wrist_vis)
        box = np.array(
            [cx - 0.2 * w * s, cy - 0.25 * h * s, cx + 0.2 * w * s, cy + 0.72 * h * s], np.float32
        )
        return box, k


def _blend_people(a, b, u):
    if len(a) != len(b):
        return None  # people appear/leave: step change
    out = []
    for p, q in zip(a, b):
        mix = lambda x, y: x + (y - x) * u  # noqa: E731
        out.append(
            replace(
                p,
                cx=mix(p.cx, q.cx),
                cy=mix(p.cy, q.cy),
                scale=mix(p.scale, q.scale),
                wrist=(mix(p.wrist[0], q.wrist[0]), mix(p.wrist[1], q.wrist[1])),
                wrist_vis=mix(p.wrist_vis, q.wrist_vis),
            )
        )
    return out


class TimelinePose:
    """Keyframed people: ``[(t, [Person, ...]), ...]`` (``[]`` = nobody in frame)."""

    provider = "TimelinePose(scripted)"

    def __init__(self, clock: Clock, keys=None, kpt_threshold: float = 0.3, conf_threshold: float = 0.4):
        self.clock = clock
        self.kpt_threshold = kpt_threshold
        self.conf_threshold = conf_threshold
        self.keys = list(keys or [(0.0, [])])
        self.calls = 0

    def people(self, t: float | None = None) -> list[Person]:
        return list(_interp(self.keys, self.clock() if t is None else t, _blend_people) or [])

    def infer(self, frame_bgr, conf=None):
        self.calls += 1
        h, w = frame_bgr.shape[:2]
        ppl = self.people()
        if not ppl:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 17, 3), np.float32)
        boxes, kpts = zip(*(p.keypoints(w, h) for p in ppl))
        return np.stack(boxes), np.full(len(ppl), 0.92, np.float32), np.stack(kpts)

    def primary_person(self, frame_bgr):
        boxes, scores, kpts = self.infer(frame_bgr)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        return kpts[i], float(scores[i])

    def draw(self, frame, boxes, scores, kpts):
        return draw_people(frame.copy(), boxes, kpts, self.kpt_threshold)


_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def draw_people(img, boxes, kpts, thr=0.3):
    for box, k in zip(boxes, kpts):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
        for a, b in _SKELETON:
            if k[a, 2] >= thr and k[b, 2] >= thr:
                cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), (0, 200, 255), 2)
        if k[10, 2] >= thr:
            cv2.circle(img, (int(k[10, 0]), int(k[10, 1])), 9, (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------------------
# hands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hand:
    """One scripted MediaPipe hand. ``pinch`` is the raw thumb-index distance."""

    pinch: float = 1.2
    roll: float = 0.0  # thumb->index angle, deg
    size: float = 0.16


def _blend_hand(a, b, u):
    if a is None or b is None:
        return None
    mix = lambda x, y: x + (y - x) * u  # noqa: E731
    return Hand(pinch=mix(a.pinch, b.pinch), roll=mix(a.roll, b.roll), size=mix(a.size, b.size))


class TimelineHands:
    """Keyframed hand: ``[(t, Hand | None), ...]``; ``obs_cls`` = the stack's HandObservation."""

    def __init__(self, clock: Clock, obs_cls, keys=None):
        self.clock = clock
        self.obs_cls = obs_cls
        self.keys = list(keys or [(0.0, None)])

    def hand(self, t: float | None = None) -> Hand | None:
        return _interp(self.keys, self.clock() if t is None else t, _blend_hand)

    def process(self, frame_bgr):
        hd = self.hand()
        if hd is None:
            return None, None
        h, w = frame_bgr.shape[:2]
        obs = self.obs_cls(
            pinch=float(hd.pinch),
            thumb_px=(int(0.47 * w), int(0.5 * h)),
            index_px=(int(0.53 * w), int(0.5 * h)),
            handedness="Right",
            hand_size=float(hd.size),
            roll=float(hd.roll),
            roll_ti=float(hd.roll),
            roll_ref=float(hd.roll),
            roll_ti_weight=1.0,
            wrist_px=(int(0.5 * w), int(0.62 * h)),
        )
        return obs, None

    def draw(self, frame, obs, results):
        return frame

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# operator camera + signals pane
# ---------------------------------------------------------------------------
class SignalBoard:
    """What the video's signals pane shows besides the person (thread-safe-ish)."""

    def __init__(self):
        self.lines: list[str] = []
        self.transcript = ""
        self.intent = ""

    def on_event(self, event: str, **data) -> None:
        if event == "heard":
            self.transcript = data.get("text", "")
        elif event == "parsed":
            self.intent = f"{data.get('intent')} {data.get('params') or ''}".strip()
        elif event == "stop":
            self.intent = "STOP (fast path)"
        elif event == "ignored":
            self.intent = f"ignored ({data.get('reason')})"


class ScriptedOperatorCamera:
    """The operator camera: frames of the scripted person at ``fps`` (blocking read)."""

    role = "operator"

    def __init__(self, pose: TimelinePose, hands: TimelineHands | None = None,
                 board: SignalBoard | None = None, width: int = 640, height: int = 480, fps: float = 30.0):
        self.pose, self.hands, self.board = pose, hands, board
        self.width, self.height, self.fps = width, height, fps
        self._next = 0.0
        self._open = True
        self.frames = 0

    def _render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), (58, 52, 48), np.uint8)
        cv2.rectangle(img, (0, int(0.8 * self.height)), (self.width, self.height), (70, 70, 70), -1)
        boxes, _s, kpts = self.pose.infer(img)
        self.pose.calls -= 1  # rendering isn't a model call
        draw_people(img, boxes, kpts, self.pose.kpt_threshold)
        return img

    def read(self):
        if not self._open:
            return False, None
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + 1.0 / self.fps, time.monotonic())
        self.frames += 1
        return True, self._render()

    def ok(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False

    def signals_pane(self) -> np.ndarray:
        """Recorder pane: scripted person + hand + transcript/intent."""
        img = self._render()
        hd = self.hands.hand() if self.hands is not None else None
        if hd is not None:
            txt = f"hand: pinch {hd.pinch:.2f}  roll {hd.roll:+.0f}  size {hd.size:.2f}"
        else:
            txt = "hand: not seen"
        lines = [f"t={self.pose.clock():5.2f}s", txt]
        if self.board is not None:
            lines += [f"heard: {self.board.transcript!r}"[:60], f"intent: {self.board.intent}"[:60]]
        y = 40
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 26
        return img


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
class SimDetector:
    """``ObjectDetector`` stand-in: ground-truth boxes from the twin's segmentation."""

    def __init__(self, seg_detector, detection_cls, resolve_class):
        self.seg = seg_detector
        self.Detection = detection_cls
        self.resolve_class = resolve_class
        self.conf_threshold = 0.4
        self.calls = 0

    def infer(self, frame_bgr, conf=None):
        self.calls += 1
        return [
            self.Detection(name=name, score=score, box=np.array(box, np.float32))
            for name, score, box, _body in self.seg.detect(frame_bgr)
        ]

    def find(self, frame_bgr, spoken_name, conf=None):
        cls = self.resolve_class(spoken_name)
        if cls is None:
            return [], None
        dets = [d for d in self.infer(frame_bgr, conf) if d.name == cls]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, cls

    def draw(self, frame, dets):
        return frame


class CountingDetector:
    """The stack's real ``ObjectDetector`` (YOLOv26s on the rendered frames),
    counting the frames it was asked about."""

    def __init__(self, detector, resolve_class):
        self.inner = detector
        self.resolve_class = resolve_class
        self.conf_threshold = detector.conf_threshold
        self.calls = 0

    def infer(self, frame_bgr, conf=None):
        self.calls += 1
        return self.inner.infer(frame_bgr, conf)

    def find(self, frame_bgr, spoken_name, conf=None):
        cls = self.resolve_class(spoken_name)
        if cls is None:
            return [], None
        dets = [d for d in self.infer(frame_bgr, conf) if d.name == cls]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, cls

    def draw(self, frame, dets):
        return self.inner.draw(frame, dets)


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------
class ScriptedTranscripts:
    """Deliver ``[(t, text), ...]`` to ``deliver(text)`` at scenario time ``t``.

    ``concurrent=True`` runs each delivery on its own thread - for stacks
    whose transcript handler runs the behavior synchronously (the workshop's
    ``handle_transcript``), so a later "stop" can still interrupt it, as the
    always-hot listener does on the rig.
    """

    def __init__(self, clock: Clock, script, deliver, concurrent: bool = False):
        self.clock = clock
        self.script = sorted(script, key=lambda s: s[0])
        self.deliver = deliver
        self.concurrent = concurrent
        self.sent: list[tuple[float, str]] = []
        self.done = threading.Event()
        self.workers: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._run, name="scripted-transcripts", daemon=True)

    def start(self) -> "ScriptedTranscripts":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for t, text in self.script:
                delay = t - self.clock()
                if delay > 0:
                    time.sleep(delay)
                self.sent.append((self.clock(), text))
                if self.concurrent:
                    th = threading.Thread(target=self.deliver, args=(text,), daemon=True)
                    th.start()
                    self.workers.append(th)
                else:
                    self.deliver(text)
        finally:
            self.done.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        for th in self.workers:
            th.join(timeout)

    def sent_at(self, text: str) -> float | None:
        for t, s in self.sent:
            if s == text:
                return t
        return None
