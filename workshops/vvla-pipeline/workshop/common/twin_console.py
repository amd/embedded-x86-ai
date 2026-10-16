# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Twin console - one image of what the robot sees, hears and does (PROVIDED).

The same layout as the pipeline's ``run_pipeline.sh --digital-twin`` window
(1200 x 1100)::

    +-----------------+-----------------+-----------------+
    | operator: YOLO pose| operator: MediaPipe| arm: YOLO detect|   400 x 300 each
    +-----------------+-----------------+-----------------+
    |                                   | voice / intent  |
    |        3D view of the twin        | behavior log    |   800 x 800 + 400 x 800
    |                                   | gripper, joints |
    +-----------------------------------+-----------------+

The console only observes. Cameras are wrapped in :class:`FanoutCamera` so a
running behavior and the console both get every frame; the models are tapped
in place (:class:`ModelTap`) so the tiles show what the behavior computed,
and the console runs a model itself only while no behavior is using it.
:meth:`TwinConsole.compose` returns the frame; showing it (a notebook
widget, a HighGUI window) is up to the caller.
"""

from __future__ import annotations

import logging
import textwrap
import threading
import time
from collections import deque

import cv2
import numpy as np

from common.imaging import draw_skeleton
from common.motion import GRIPPER_TORQUE_LIMIT

logger = logging.getLogger(__name__)

TILE_W, TILE_H = 400, 300
VIEW_W, VIEW_H = 800, 800  # tall enough for the raised arm (wave)
LOG_LINES = (VIEW_H - 230) // 20  # what fits between the header and the arm HUD
PANEL_W = 400
IDLE_AFTER_S = 0.5  # a model untouched this long is run by the console itself
SHORT = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "flex",
    "wrist_roll": "roll",
    "gripper": "grip",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY, DIM = (235, 235, 235), (150, 150, 150), (90, 90, 90)
GREEN, YELLOW, RED, CYAN = (80, 200, 80), (60, 200, 230), (60, 60, 230), (220, 200, 80)


class FanoutCamera:
    """One reader thread; every consumer thread gets each new frame once.

    ``read()`` blocks for the next frame this thread hasn't seen, so behaviors
    keep camera-like pacing while the console reads the same stream.
    """

    def __init__(self, inner, timeout_s: float = 1.0):
        self.inner = inner
        self._timeout = timeout_s
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stamp = 0.0
        self._wall = 0.0
        self._seen: dict[int, int] = {}
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True, name="fanout")
        self._thread.start()

    def _pump(self) -> None:
        last = None
        while self._running:
            try:
                ok, frame = self.inner.read()
            except Exception as e:
                logger.debug("[console] camera read failed: %s", e)
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if frame is last:  # latest-frame clients return the same array until a new one lands
                time.sleep(0.005)
                continue
            last = frame
            # A camera that knows when its frame's scene was captured
            # (the twin's) says so in last_stamp; otherwise the arrival time.
            captured = getattr(self.inner, "last_stamp", None)
            with self._cond:
                self._frame, self._stamp = frame, time.monotonic()
                self._wall = time.time() if captured is None else float(captured)
                self._seq += 1
                self._cond.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        tid = threading.get_ident()
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._seq > self._seen.get(tid, 0), timeout=self._timeout
            ):
                return False, None
            self._seen[tid] = self._seq
            return True, self._frame

    def read_after(self, settled_at: float):
        """Next pumped frame captured (or, without a capture stamp, arrived) after ``settled_at``."""
        deadline = time.monotonic() + max(self._timeout, 0.5)
        with self._cond:
            while self._frame is None or self._wall <= settled_at:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=remaining)
            self._seen[threading.get_ident()] = self._seq
            return True, self._frame

    def latest(self) -> np.ndarray | None:
        """Newest frame without blocking, or None if the stream is stale."""
        with self._cond:
            if self._frame is None or time.monotonic() - self._stamp > self._timeout:
                return None
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        close = getattr(self.inner, "close", None) or getattr(self.inner, "release", None)
        if close:
            close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ModelTap:
    """Wrap ``obj.method`` in place: serialise calls and keep the last result.

    Patching the instance (not wrapping the object) also catches internal
    calls such as ``detector.find`` -> ``self.infer``.
    """

    def __init__(self, obj, method: str):
        self.obj = obj
        self.last = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._inner = getattr(obj, method)

        def tapped(*args, **kwargs):
            result = self.call(*args, **kwargs)
            self._stamp = time.monotonic()
            return result

        setattr(obj, method, tapped)

    def call(self, *args, **kwargs):
        """Run the model for the console; doesn't count as a behavior using it."""
        with self._lock:
            result = self._inner(*args, **kwargs)
        self.last = result
        return result

    def idle(self) -> bool:
        """True while no behavior has called the model recently."""
        return time.monotonic() - self._stamp > IDLE_AFTER_S


def _fit(img: np.ndarray | None, w: int, h: int, label: str = "no signal") -> np.ndarray:
    if img is None:
        out = np.zeros((h, w, 3), np.uint8)
        cv2.putText(out, label, (w // 2 - 60, h // 2), FONT, 0.6, DIM, 1, cv2.LINE_AA)
        return out
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _title(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), FONT, 0.5, WHITE, 1, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, mark: float | None = None) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    fill = int(np.clip(frac, 0.0, 1.0) * (w - 2))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)
    if mark is not None:
        mx = x + int(np.clip(mark, 0.0, 1.0) * w)
        cv2.line(img, (mx, y - 3), (mx, y + h + 3), WHITE, 1)


PROCESSOR = {"VitisAIExecutionProvider": "NPU", "CPUExecutionProvider": "CPU"}


def _provider(model, default: str = "CPU") -> str:
    """The processor a model runs on, for a tile title: NPU, CPU, or the provider's name."""
    p = str(getattr(model, "provider", None) or default)
    return PROCESSOR.get(p, p.replace("ExecutionProvider", ""))


def jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    """``img`` as JPEG bytes, what a notebook ``widgets.Image`` shows."""
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def open_webcam(cfg: dict):
    """The webcam, YOLO-pose and MediaPipe Hands: ``(camera, pose, hands)``.

    A scripted person stands in when the webcam gives no frame, for example
    because another kernel holds it.
    """
    from common.fixtures import ScriptedHands, ScriptedPose, SyntheticCamera
    from common.quiet import QuietHands, quiet_stderr
    from models.hands import HandTracker
    from models.vision import PoseEstimator
    from ros.camera_client import make_camera

    pose = PoseEstimator(cfg)
    # MediaPipe's C++ init writes to stderr (absl, EGL, TFLite); QuietHands
    # keeps later detections quiet too.
    with quiet_stderr():
        hands = QuietHands(HandTracker(cfg))
        hands.process(np.zeros((64, 64, 3), np.uint8))
    cam = make_camera(cfg, "operator")
    if cam.read()[0]:
        return cam, pose, hands
    cam.close()
    hands.close()
    device = cfg["cameras"]["operator"].get("device")
    print(f"webcam: no frames from {device} (in use by another kernel?) - a scripted person stands in")
    return SyntheticCamera(), ScriptedPose(), ScriptedHands()


class TwinConsole:
    """Compose the twin console frame from the arm, cameras, models and voice.

    After construction use :attr:`operator` / :attr:`armcam` (the fanned-out
    cameras) wherever a behavior reads frames, and feed the voice side
    through :meth:`on_level`, :meth:`heard`, :meth:`say`, :meth:`parsed`,
    :meth:`stopped` and :meth:`track`.
    """

    def __init__(
        self,
        arm,
        *,
        operator=None,
        pose=None,
        hands=None,
        armcam=None,
        detector=None,
        view=None,
        load_threshold: float = 120.0,
        voice: str = "mic",
        hint: str = "say 'stop' or press the stop button",
    ):
        self.arm = arm
        self.voice = voice  # "mic" | "typed"
        self.hint = hint
        self.load_threshold = float(load_threshold)
        self.operator = FanoutCamera(operator) if operator is not None else None
        self.armcam = FanoutCamera(armcam) if armcam is not None else None
        self._pose, self._hands, self._detector = pose, hands, detector
        self.pose = ModelTap(pose, "infer") if pose is not None else None
        self.hands = ModelTap(hands, "process") if hands is not None else None
        self.detector = ModelTap(detector, "infer") if detector is not None else None
        self.view = view
        self.transcribing = lambda: False  # replaced by the caller (e.g. MicListener)

        self._log: deque[tuple[str, tuple]] = deque(maxlen=LOG_LINES)
        self._lock = threading.Lock()
        self._level, self._speaking, self._level_t = 0.0, False, 0.0
        self._behavior: str | None = None
        self._behavior_t0 = 0.0
        self._stop_flash = 0.0

    # ------------------------------------------------------------ voice side

    def on_level(self, rms: float, speaking: bool) -> None:
        """Microphone block level (``MicListener(on_level=...)``)."""
        with self._lock:
            self._level, self._speaking, self._level_t = rms, speaking, time.monotonic()

    def heard(self, text: str) -> None:
        self.log(f"you: {text}", WHITE)

    def ignored(self, text: str) -> None:
        self.log(f"  (ignored: {text})", GRAY)

    def say(self, message: str) -> None:
        """Robot speech: logged here and printed (use as ``ctx.say``)."""
        self.log(f"robot: {message}", GREEN)
        print(message)

    def parsed(self, cmd) -> None:
        params = ", ".join(f"{k}={v}" for k, v in (cmd.params or {}).items() if v)
        self.log(f"-> {cmd.intent.value}" + (f" ({params})" if params else ""), CYAN)

    def stopped(self) -> None:
        self._stop_flash = time.monotonic()
        self.log("STOP", RED)

    def track(self, name: str, handler):
        """Wrap a behavior handler so the panel shows it running and how it ended."""

        def run(ctx, cmd=None):
            self._behavior, self._behavior_t0 = name, time.monotonic()
            how, result = "failed", None
            try:
                result = handler(ctx, cmd)
                how = "stopped" if ctx.stop_check() else ("failed" if result is False else "done")
            except Exception as e:
                self.log(f"  crashed: {e}", RED)
                raise
            finally:
                took = time.monotonic() - self._behavior_t0
                self.log(f"  {name} {how} after {took:.1f}s", GRAY)
                self._behavior = None
            return result

        return run

    def log(self, text: str, color: tuple = WHITE) -> None:
        lines = textwrap.wrap(text, 46, subsequent_indent="    ") or [""]
        with self._lock:
            self._log.extend((line, color) for line in lines)

    # ------------------------------------------------------------ tiles

    def _pose_tile(self, frame):
        if frame is None or self.pose is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.pose.idle():
            self.pose.call(frame)
        if self.pose.last is None:
            return _fit(frame, TILE_W, TILE_H)
        boxes, scores, kpts = self.pose.last[:3]
        thr = getattr(self._pose, "kpt_threshold", 0.3)
        return _fit(draw_skeleton(frame, boxes, scores, kpts, thr), TILE_W, TILE_H)

    def _hands_tile(self, frame):
        if frame is None or self.hands is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.hands.idle():
            self.hands.call(frame)
        if self.hands.last is None:
            return _fit(frame, TILE_W, TILE_H)
        obs, raw = self.hands.last
        return _fit(self._hands.draw(frame, obs, raw), TILE_W, TILE_H)

    def _detect_tile(self, frame):
        if frame is None or self.detector is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.detector.idle():
            self.detector.call(frame)
        return _fit(self._detector.draw(frame, self.detector.last or []), TILE_W, TILE_H)

    def _view_tile(self):
        frame = None
        if self.view is not None:
            ok, frame = self.view.read()
            frame = frame if ok else None
        img = _fit(frame, VIEW_W, VIEW_H, "no 3D view").copy()
        _title(img, "digital twin")
        return img

    def _panel(self) -> np.ndarray:
        p = np.full((VIEW_H, PANEL_W, 3), 28, np.uint8)
        now = time.monotonic()
        y = 28
        cv2.putText(p, "VOICE", (12, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        with self._lock:
            level, speaking, level_t = self._level, self._speaking, self._level_t
            log = list(self._log)
        if self.voice == "typed":
            state, color = "typed commands", GRAY
        elif level_t == 0.0 or now - level_t > 2.0:
            state, color = "no microphone signal", RED
        elif speaking:
            state, color = "hearing you", GREEN
        elif self.transcribing():
            state, color = "transcribing...", YELLOW
        else:
            state, color = "listening", GRAY
        cv2.putText(p, state, (90, y), FONT, 0.5, color, 1, cv2.LINE_AA)
        if self.voice == "mic":
            db = 20.0 * np.log10(max(level, 1e-6))
            _bar(p, 12, y + 12, PANEL_W - 24, 10, (db + 60.0) / 50.0, GREEN if speaking else DIM)
        y += 52

        if now - self._stop_flash < 1.0:
            cv2.rectangle(p, (0, y - 18), (PANEL_W, y + 8), RED, -1)
            cv2.putText(p, "STOP", (12, y), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        elif self._behavior:
            text = f"running: {self._behavior}  {now - self._behavior_t0:4.1f}s"
            cv2.putText(p, text, (12, y), FONT, 0.55, CYAN, 1, cv2.LINE_AA)
        else:
            cv2.putText(p, "idle - say a command", (12, y), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        y += 22
        cv2.line(p, (12, y), (PANEL_W - 12, y), DIM, 1)
        y += 24

        for text, color in log:
            cv2.putText(p, text, (12, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 20

        self._arm_hud(p, VIEW_H - 92)
        cv2.putText(p, self.hint, (12, VIEW_H - 12), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        return p

    def _arm_hud(self, p, y) -> None:
        try:
            load = self.arm.get_gripper_load()
        except Exception:
            load = None
        try:
            joints = self.arm.get_joints()
        except Exception:
            joints = None
        cv2.line(p, (12, y - 16), (PANEL_W - 12, y - 16), DIM, 1)
        full = 2.0 * self.load_threshold
        # load_threshold is the close's level at GRIPPER_TORQUE_LIMIT; a hold runs lower
        level = self.load_threshold * getattr(self.arm, "gripper_torque_limit", GRIPPER_TORQUE_LIMIT) / GRIPPER_TORQUE_LIMIT
        if load is None:
            cv2.putText(p, "gripper current: n/a", (12, y), FONT, 0.45, GRAY, 1, cv2.LINE_AA)
        else:
            grasp = load >= level
            text = f"gripper current {load:5.0f} / {level:.0f}"
            cv2.putText(p, text, (12, y), FONT, 0.45, GREEN if grasp else WHITE, 1, cv2.LINE_AA)
            _bar(p, 12, y + 8, PANEL_W - 24, 8, load / full, GREEN if grasp else GRAY, mark=level / full)
        if joints:
            parts = [f"{SHORT.get(n, n)} {v:+.0f}" for n, v in joints.items()]
            for i in range(0, len(parts), 3):
                cv2.putText(p, "   ".join(parts[i : i + 3]), (12, y + 36 + 18 * (i // 3)),
                            FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    def _webcam_tiles(self, frame) -> list[np.ndarray]:
        pose, hands = self._pose_tile(frame).copy(), self._hands_tile(frame).copy()
        _title(pose, f"operator: YOLO pose [{_provider(self._pose, 'off')}]")
        _title(hands, f"operator: MediaPipe hands [{'CPU' if self._hands else 'off'}]")
        return [pose, hands]

    def webcam_tiles(self) -> np.ndarray:
        """The console's two webcam tiles, YOLO pose and MediaPipe hands, side by side."""
        frame = self.operator.latest() if self.operator is not None else None
        return np.hstack(self._webcam_tiles(frame))

    def compose(self) -> np.ndarray:
        """One 1200 x 1100 BGR frame of the whole console."""
        operator_frame = self.operator.latest() if self.operator is not None else None
        armcam = self.armcam.latest() if self.armcam is not None else None
        detect = self._detect_tile(armcam).copy()
        _title(detect, f"arm: YOLO detect [{_provider(self._detector, 'off')}]")
        tiles = [*self._webcam_tiles(operator_frame), detect]
        return np.vstack([np.hstack(tiles), np.hstack([self._view_tile(), self._panel()])])

    def close(self) -> None:
        """Stop the camera fan-out threads (and close the cameras they wrap)."""
        for cam in (self.operator, self.armcam):
            if cam is not None:
                cam.close()
