"""Twin console - one desktop window showing what the pipeline sees, hears and does.

Layout (1200 x 1100)::

    +-----------------+-----------------+-----------------+
    | operator: YOLO pose| operator: MediaPipe| arm: YOLO detect|   400 x 300 each
    +-----------------+-----------------+-----------------+
    |                                   | voice / intent  |
    |        3D view of the twin        | behavior log    |   800 x 800 + 400 x 800
    |                                   | gripper, joints |
    +-----------------------------------+-----------------+

The console only observes. Cameras are wrapped in :class:`FanoutCamera` so the
console and a running behavior both get every frame; the models are tapped in
place (:class:`ModelTap`) so the tiles show what the behavior computed, and
the console runs a model itself only while no behavior is using it.

All HighGUI calls happen on the thread that calls :meth:`TwinConsole.run`.
"""

from __future__ import annotations

import logging
import textwrap
import threading
import time
from collections import deque

import cv2
import numpy as np

from vla_pipeline.robot.arm_interface import GRIPPER_TORQUE_LIMIT

logger = logging.getLogger(__name__)

WINDOW = "SO-101 twin console"
TILE_W, TILE_H = 400, 300
VIEW_W, VIEW_H = 800, 800  # tall enough for the raised arm (wave)
LOG_LINES = (VIEW_H - 230) // 20  # what fits between the header and the arm HUD
PANEL_W = 400
FPS = 15.0
IDLE_AFTER_S = 0.5  # a model untouched this long is run by the console itself
_PROCESSOR = {"VitisAIExecutionProvider": "NPU", "CPUExecutionProvider": "CPU"}
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
    keep V4L2-like pacing while the console reads the same stream.
    """

    def __init__(self, inner, timeout_s: float = 1.0):
        self.inner = inner
        self.role = getattr(inner, "role", "?")
        self._timeout = timeout_s
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stamp = 0.0
        self._wall = 0.0
        self._seen: dict[int, int] = {}
        self._running = True
        self._thread = threading.Thread(
            target=self._pump, daemon=True, name=f"fanout-{self.role}"
        )
        self._thread.start()

    def _pump(self) -> None:
        last = None
        while self._running:
            try:
                ok, frame = self.inner.read()
            except Exception as e:
                logger.debug("[console] %s read failed: %s", self.role, e)
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if frame is last:  # latest-frame clients return the same array until a new one lands
                time.sleep(0.005)
                continue
            last = frame
            with self._cond:
                self._frame, self._stamp, self._wall = frame, time.monotonic(), time.time()
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

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """Next pumped frame whose arrival is later than ``settled_at``."""
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

    def ok(self) -> bool:
        return self.latest() is not None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        close = getattr(self.inner, "close", None)
        if close:
            close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ModelTap:
    """Wrap ``obj.method`` in place: serialise calls and keep the last result.

    Patching the instance (not wrapping the object) also catches internal
    calls such as ``ObjectDetector.find`` -> ``self.infer``.
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

    def provider(self) -> str:
        """The processor the model runs on, for a tile title: NPU, CPU, or the provider's name."""
        session = getattr(self.obj, "session", None)
        if session is None:
            return "CPU"
        p = session.get_providers()[0]
        return _PROCESSOR.get(p, p.replace("ExecutionProvider", ""))


def _view_source(cfg: dict):
    """The twin's free 3D camera: the ROS topic from so101_sim.node, or an in-process render."""
    if (cfg.get("robot") or {}).get("backend") != "sim":
        return None
    try:
        if (cfg.get("robot") or {}).get("use_ros2"):
            from so101_sim.node import VIEW_TOPIC

            from vla_pipeline.vision.camera import Ros2Camera

            return Ros2Camera("view", VIEW_TOPIC, stale_after_s=1.0)
        from so101_sim import SimCamera

        from vla_pipeline.robot.sim_arm import sim_world

        return SimCamera(sim_world(cfg), "view")
    except Exception as e:
        logger.warning("[console] no 3D view: %s", e)
        return None


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


class TwinConsole:
    """Observe a :class:`~vla_pipeline.main.PipelineContext` and draw it in one window."""

    def __init__(self, ctx, voice: str = "mic"):
        self.ctx = ctx
        self.voice = voice  # "mic" | "typed"
        cfg = ctx.cfg
        self.load_threshold = float(
            ((cfg.get("behaviors") or {}).get("grip") or {}).get("load_threshold", 120.0)
        )

        if ctx.operator_camera is not None:
            ctx.operator_camera = FanoutCamera(ctx.operator_camera)
        if ctx.arm_camera is not None:
            ctx.arm_camera = FanoutCamera(ctx.arm_camera)
        self.pose = ModelTap(ctx.pose, "infer") if ctx.pose is not None else None
        self.hands = ModelTap(ctx.hands, "process") if ctx.hands is not None else None
        self.detector = (
            ModelTap(ctx.detector, "infer") if ctx.detector is not None else None
        )
        self.view = _view_source(cfg)

        self._log: deque[tuple[str, tuple]] = deque(maxlen=LOG_LINES)
        self._lock = threading.Lock()
        self._level, self._speaking, self._level_t = 0.0, False, 0.0
        self._asr_pending = False
        self._behavior: str | None = None
        self._behavior_t0 = 0.0
        self._stop_flash = 0.0

        say = ctx.say

        def _say(message: str) -> None:
            self._add(f"robot: {message}", GREEN)
            say(message)

        ctx.say = _say
        # Behaviors draw into this window, never their own.
        ctx.headless = True

    # ------------------------------------------------------------ inputs

    def attach(self, dispatcher) -> None:
        self._dispatcher = dispatcher
        dispatcher.listeners.append(self.on_event)

    def _restore_twin(self) -> bool:
        """Park the twin and restore its props. True when this process can."""
        arm = self.ctx.arm
        world = getattr(arm, "world", None)
        if world is not None:
            from vla_pipeline.robot.arm_interface import restore_twin

            restore_twin(arm, world)
            return True
        restore = getattr(arm, "reset_sim", None)
        return bool(restore and restore())

    def reset_scene(self) -> None:
        """Stop the behavior, park the arm with the jaws open, and restore the props."""
        ctx = self.ctx
        dispatcher = getattr(self, "_dispatcher", None)
        holding = getattr(dispatcher, "holding", None)
        if holding is not None:
            holding.set()
        ctx.stop_event.set()
        if dispatcher is not None:
            dispatcher.discard()
        deadline = time.time() + 3.0
        busy = getattr(dispatcher, "busy", None)
        while busy is not None and busy.is_set() and time.time() < deadline:
            time.sleep(0.05)
        if dispatcher is not None:
            dispatcher.discard()
        try:
            state = getattr(ctx, "shared_state", None)
            if isinstance(state, dict):
                state.pop("grip_lock", None)
                state.pop("grip_lock_pos", None)
            if not self._restore_twin():
                ctx.say("Reset is only available on the digital twin.")
                return
            ctx.say("Reset. Arm parked, objects back on the table.")
        finally:
            if dispatcher is not None:
                dispatcher.discard()
            ctx.stop_event.clear()
            if holding is not None:
                holding.clear()

    def on_level(self, rms: float, speaking: bool) -> None:
        """Microphone block level from :meth:`FastListener.run_forever`."""
        with self._lock:
            self._level, self._speaking, self._level_t = rms, speaking, time.monotonic()

    def note_asr(self, busy: bool) -> None:
        """True while Whisper is decoding. The meter keeps running either way."""
        with self._lock:
            self._asr_pending = busy

    def on_event(self, event: str, **data) -> None:
        if event == "heard":
            self._asr_pending = False
            self._add(f"you: {data['text']}", WHITE)
        elif event == "ignored":
            self._add(f"  (ignored: {data['reason']})", GRAY)
        elif event == "stop":
            self._stop_flash = time.monotonic()
            self._add("STOP", RED)
        elif event == "parsed":
            params = ", ".join(f"{k}={v}" for k, v in data["params"].items())
            self._add(f"-> {data['intent']}" + (f" ({params})" if params else ""), CYAN)
        elif event == "behavior_start":
            self._behavior, self._behavior_t0 = data["intent"], time.monotonic()
        elif event == "behavior_crash":
            self._add(f"  crashed: {data['error']}", RED)
        elif event == "behavior_end":
            took = time.monotonic() - self._behavior_t0
            how = "stopped" if data.get("stopped") else ("done" if data.get("ok") else "failed")
            self._add(f"  {data['intent']} {how} after {took:.1f}s", GRAY)
            self._behavior = None

    def _add(self, text: str, color: tuple) -> None:
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
        boxes, scores, kpts = self.pose.last
        return _fit(self.ctx.pose.draw(frame, boxes, scores, kpts), TILE_W, TILE_H)

    def _hands_tile(self, frame):
        if frame is None or self.hands is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.hands.idle():
            self.hands.call(frame)
        if self.hands.last is None:
            return _fit(frame, TILE_W, TILE_H)
        obs, results = self.hands.last
        return _fit(self.ctx.hands.draw(frame, obs, results), TILE_W, TILE_H)

    def _detect_tile(self, frame):
        if frame is None or self.detector is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.detector.idle():
            self.detector.call(frame)
        dets = self.detector.last or []
        return _fit(self.ctx.detector.draw(frame, dets), TILE_W, TILE_H)

    def _view_tile(self):
        frame = None
        if self.view is not None:
            ok, frame = self.view.read()
            frame = frame if ok else None
        label = "waiting for /sim/view" if self.view is not None else "no 3D view"
        img = _fit(frame, VIEW_W, VIEW_H, label).copy()
        _title(img, "digital twin")
        return img

    def _panel(self) -> np.ndarray:
        p = np.full((VIEW_H, PANEL_W, 3), 28, np.uint8)
        now = time.monotonic()
        y = 28
        cv2.putText(p, "VOICE", (12, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        with self._lock:
            level, speaking, level_t = self._level, self._speaking, self._level_t
            asr, log = self._asr_pending, list(self._log)
        if self.voice == "typed":
            state, color = "typed commands (stdin)", GRAY
        elif level_t == 0.0 or now - level_t > 2.0:
            state, color = "no microphone signal", RED
        elif speaking:
            state, color = "hearing you", GREEN
        elif asr:
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
        cv2.putText(p, "space: stop    r: reset    q: quit", (12, VIEW_H - 12), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        return p

    def _arm_hud(self, p, y) -> None:
        arm = self.ctx.arm
        try:
            load = arm.get_gripper_load()
        except Exception:
            load = None
        try:
            joints = arm.get_joints()
        except Exception:
            joints = None
        cv2.line(p, (12, y - 16), (PANEL_W - 12, y - 16), DIM, 1)
        full = 2.0 * self.load_threshold
        # load_threshold is the close's level at GRIPPER_TORQUE_LIMIT; a hold runs lower
        level = self.load_threshold * getattr(arm, "gripper_torque_limit", GRIPPER_TORQUE_LIMIT) / GRIPPER_TORQUE_LIMIT
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
                cv2.putText(p, "   ".join(parts[i : i + 3]), (12, y + 36 + 18 * (i // 3)), FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    def compose(self) -> np.ndarray:
        operator_frame = self.ctx.operator_camera.latest() if self.ctx.operator_camera is not None else None
        armcam = self.ctx.arm_camera.latest() if self.ctx.arm_camera is not None else None
        tiles = [
            self._pose_tile(operator_frame).copy(),
            self._hands_tile(operator_frame).copy(),
            self._detect_tile(armcam).copy(),
        ]
        _title(tiles[0], f"operator: YOLO pose [{self.pose.provider() if self.pose else 'off'}]")
        _title(tiles[1], f"operator: MediaPipe hands [{self.hands.provider() if self.hands else 'off'}]")
        _title(tiles[2], f"arm: YOLO detect [{self.detector.provider() if self.detector else 'off'}]")
        top = np.hstack(tiles)
        bottom = np.hstack([self._view_tile(), self._panel()])
        return np.vstack([top, bottom])

    # ------------------------------------------------------------ loop

    def run(self, shutdown: threading.Event, on_stop) -> None:
        """Draw until ``shutdown`` is set or the window is closed.

        Space calls ``on_stop()``. ``r`` resets the twin: park the arm and put
        the props back.
        """
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        scale = min(1.0, 950 / (TILE_H + VIEW_H))  # fit a 1080p screen; drag to resize
        cv2.resizeWindow(WINDOW, int(TILE_W * 3 * scale), int((TILE_H + VIEW_H) * scale))
        period = 1.0 / FPS
        try:
            while not shutdown.is_set():
                t0 = time.monotonic()
                try:
                    cv2.imshow(WINDOW, self.compose())
                except Exception:
                    logger.exception("[console] frame failed")
                key = cv2.waitKey(max(1, int((period - (time.monotonic() - t0)) * 1e3))) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    on_stop()
                elif key in (ord("r"), ord("R")):
                    self.reset_scene()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            shutdown.set()
            cv2.destroyWindow(WINDOW)
            if self.view is not None:
                (getattr(self.view, "close", None) or self.view.release)()
