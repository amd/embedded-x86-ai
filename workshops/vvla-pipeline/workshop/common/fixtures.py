# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Synthetic fixtures for off-hardware runs (PROVIDED).

Everything the notebooks, self-tests, and the live UI need to *do something
useful* without an NPU, a robot, cameras, or a running llama-server:

- a tiny embedded ONNX model (``y = 2x``) to exercise your session factory,
- a synthetic camera producing a moving test pattern,
- a scripted "person" pose source and hand source so the mimic chain runs
  end-to-end with no webcam,
- a mock llama-server that speaks just enough OpenAI-compatible HTTP to test
  your intent parsing (and records what your client actually sent),
- an offline keyword parser as a stand-in intent brain.
"""

from __future__ import annotations

import base64
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Tiny ONNX model (y = x * 2, input [1,4] float32) - 107 bytes.
# ----------------------------------------------------------------------------

TINY_ONNX_B64 = (
    "CAgSCHdvcmtzaG9wOlcKEAoBeAoDdHdvEgF5IgNNdWwSCHRpbnlfbXVsKg8IARABIgQAAABAQgN0"
    "d29aEwoBeBIOCgwIARIICgIIAQoCCARiEwoBeRIOCgwIARIICgIIAQoCCARCBAoAEA0="
)


def tiny_onnx_path(directory) -> Path:
    """Write the embedded tiny model into ``directory`` and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tiny_mul.onnx"
    if not path.exists():
        path.write_bytes(base64.b64decode(TINY_ONNX_B64))
    return path


# ----------------------------------------------------------------------------
# Synthetic camera
# ----------------------------------------------------------------------------


class SyntheticCamera:
    """Frame source with the same ``read() -> (ok, frame)`` API as a camera.

    Produces a deterministic moving test pattern (gradient + orbiting disc +
    frame counter) so topic round-trips can be verified pixel-for-pixel.
    """

    role = "synthetic"

    def __init__(self, width: int = 640, height: int = 480):
        import cv2

        self._cv2 = cv2
        self.width, self.height = width, height
        self.frames = 0

    def read(self):
        cv2 = self._cv2
        w, h = self.width, self.height
        t = self.frames
        self.frames += 1
        xx = np.linspace(0, 255, w, dtype=np.uint8)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = xx[None, :]
        frame[:, :, 1] = np.linspace(0, 180, h, dtype=np.uint8)[:, None]
        cx = int(w / 2 + w / 3 * math.sin(t / 20.0))
        cy = int(h / 2 + h / 4 * math.cos(t / 20.0))
        cv2.circle(frame, (cx, cy), 40, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"synthetic frame {t}",
            (12, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return True, frame

    def close(self):
        pass


# ----------------------------------------------------------------------------
# Scripted pose / hands (stand-ins for the NPU + MediaPipe models)
# ----------------------------------------------------------------------------


class ScriptedPose:
    """Drop-in for ``models.vision.PoseEstimator`` with a scripted person.

    ``infer`` returns one full-body detection whose right wrist orbits the
    frame, so the mimic composition (wrist -> mapper -> arm) runs without
    YOLO or a webcam.
    """

    provider = "Scripted(synthetic)"
    kpt_threshold = 0.3
    conf_threshold = 0.4

    def __init__(self, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self._t = 0

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._t += 1
        t = self._t / 25.0
        cx, cy = w * 0.5, h * 0.45

        # start fully invisible; only joints we place become visible
        kpts = np.zeros((1, 17, 3), dtype=np.float32)

        def put(i, x, y, v=0.9):
            kpts[0, i] = (x, y, v)

        # head cluster
        put(0, cx, cy - h * 0.18)  # nose
        put(1, cx - w * 0.02, cy - h * 0.20)  # left eye
        put(2, cx + w * 0.02, cy - h * 0.20)  # right eye
        put(3, cx - w * 0.045, cy - h * 0.185)  # left ear
        put(4, cx + w * 0.045, cy - h * 0.185)  # right ear

        # shoulders + hips
        put(5, cx - w * 0.09, cy - h * 0.05)  # left shoulder
        put(6, cx + w * 0.09, cy - h * 0.05)  # right shoulder
        put(11, cx - w * 0.07, cy + h * 0.22)  # left hip
        put(12, cx + w * 0.07, cy + h * 0.22)  # right hip

        # left arm (relaxed)
        put(7, cx - w * 0.14, cy + h * 0.08)  # left elbow
        put(9, cx - w * 0.18, cy + h * 0.18, 0.4)  # left wrist (low vis)

        # legs (these were missing entirely)
        put(13, cx - w * 0.07, cy + h * 0.45)  # left knee
        put(14, cx + w * 0.07, cy + h * 0.45)  # right knee
        put(15, cx - w * 0.08, cy + h * 0.66)  # left ankle
        put(16, cx + w * 0.08, cy + h * 0.66)  # right ankle

        # right arm: elbow, then the orbiting wrist (the control signal)
        put(8, cx + w * 0.14, cy + h * 0.08)  # right elbow
        kpts[0, 10] = (
            cx + w * 0.28 * math.sin(t),
            cy + h * 0.22 * math.cos(0.7 * t),
            0.95,
        )  # right wrist

        boxes = np.array(
            [[cx - w * 0.2, cy - h * 0.25, cx + w * 0.2, cy + h * 0.72]],
            dtype=np.float32,
        )
        scores = np.array([0.93], dtype=np.float32)
        return boxes, scores, kpts


class ScriptedHands:
    """Drop-in for ``models.hands.HandTracker`` with a scripted pinch/roll."""

    def __init__(self):
        self._t = 0

    def process(self, frame_bgr):
        from models.hands import HandObservation

        self._t += 1
        t = self._t / 30.0
        pinch = 0.75 + 0.65 * math.sin(t)  # oscillates open/closed
        obs = HandObservation(
            pinch=max(pinch, 0.0),
            thumb_px=(300, 240),
            index_px=(340, 240),
            handedness="Right",
            hand_size=0.16 + 0.03 * math.sin(0.5 * t),
            roll=20.0 * math.sin(0.8 * t),
            roll_ti=25.0 * math.sin(0.8 * t),
            roll_ref=25.0 * math.sin(0.8 * t),
            roll_ti_weight=1.0,
            wrist_px=(320, 300),
        )
        return obs, None

    def draw(self, frame, obs, results):
        return frame

    def close(self):
        pass


# ----------------------------------------------------------------------------
# Mock llama-server
# ----------------------------------------------------------------------------

_KEYWORD_ROUTES = [
    (r"\b(stop|halt|freeze|abort)\b", "stop", None),
    (r"\b(release|open|drop)\b|\blet go\b", "release", None),
    (r"\b(copy|mirror|follow|mimic)\b", "gesture_mimic", None),
    (r"\b(block|cube)\b", "fetch_block", None),
    (r"\b(pick|grab|bring|fetch|get)\b", "pick_place", "OBJECT"),
    (r"\b(dance|groove|party|moves)\b", "dance", None),
    (r"\b(hold|take)\b", "grip", None),
    (r"\b(wave|hi|hello|greet)\b", "wave", None),
    (r"\b(home|rest|reset)\b", "home", None),
]

_OBJECT_WORDS = (
    "ball",
    "bottle",
    "cup",
    "mug",
    "phone",
    "apple",
    "banana",
    "book",
    "toothbrush",
    "remote",
    "orange",
    "teddy",
)


def offline_parse_text(text: str) -> dict:
    """Keyword routing → the same ``{"command", "object"}`` dict the LLM emits.

    Used by the mock server below and as the app's ``--no-llm`` brain. It is a
    *stand-in*, not the lesson - the real thing is grammar-constrained Llama.
    """
    low = (text or "").lower()
    for pattern, command, obj_mode in _KEYWORD_ROUTES:
        if re.search(pattern, low):
            obj = ""
            if obj_mode == "OBJECT":
                for word in _OBJECT_WORDS:
                    if word in low:
                        obj = word
                        break
            return {"command": command, "object": obj}
    return {"command": "unknown", "object": ""}


class MockLlamaServer:
    """A llama-server imposter: ``GET /health`` + ``POST /v1/chat/completions``.

    Routes intents with keywords and *records every request payload* so the
    self-test can verify your client sent the GBNF grammar, the system
    prompt, and temperature 0 - the things that make the intent parser
    deterministic on the real iGPU server.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({"status": "ok"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                mock.requests.append(payload)
                user = ""
                for m in payload.get("messages", []):
                    if m.get("role") == "user":
                        user = m.get("content", "")
                content = json.dumps(offline_parse_text(user))
                body = json.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": content}}
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.host = host
        self.port = self._httpd.server_port
        self.base_url = f"http://{self.host}:{self.port}"
        self.requests: list = []
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                import urllib.request

                urllib.request.urlopen(self.base_url + "/health", timeout=0.3)
                return self
            except Exception:
                time.sleep(0.02)
        raise RuntimeError("mock llama-server failed to start")

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def last_payload(self):
        return self.requests[-1] if self.requests else None
