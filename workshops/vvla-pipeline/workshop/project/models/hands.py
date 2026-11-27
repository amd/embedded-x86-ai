# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MediaPipe Hands on the CPU - the pinch/roll/depth sense.

Why the CPU? Budget arithmetic: the NPU is saturated by the two YOLO models
(camera-rate pose is latency-critical), the iGPU belongs to Gemma, and
MediaPipe Hands is cheap enough that the CPU runs it at frame rate. Placing
a model is a *decision*, and this is the CPU leg of it.

Your TODO here is the initialization (:func:`create_hand_tracker`). The
landmark → signal math in :meth:`HandTracker.process` is generic geometry
and is provided - read it once to know what the mapper receives.

Notebook: ``notebooks/01_ai_models/02_cpu_mediapipe.ipynb``
Self-test: ``python -m models.selftest``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
INDEX_MCP, PINKY_MCP = 5, 17


@dataclass
class HandObservation:
    """Per-frame hand signals extracted from MediaPipe landmarks (PROVIDED)."""

    pinch: float                 # normalized thumb↔index distance (gripper)
    thumb_px: tuple              # pixel coords
    index_px: tuple
    handedness: str              # physical side: "Left" / "Right"
    hand_size: float = 0.0       # wrist→middle-MCP span / frame height (depth cue)
    roll: float = 0.0            # in-plane hand angle, deg - fallback roll
    roll_ti: float = 0.0         # thumb→index vector angle, deg - primary roll
    roll_ref: float = 0.0        # knuckle-line angle, deg - stable reference
    roll_ti_weight: float = 0.0  # reliability of roll_ti in [0,1]
    wrist_px: tuple = (0, 0)


def create_hand_tracker(cfg: dict):
    """Build a configured ``mediapipe.solutions.hands.Hands`` (CPU).

    Read ``mediapipe:`` from the config: ``max_num_hands``,
    ``min_detection_confidence``, ``min_tracking_confidence``. Video mode
    (``static_image_mode=False``) so the tracker exploits temporal
    continuity between frames.
    """
    import mediapipe as mp  # heavy import, keep local

    m = (cfg or {}).get("mediapipe", {})
    # >>> TODO 1.4: create_hand_tracker - notebooks/01_ai_models/02_cpu_mediapipe.ipynb
    raise NotImplementedError(
        "TODO 1.4 (create_hand_tracker): implement me - see notebooks/01_ai_models/02_cpu_mediapipe.ipynb")
    # <<< TODO 1.4


class HandTracker:
    """Wraps MediaPipe Hands and extracts the pipeline's control signals.

    ``hands`` may be injected (a prebuilt ``mp.solutions.hands.Hands``), which
    the notebooks use so they stay runnable before TODO 1.4 is implemented.
    """

    def __init__(self, cfg: dict | None = None, hands=None):
        m = (cfg or {}).get("mediapipe", {})
        self._roll_ti_full_px = float(m.get("roll_ti_full_px", 40.0))
        # Which hand drives the arm: "right" (default), "left", or "any".
        self._target_hand = str(m.get("track_hand", "right")).strip().lower()
        # MediaPipe labels handedness as if the image were mirrored (selfie
        # view). Our rig frames are NOT mirrored, so MediaPipe's "Right" is
        # really your physical LEFT hand - we flip to recover the physical
        # side. Set mediapipe.frames_mirrored: true if you pre-flip frames.
        self._frames_mirrored = bool(m.get("frames_mirrored", False))
        self.hands = hands if hands is not None else create_hand_tracker(cfg or {})

    def _physical_handedness(self, mp_label: str) -> str:
        """Correct MediaPipe's mirrored-view Left/Right to the physical side."""
        if self._frames_mirrored:
            return mp_label
        return {"Left": "Right", "Right": "Left"}.get(mp_label, mp_label)

    def _select_hand(self, results):
        """Index of the hand to track per ``track_hand``, or None if absent.

        ``track_hand: any`` keeps the first hand MediaPipe returns (the old
        behavior). Otherwise keep ONLY the requested physical hand and return
        None when it is not in frame, so the arm holds its pose instead of
        following the wrong hand.
        """
        landmarks = results.multi_hand_landmarks
        if not landmarks:
            return None
        if self._target_hand in ("any", "both", ""):
            return 0
        for i, handed in enumerate(results.multi_handedness or []):
            label = handed.classification[0].label
            if self._physical_handedness(label).lower() == self._target_hand:
                return i
        return None

    def process(self, frame_bgr):
        """PROVIDED: landmarks → :class:`HandObservation` (or ``(None, raw)``).

        Only the hand named by ``mediapipe.track_hand`` (default: your right)
        becomes signals; any other hand in frame is ignored.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        idx = self._select_hand(results)
        if idx is None:
            return None, results

        lm = results.multi_hand_landmarks[idx].landmark
        thumb = np.array([lm[THUMB_TIP].x, lm[THUMB_TIP].y])
        index = np.array([lm[INDEX_TIP].x, lm[INDEX_TIP].y])
        wrist = np.array([lm[WRIST].x, lm[WRIST].y])
        mcp = np.array([lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y])

        hand_size_norm = float(np.linalg.norm(mcp - wrist)) or 1e-6
        pinch = float(np.linalg.norm(thumb - index)) / hand_size_norm

        wrist_px = np.array([wrist[0] * w, wrist[1] * h])
        mcp_px = np.array([mcp[0] * w, mcp[1] * h])
        v = mcp_px - wrist_px
        hand_size = float(np.linalg.norm(v)) / float(h or 1)
        roll = float(np.degrees(np.arctan2(v[0], -v[1])))

        thumb_px = np.array([thumb[0] * w, thumb[1] * h])
        index_px = np.array([index[0] * w, index[1] * h])
        ti = index_px - thumb_px
        roll_ti = float(np.degrees(np.arctan2(-ti[1], ti[0])))
        roll_ti_weight = float(np.clip(
            float(np.linalg.norm(ti)) / max(self._roll_ti_full_px, 1e-6), 0.0, 1.0))

        imcp = np.array([lm[INDEX_MCP].x * w, lm[INDEX_MCP].y * h])
        pmcp = np.array([lm[PINKY_MCP].x * w, lm[PINKY_MCP].y * h])
        ref = pmcp - imcp
        roll_ref = float(np.degrees(np.arctan2(-ref[1], ref[0])))

        handedness = "Unknown"
        if results.multi_handedness and idx < len(results.multi_handedness):
            mp_label = results.multi_handedness[idx].classification[0].label
            handedness = self._physical_handedness(mp_label)

        obs = HandObservation(
            pinch=pinch,
            thumb_px=(int(thumb[0] * w), int(thumb[1] * h)),
            index_px=(int(index[0] * w), int(index[1] * h)),
            handedness=handedness, hand_size=hand_size, roll=roll,
            roll_ti=roll_ti, roll_ref=roll_ref, roll_ti_weight=roll_ti_weight,
            wrist_px=(int(wrist_px[0]), int(wrist_px[1])),
        )
        return obs, results

    def draw(self, frame, obs, results):
        """PROVIDED: landmark + pinch overlay."""
        import mediapipe as mp

        out = frame.copy()
        if results is not None and getattr(results, "multi_hand_landmarks", None):
            for hand_lms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    out, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
        if obs:
            cv2.line(out, obs.thumb_px, obs.index_px, (0, 0, 255), 3)
            mid = ((obs.thumb_px[0] + obs.index_px[0]) // 2,
                   (obs.thumb_px[1] + obs.index_px[1]) // 2)
            cv2.putText(out, f"pinch {obs.pinch:.2f} ({obs.handedness})", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    def close(self) -> None:
        self.hands.close()
