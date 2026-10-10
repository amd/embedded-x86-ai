# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Generic image plumbing (PROVIDED).

Letterboxing, output decoding, and overlay drawing are byte-for-byte the same
in any YOLO project - nothing here is AMD- or ROS-specific, which is exactly
why it is provided instead of being a TODO. Read it once so you know what the
helpers hand you; then spend your time on the NPU/ROS/wiring TODOs.

Model heads. An end-to-end export needs no NMS:

- YOLOv26s-pose:   ``[1, 300, 57]`` = xyxy, score, class, 17*(x, y, v)
- YOLOv26s-detect: ``[1, 300, 6]``  = xyxy, score, class

The default ultralytics ONNX export leaves the raw head instead (``nms``
defaults to ``None``, which turns the end-to-end branch off):

- YOLOv26s-pose:   ``[1, 56, 8400]`` = xywh, score, 17*(x, y, v)
- YOLOv26s-detect: ``[1, 84, 8400]`` = xywh, 80 class scores
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# COCO constants
# ----------------------------------------------------------------------------

# COCO-17 keypoint indices used throughout the pipeline.
KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

# COCO-17 indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# BGR colors
LEFT = (0, 255, 0)  # person's anatomical left
RIGHT = (0, 165, 255)  # person's anatomical right
CENTER = (255, 128, 0)  # spine / pelvis / shoulders
FACE = (255, 0, 200)

# (a, b, color)
SKELETON = [
    # legs
    (L_ANK, L_KNE, LEFT),
    (L_KNE, L_HIP, LEFT),
    (R_ANK, R_KNE, RIGHT),
    (R_KNE, R_HIP, RIGHT),
    # pelvis + torso
    (L_HIP, R_HIP, CENTER),
    (L_SHO, L_HIP, LEFT),
    (R_SHO, R_HIP, RIGHT),
    (L_SHO, R_SHO, CENTER),
    # arms
    (L_SHO, L_ELB, LEFT),
    (L_ELB, L_WRI, LEFT),
    (R_SHO, R_ELB, RIGHT),
    (R_ELB, R_WRI, RIGHT),
    # face
    (NOSE, L_EYE, FACE),
    (NOSE, R_EYE, FACE),
    (L_EYE, L_EAR, FACE),
    (R_EYE, R_EAR, FACE),
    # head to torso -- the link most broken skeletons drop
    (L_EAR, L_SHO, LEFT),
    (R_EAR, R_SHO, RIGHT),
]

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Spoken-word → COCO class synonyms (static asset; extend for your own objects).
SYNONYMS = {
    "ball": "sports ball",
    "cube": None,
    "block": None,
    "phone": "cell phone",
    "mobile": "cell phone",
    "glass": "wine glass",
    "mug": "cup",
    "pen": None,
    "pencil": None,
    "drink": "bottle",
    "water bottle": "bottle",
    "computer": "laptop",
    "plant": "potted plant",
    "teddy": "teddy bear",
    "bear": "teddy bear",
    "controller": "remote",
}


def resolve_class(spoken: str):
    """Map a spoken object name to a COCO class name, or None if unknown."""
    s = (spoken or "").strip().lower()
    if s in COCO_NAMES:
        return s
    if s in SYNONYMS:
        return SYNONYMS[s]
    for word in s.split():
        if word in COCO_NAMES:
            return word
        if word in SYNONYMS and SYNONYMS[word]:
            return SYNONYMS[word]
    for name in COCO_NAMES:
        if name in s:
            return name
    return None


# ----------------------------------------------------------------------------
# Letterbox + decode
# ----------------------------------------------------------------------------


@dataclass
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float


def letterbox(frame_bgr: np.ndarray, imgsz: int):
    """Resize+pad a BGR frame to ``(1, 3, imgsz, imgsz)`` float32 NCHW in [0,1].

    Returns ``(nchw, LetterboxInfo)`` - keep the info, the decoders need it to
    map boxes/keypoints back to original pixel coordinates.
    """
    h, w = frame_bgr.shape[:2]
    ratio = min(imgsz / w, imgsz / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_x, pad_y = (imgsz - new_w) / 2, (imgsz - new_h) / 2

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(nchw), LetterboxInfo(ratio, pad_x, pad_y)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Center-format boxes ``[cx, cy, w, h]`` to corner format ``[x1, y1, x2, y2]``."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms_keep(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.45) -> np.ndarray:
    """Indices kept by greedy NMS. Empty when nothing is above the boxes array."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    picked = cv2.dnn.NMSBoxes(boxes.tolist(), scores.astype(float).tolist(), 0.0, iou)
    if picked is None or len(picked) == 0:
        return np.empty(0, dtype=int)
    return np.asarray(picked, dtype=int).reshape(-1)


def _unmap_boxes(boxes: np.ndarray, info: LetterboxInfo, orig_shape) -> np.ndarray:
    """Letterbox-pixel xyxy boxes back to the original frame, clipped."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y
    boxes /= info.ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def decode_pose(outputs, info: LetterboxInfo, orig_shape, conf: float):
    """Decode a YOLOv26s-pose output into original-pixel space.

    Accepts the end-to-end head ``[1, 300, 57]`` and the raw head
    ``[1, 56, 8400]``. Returns ``boxes [N,4] xyxy``, ``scores [N]``,
    ``kpts [N,17,3] (x, y, vis)``.
    """
    preds = np.asarray(outputs[0])
    if preds.ndim == 3:
        preds = preds[0]
    raw_head = preds.shape[-1] != 57
    if preds.shape[0] in (56, 57) and preds.shape[-1] not in (56, 57):
        preds = preds.T
    if preds.shape[-1] == 57:
        boxes = preds[:, :4].copy()
        scores = preds[:, 4].copy()
        kpts = preds[:, 6:].reshape(-1, 17, 3).copy()
    elif preds.shape[-1] == 56:
        boxes = _xywh_to_xyxy(preds[:, :4])
        scores = preds[:, 4].copy()
        kpts = preds[:, 5:].reshape(-1, 17, 3).copy()
    else:
        raise ValueError(
            f"pose output shape {tuple(np.asarray(outputs[0]).shape)} is neither "
            "[1, 300, 57] nor [1, 56, 8400]"
        )

    mask = scores > conf
    boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]
    if raw_head and len(boxes):
        keep = _nms_keep(boxes, scores)
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    if len(boxes) == 0:
        return boxes, scores, kpts

    boxes = _unmap_boxes(boxes, info, orig_shape)
    kpts[..., 0] -= info.pad_x
    kpts[..., 1] -= info.pad_y
    kpts[..., :2] /= info.ratio
    return boxes, scores, kpts


@dataclass
class Detection:
    name: str
    score: float
    box: np.ndarray  # xyxy, original pixels

    @property
    def center(self):
        return (
            float(self.box[0] + self.box[2]) / 2.0,
            float(self.box[1] + self.box[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        return float(
            max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)
        )


def decode_detections(outputs, info: LetterboxInfo, orig_shape, conf: float):
    """Decode a YOLOv26s-detect output into a list of :class:`Detection`.

    Accepts the end-to-end head ``[1, 300, 6]`` and the raw head
    ``[1, 84, 8400]``.
    """
    preds = np.asarray(outputs[0])
    if preds.ndim == 3:
        preds = preds[0]
    raw_head = preds.shape[-1] != 6
    if preds.shape[0] in (6, 84) and preds.shape[-1] not in (6, 84):
        preds = preds.T
    if preds.shape[-1] == 6:
        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        classes = preds[:, 5].astype(np.int32)
    elif preds.shape[-1] == 84:
        boxes = _xywh_to_xyxy(preds[:, :4].astype(np.float32))
        cls_scores = preds[:, 4:]
        classes = cls_scores.argmax(axis=1).astype(np.int32)
        scores = cls_scores.max(axis=1).astype(np.float32)
    else:
        raise ValueError(
            f"detect output shape {tuple(np.asarray(outputs[0]).shape)} is neither "
            "[1, 300, 6] nor [1, 84, 8400]"
        )

    mask = scores > conf
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
    if raw_head and len(boxes):
        keep = _nms_keep(boxes, scores)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes = _unmap_boxes(boxes, info, orig_shape) if len(boxes) else boxes
    out = []
    for box, score, cls in zip(boxes, scores, classes):
        idx = int(cls)
        name = COCO_NAMES[idx] if 0 <= idx < len(COCO_NAMES) else f"cls{idx}"
        out.append(Detection(name=name, score=float(score), box=box))
    return out


def pick_wrist(kpts: np.ndarray, threshold: float):
    """Pixel ``(x, y)`` of the more-visible wrist, or None if neither is."""
    best, best_v = None, threshold
    for w in (KP["l_wrist"], KP["r_wrist"]):
        if kpts[w, 2] >= best_v:
            best, best_v = (float(kpts[w, 0]), float(kpts[w, 1])), float(kpts[w, 2])
    return best


def pick_arm(kpts: np.ndarray, threshold: float, side: str = "r"):
    """(wrist_xy, elbow_xy) for a fixed side (default right).

    wrist_xy is None when that side's wrist doesn't clear threshold;
    elbow_xy is None when the matching elbow doesn't.
    """
    w, e = KP[f"{side}_wrist"], KP[f"{side}_elbow"]
    if float(kpts[w, 2]) < threshold:
        return None, None
    wrist = (float(kpts[w, 0]), float(kpts[w, 1]))
    elbow = None
    if float(kpts[e, 2]) >= threshold:
        elbow = (float(kpts[e, 0]), float(kpts[e, 1]))
    return wrist, elbow


def pick_nearest(boxes: np.ndarray) -> int:
    """Index of the person CLOSEST to the camera - the largest-area box.

    With multiple people in frame the mimic pipeline follows exactly one: the
    nearest. Box area is the proximity proxy (a closer person subtends more
    pixels), so "nearest" is ``argmax`` over ``(x2-x1) * (y2-y1)``. ``boxes``
    is the ``[N, 4]`` xyxy array from :func:`decode_pose`; callers must ensure
    ``len(boxes) > 0`` (there is no valid index to return for an empty frame).
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


# ----------------------------------------------------------------------------
# Overlays
# ----------------------------------------------------------------------------


def draw_skeleton(
    frame,
    boxes,
    scores,
    kpts,
    kpt_threshold: float = 0.3,
    highlight_right_hand: bool = True,
):
    """Draw person boxes + COCO-17 skeletons; returns a copy."""
    out = frame.copy()

    def valid(pt):
        x, y, v = pt
        return v > kpt_threshold and np.isfinite(x) and np.isfinite(y)

    for box, score, person in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = box.astype(int)

        # scale stroke and joint size with body height so it looks right near and far
        t = max(1, round((y2 - y1) / 150))
        r = max(2, round((y2 - y1) / 90))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), t, cv2.LINE_AA)
        cv2.putText(
            out,
            f"person {score:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

        # bones first
        for a, b, color in SKELETON:
            pa, pb = person[a], person[b]
            if valid(pa) and valid(pb):
                cv2.line(
                    out,
                    (int(pa[0]), int(pa[1])),
                    (int(pb[0]), int(pb[1])),
                    color,
                    t,
                    cv2.LINE_AA,
                )

        # joints on top, side-colored, with a thin dark outline so they stay crisp
        for i, (x, y, v) in enumerate(person):
            if not valid((x, y, v)):
                continue
            if i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR):
                jc = FACE
            elif i % 2 == 1:  # odd COCO indices are the person's left side
                jc = LEFT
            else:
                jc = RIGHT
            cv2.circle(out, (int(x), int(y)), r, jc, -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), r, (30, 30, 30), 1, cv2.LINE_AA)

        # anchor for the mediapipe right-hand crop
        if highlight_right_hand and valid(person[R_WRI]):
            wx, wy = int(person[10][0]), int(person[10][1])
            cv2.circle(
                out, (wx, wy), 8, (0, 255, 255), 2, cv2.LINE_AA
            )  # yellow ring, not filled red
    return out


def draw_detections(frame, dets):
    """Draw labelled detection boxes; returns a copy."""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 160, 255), 2)
        cv2.putText(
            out,
            f"{d.name} {d.score:.2f}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 160, 255),
            2,
        )
    return out


def annotate_lines(frame, lines, origin=(12, 30), color=(0, 255, 255), scale=0.55):
    """Stack text lines onto a frame (in place); returns the frame."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(28 * scale / 0.55)
    return frame


# ----------------------------------------------------------------------------
# Schematic arm panel (for the live UI)
# ----------------------------------------------------------------------------

_L1, _L2, _L3 = 70.0, 62.0, 34.0  # link lengths in panel pixels


def draw_arm_panel(joints, width: int = 300, height: int = 480):
    """Render a schematic SO-101 from a joint dict (degrees).

    Top half: side view of the shoulder_lift → elbow_flex → wrist_flex chain.
    Bottom half: top-down shoulder_pan needle + gripper jaw opening.
    Pure visualization - the real kinematics live on the robot.
    """
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    j = {
        k: float(joints.get(k, 0.0))
        for k in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    }

    # --- side view -----------------------------------------------------
    base = (width // 2, int(height * 0.42))
    cv2.putText(
        panel, "side view", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1
    )
    # Angle conventions: shoulder_lift 0 = upright link; positive leans forward
    # (screen right). elbow_flex/wrist_flex accumulate.
    a1 = np.radians(j["shoulder_lift"] + 90.0)  # -90 (rest) → straight up
    p1 = (int(base[0] + _L1 * np.cos(a1)), int(base[1] - _L1 * np.sin(a1)))
    a2 = a1 - np.radians(j["elbow_flex"] - 90.0)
    p2 = (int(p1[0] + _L2 * np.cos(a2)), int(p1[1] - _L2 * np.sin(a2)))
    a3 = a2 - np.radians(j["wrist_flex"])
    p3 = (int(p2[0] + _L3 * np.cos(a3)), int(p2[1] - _L3 * np.sin(a3)))
    cv2.rectangle(
        panel, (base[0] - 26, base[1]), (base[0] + 26, base[1] + 14), (90, 90, 90), -1
    )
    for a, b, c in (
        (base, p1, (0, 200, 255)),
        (p1, p2, (0, 255, 160)),
        (p2, p3, (255, 200, 0)),
    ):
        cv2.line(panel, a, b, c, 5)
        cv2.circle(panel, b, 5, (255, 255, 255), -1)

    # --- top view: shoulder_pan needle ----------------------------------
    c2 = (width // 2, int(height * 0.68))
    cv2.putText(
        panel,
        "top view (pan)",
        (12, c2[1] - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )
    cv2.circle(panel, c2, 40, (90, 90, 90), 2)
    pan = np.radians(j["shoulder_pan"])
    tip = (int(c2[0] + 40 * np.sin(pan)), int(c2[1] - 40 * np.cos(pan)))
    cv2.line(panel, c2, tip, (0, 200, 255), 4)

    # --- gripper jaws ----------------------------------------------------
    gy = int(height * 0.86)
    cv2.putText(
        panel,
        "gripper",
        (12, gy - 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )
    half = 6 + int(44 * np.clip(j["gripper"] / 100.0, 0.0, 1.0))
    cx = width // 2
    for sign in (-1, 1):
        x = cx + sign * half
        cv2.line(panel, (x, gy - 18), (x, gy + 18), (0, 160, 255), 6)

    # --- numbers ---------------------------------------------------------
    y = height - 74
    for k in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
        cv2.putText(
            panel,
            f"{k:>13s} {j[k]:+7.1f}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )
        y += 18
    for k in ("wrist_flex", "wrist_roll", "gripper"):
        cv2.putText(
            panel,
            f"{k:>13s} {j[k]:+7.1f}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )
        y += 18
    return panel
