# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLOv26s object detection on the Ryzen AI NPU - voice-command grounding.

Same structure as :class:`vla_pipeline.vision.yolo_pose_npu.PoseEstimator`
(letterbox → VitisAI/CPU session → de-letterboxed outputs) but for the
80-class COCO detection head ([1, 300, 6]: xyxy, score, class). Used by:

- ``pick_place`` - locate the object named in the voice command ("pick up
  the **ball**") in the arm/operator camera frame;
- ``grip`` - optional verification that an object is present between the jaws.

Spoken names are matched through a synonym table (``"ball" → sports ball``)
so natural phrasing works without retraining.

Independent test
----------------
    python -m vla_pipeline.vision.yolo_detect_npu                  # webcam overlay
    python -m vla_pipeline.vision.yolo_detect_npu --find ball
    python -m vla_pipeline.vision.yolo_detect_npu --device cpu
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from vla_pipeline.utils.config import load_config
from vla_pipeline.utils.npu_session import build_session, size_npu
from vla_pipeline.vision.yolo_pose_npu import head_rows, letterbox, nms_keep, xywh_to_xyxy

logger = logging.getLogger(__name__)

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

# Spoken-word → COCO class synonyms (lowercased, applied after exact match).
SYNONYMS = {
    "ball": "sports ball",
    "sports_ball": "sports ball",
    "soccer": "sports ball",
    "soccer ball": "sports ball",
    "football": "sports ball",
    "foot ball": "sports ball",
    "cube": None,
    "block": None,
    "phone": "cell phone",
    "mobile": "cell phone",
    "glass": "wine glass",
    "mug": "cup",
    "pen": None,
    "pencil": None,
    "brush": "toothbrush",
    "drink": "bottle",
    "water bottle": "bottle",
    "computer": "laptop",
    "plant": "potted plant",
    "teddy": "teddy bear",
    "bear": "teddy bear",
    "controller": "remote",
}
# Classes with value None aren't in COCO-80; pick_place reports "can't
# recognize that object" rather than guessing. The twin uses four real COCO
# classes: sports ball, book, bottle, and toothbrush.


@dataclass
class Detection:
    """One detected object: class name, confidence, and pixel-space box."""

    name: str
    score: float
    box: np.ndarray  # xyxy, original pixels

    @property
    def center(self) -> tuple[float, float]:
        """Center of the box in pixels."""
        return (
            float(self.box[0] + self.box[2]) / 2.0,
            float(self.box[1] + self.box[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        """Area of the box in square pixels."""
        return float(
            max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)
        )


def resolve_class(spoken: str) -> str | None:
    """Map a spoken object name to a COCO class, or None if unknown."""
    s = (spoken or "").strip().lower()
    if s in COCO_NAMES:
        return s
    if s in SYNONYMS:
        return SYNONYMS[s]
    # Loose containment ("the red ball" → ball → sports ball).
    for word in s.split():
        if word in COCO_NAMES:
            return word
        if word in SYNONYMS and SYNONYMS[word]:
            return SYNONYMS[word]
    for name in COCO_NAMES:
        if name in s:
            return name
    return None


class ObjectDetector:
    """YOLOv26s detection ONNX inference (end-to-end head [1, 300, 6]; raw head [1, 84, 8400] + NMS)."""

    def __init__(self, cfg: dict, device: str | None = None):
        y = cfg["yolo_detect"]
        self.imgsz: int = int(y["imgsz"])
        size_npu(cfg, device)
        self.session = build_session(
            y["onnx"],
            device or y["device"],
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "cache"),
            cache_key=y["cache_key"],
        )
        on_npu = "VitisAIExecutionProvider" in self.session.get_providers()
        self.conf_threshold: float = float(
            y.get("npu_conf_threshold", y["conf_threshold"]) if on_npu else y["conf_threshold"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(
        self, frame_bgr: np.ndarray, conf: float | None = None
    ) -> list[Detection]:
        """Run detection on one frame and return all detections above the confidence threshold."""
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        preds, raw = head_rows(self.session.run(None, {self.input_name: nchw})[0], 6)
        if raw:
            boxes = xywh_to_xyxy(preds[:, :4])
            scores, classes = preds[:, 4:].max(axis=1), preds[:, 4:].argmax(axis=1)
        else:
            boxes, scores, classes = preds[:, :4], preds[:, 4], preds[:, 5]
        mask = scores > conf
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        if raw and len(boxes):
            keep = nms_keep(boxes, scores)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        out: list[Detection] = []
        h, w = frame_bgr.shape[:2]
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            box[[0, 2]] = np.clip((box[[0, 2]] - info.pad_x) / info.ratio, 0, w)
            box[[1, 3]] = np.clip((box[[1, 3]] - info.pad_y) / info.ratio, 0, h)
            idx = int(cls)
            name = COCO_NAMES[idx] if 0 <= idx < len(COCO_NAMES) else f"cls{idx}"
            out.append(Detection(name=name, score=float(score), box=box))
        return out

    def find(
        self, frame_bgr: np.ndarray, spoken_name: str, conf: float | None = None
    ) -> tuple[list[Detection], str | None]:
        """All detections matching a spoken object name.

        Returns ``(matches, resolved_class)``; ``resolved_class`` is ``None``
        when the spoken name doesn't map to any known class.
        """
        cls = resolve_class(spoken_name)
        if cls is None:
            return [], None
        dets = [d for d in self.infer(frame_bgr, conf) if d.name == cls]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, cls

    def draw(self, frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
        """Draw detection boxes and labels onto frame."""
        import cv2

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


# =============================================================================
# Standalone component test (live overlay)
# =============================================================================


def main() -> None:
    """CLI entry point: live camera overlay of detections, optionally filtered by --find."""
    import cv2

    from vla_pipeline.vision.camera import make_camera

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="YOLOv26s detection on Ryzen AI NPU - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--role", choices=["operator", "arm"], default="arm")
    parser.add_argument(
        "--find", default=None, help="highlight matches for a spoken name"
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    det = ObjectDetector(cfg, device=args.device)
    cam = make_camera(cfg, args.role)
    fps_window: deque[float] = deque(maxlen=30)
    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            if args.find:
                dets, cls = det.find(frame, args.find)
                label = f"find '{args.find}' → {cls}: {len(dets)}"
            else:
                dets = det.infer(frame)
                label = f"{len(dets)} objects"
            fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            out = det.draw(frame, dets)
            cv2.putText(
                out,
                f"{label}  {sum(fps_window)/len(fps_window):5.1f} FPS",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("yolo26s detect", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
