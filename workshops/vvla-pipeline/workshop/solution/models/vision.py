# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLOv26s wrappers (PROVIDED) - thin shells around YOUR session factory.

Both vision models are ~20 lines each because everything generic lives in
``common.imaging`` (letterbox, decode, draw) and everything AMD-specific
lives in ``models.npu`` (your TODOs). This is the shape of the whole
workshop: the wrappers are free, the stack is the skill.

Standalone check (after you implement models/npu.py):

    python -m models.vision --which pose --device cpu
"""

from __future__ import annotations

import argparse
import logging
import time

from common.config import load_config
from common.imaging import (decode_detections, decode_pose, draw_detections,
                            draw_skeleton, letterbox, pick_nearest,
                            resolve_class)
from models.npu import build_npu_session, report_placement

logger = logging.getLogger(__name__)

_npu_sized = False
_npu_sizer = None  # holds the NPU buffers for the life of the process


def size_npu(cfg: dict, device=None) -> None:
    """Open the larger YOLO model the config puts on the NPU and keep it
    open, once per process, before any other NPU session.

    The models on the NPU share buffers within a process, created by its
    first NPU session and freed when the last one closes: YOLO-pose opened
    while the detector is open fails with ``HW context creation
    unsuccessful`` (``sub buffer size and offset``). Sized by YOLO-pose,
    both fit, in either order.
    """
    global _npu_sized, _npu_sizer
    if _npu_sized:
        return
    for section in ("yolo_pose", "yolo_detect"):  # largest buffers first
        y = cfg[section]
        if (device or y["device"]) == "npu":
            _npu_sized = True
            _npu_sizer = build_npu_session(
                y["onnx"], "npu",
                vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
                cache_dir=cfg.get("system", {}).get("cache_dir", "../../cache"),
                cache_key=y["cache_key"],
            )
            return


class PoseEstimator:
    """YOLOv26s-pose (end-to-end head [1, 300, 57], no NMS) on NPU or CPU."""

    def __init__(self, cfg: dict, device=None):
        y = cfg["yolo_pose"]
        self.imgsz = int(y["imgsz"])
        self.conf_threshold = float(y["conf_threshold"])
        self.kpt_threshold = float(y["kpt_threshold"])
        requested = device or y["device"]
        size_npu(cfg, device)
        self.session = build_npu_session(
            y["onnx"], requested,
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "../../cache"),
            cache_key=y["cache_key"],
        )
        self.provider = report_placement(self.session, requested, "yolo26s-pose")
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr, conf=None):
        """→ ``boxes [N,4]``, ``scores [N]``, ``kpts [N,17,3]`` (orig pixels)."""
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        outputs = self.session.run(None, {self.input_name: nchw})
        return decode_pose(outputs, info, frame_bgr.shape, conf)

    def primary_person(self, frame_bgr):
        """``(kpts[17,3], score)`` for the nearest (largest-box) person, or None.

        Only the person closest to the camera drives the arm; ``pick_nearest``
        selects them by box area (the proximity proxy).
        """
        boxes, scores, kpts = self.infer(frame_bgr)
        if len(boxes) == 0:
            return None
        i = pick_nearest(boxes)
        return kpts[i], float(scores[i])

    def draw(self, frame, boxes, scores, kpts):
        """Render keypoints and skeleton lines over ``frame``."""
        return draw_skeleton(frame, boxes, scores, kpts, self.kpt_threshold)


class ObjectDetector:
    """YOLOv26s detection (end-to-end head [1, 300, 6]) on NPU or CPU."""

    def __init__(self, cfg: dict, device=None):
        y = cfg["yolo_detect"]
        self.imgsz = int(y["imgsz"])
        requested = device or y["device"]
        size_npu(cfg, device)
        self.session = build_npu_session(
            y["onnx"], requested,
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "../../cache"),
            cache_key=y["cache_key"],
        )
        self.provider = report_placement(self.session, requested, "yolo26s-detect")
        on_npu = "VitisAIExecutionProvider" in self.session.get_providers()
        self.conf_threshold = float(
            y.get("npu_conf_threshold", y["conf_threshold"]) if on_npu else y["conf_threshold"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr, conf=None):
        """Run the detector and decode raw outputs into detections for this frame."""
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        outputs = self.session.run(None, {self.input_name: nchw})
        return decode_detections(outputs, info, frame_bgr.shape, conf)

    def find(self, frame_bgr, spoken_name: str, conf=None):
        """All detections matching a spoken name → ``(matches, class|None)``."""
        cls = resolve_class(spoken_name)
        if cls is None:
            return [], None
        dets = [d for d in self.infer(frame_bgr, conf) if d.name == cls]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, cls

    def draw(self, frame, dets):
        """Render detection boxes over ``frame``."""
        return draw_detections(frame, dets)


# =============================================================================
# Standalone component test (webcam overlay, degrades to synthetic frames)
# =============================================================================

def main() -> None:
    """CLI: run pose or detection on a webcam (or synthetic frames) and report FPS."""
    import cv2

    from common.fixtures import SyntheticCamera

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="YOLO on Ryzen AI - component test")
    parser.add_argument("--which", choices=["pose", "detect"], default="pose")
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="headless N frames")
    args = parser.parse_args()

    cfg = load_config()
    model = (PoseEstimator if args.which == "pose" else ObjectDetector)(cfg, args.device)
    print(f"Session ready - provider: {model.provider}")

    cap = cv2.VideoCapture(args.cam)
    source = cap if cap.isOpened() else SyntheticCamera()
    if source is not cap:
        print("(no webcam - using the synthetic camera)")

    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = source.read()
        if not ok:
            break
        result = model.infer(frame)
        out = (model.draw(frame, *result) if args.which == "pose"
               else model.draw(frame, result))
        n += 1
        if args.frames:
            if n >= args.frames:
                fps = n / (time.perf_counter() - t0)
                print(f"{n} frames, {fps:.1f} FPS on {model.provider}")
                break
        else:
            cv2.imshow(f"yolo26s-{args.which}", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
