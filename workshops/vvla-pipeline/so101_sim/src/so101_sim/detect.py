# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Ground-truth object detection from the twin's segmentation render.

Stand-in for the YOLO detector in scripted tests: the boxes come from the
per-pixel body ids rendered alongside the very frame the behavior is looking
at, so they are exact, occlusion-aware, and deterministic. Outputs plain
tuples; host stacks wrap them in their own ``Detection`` type.
"""

from __future__ import annotations

import numpy as np

# twin body -> COCO class name the real detector would report
DEFAULT_CLASSES = {
    "sports_ball": "sports ball",
    "book": "book",
    "bottle": "bottle",
    "toothbrush": "toothbrush",
}


class SegDetector:
    """Detect twin objects in frames returned by a segmentation-enabled SimCamera."""

    def __init__(self, camera, classes: dict | None = None, min_pixels: int = 25):
        self.camera = camera
        self.classes = dict(classes or DEFAULT_CLASSES)
        self.min_pixels = int(min_pixels)
        model = camera.world.model
        self._body_ids = {}
        for body in self.classes:
            try:
                self._body_ids[body] = model.body(body).id
            except KeyError:
                continue

    def detect(self, frame=None) -> list[tuple[str, float, tuple[float, float, float, float], str]]:
        """``[(class_name, score, (x1, y1, x2, y2), body), ...]`` sorted by score."""
        seg = self.camera.segmentation(frame)
        if seg is None:
            return []
        out = []
        for body, bid in self._body_ids.items():
            mask = seg == bid
            n = int(mask.sum())
            if n < self.min_pixels:
                continue
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            area = (box[2] - box[0]) * (box[3] - box[1])
            fill = n / max(area, 1.0)  # occlusion lowers confidence like a real model
            score = float(min(0.99, 0.55 + 0.25 * min(1.0, n / 1500.0) + 0.2 * fill))
            out.append((self.classes[body], score, box, body))
        out.sort(key=lambda d: d[1], reverse=True)
        return out
