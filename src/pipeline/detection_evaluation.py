"""One-to-one 2D region coverage against uniquely paired SUN RGB-D boxes.

SUN RGB-D's paired 2D/3D labels are incomplete, so unmatched predictions are
not counted as false positives. These metrics are recall proxies, not AP.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# Conservative mappings only. Generic desks/tables, monitors, bins, shelves,
# and sofa chairs have no sufficiently exact COCO equivalent here.
SUN_TO_COCO = {
    "chair": "chair", "bed": "bed", "sofa": "couch", "tv": "tv",
    "toilet": "toilet", "sink": "sink", "bicycle": "bicycle",
    "bench": "bench", "keyboard": "keyboard", "mouse": "mouse",
    "mug": "cup", "back_pack": "backpack",
}


def box_iou(left: list[int], right: list[int]) -> float:
    """Intersection-over-union for half-open [xmin,ymin,xmax,ymax] boxes."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Boxes must contain four finite coordinates")
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        raise ValueError("Boxes must have positive width and height")
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0, hi - lo)))
    area_a = float(np.prod(a[2:] - a[:2]))
    area_b = float(np.prod(b[2:] - b[:2]))
    return intersection / (area_a + area_b - intersection)


def match_boxes(reference: list[dict], predicted: list[dict],
                threshold: float = 0.5, class_aware: bool = False) -> dict:
    """Maximize the count of valid pairs, breaking ties by total IoU."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    for collection in (reference, predicted):
        ids = [item["instance_id"] for item in collection]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate object instance_id")
    eligible = [item for item in reference if not class_aware or item["label"] in SUN_TO_COCO]
    matches = []
    if eligible and predicted:
        overlap = np.zeros((len(eligible), len(predicted)), dtype=float)
        for row, truth in enumerate(eligible):
            for col, detection in enumerate(predicted):
                if class_aware and SUN_TO_COCO[truth["label"]] != detection["label"]:
                    continue
                overlap[row, col] = box_iou(truth["bbox_2d"], detection["bbox_2d"])
        valid = overlap >= threshold
        score = valid.astype(float) * (1 + overlap / (min(overlap.shape) + 1))
        rows, cols = linear_sum_assignment(-score)
        matches = [
            {"reference_id": eligible[row]["instance_id"],
             "prediction_id": predicted[col]["instance_id"], "iou": float(overlap[row, col])}
            for row, col in zip(rows, cols) if overlap[row, col] >= threshold
        ]
    return {"eligible_count": len(eligible), "matched_count": len(matches),
            "coverage": len(matches) / len(eligible) if eligible else None,
            "matches": matches}
