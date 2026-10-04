"""Configurable child labels for the existing conservative support heuristic."""

from __future__ import annotations

import numpy as np

from .support_proposals import PARENT_LABELS


def propose_labeled_support_relations(layout: dict, child_labels: set[str],
                                      max_depth_gap_m: float = 0.75) -> list[dict]:
    """Choose one parent per child using box footpoints and median depth.

    These are candidates, not verified support relations. The original
    cup/mug implementation remains frozen for earlier evaluations.
    """
    if not child_labels or any(not label or not isinstance(label, str) for label in child_labels):
        raise ValueError("child_labels must contain nonempty strings")
    if not np.isfinite(max_depth_gap_m) or max_depth_gap_m <= 0:
        raise ValueError("max_depth_gap_m must be positive and finite")
    objects = layout["objects"]
    ids = [str(obj["instance_id"]) for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object instance_id")
    for obj in objects:
        box = np.asarray(obj["bbox_2d"], dtype=float)
        center = np.asarray(obj["center_point_3d"], dtype=float)
        if box.shape != (4,) or center.shape != (3,) or not np.isfinite(box).all() or not np.isfinite(center).all():
            raise ValueError("Each object needs a finite bbox_2d and center_point_3d")
        if box[0] >= box[2] or box[1] >= box[3] or center[2] <= 0:
            raise ValueError("Invalid object box or depth")
    proposals = []
    for child in objects:
        if child["object"] not in child_labels:
            continue
        xmin, _, xmax, ymax = child["bbox_2d"]
        foot_x, foot_y = (xmin + xmax) / 2, ymax
        candidates = []
        for parent in objects:
            if parent["instance_id"] == child["instance_id"] or parent["object"] not in PARENT_LABELS:
                continue
            left, top, right, bottom = parent["bbox_2d"]
            if not (left <= foot_x < right and top <= foot_y <= bottom):
                continue
            gap = abs(float(child["center_point_3d"][2] - parent["center_point_3d"][2]))
            if gap <= max_depth_gap_m:
                candidates.append((gap, str(parent["instance_id"])))
        if candidates:
            gap, parent_id = min(candidates)
            proposals.append({"child_id": str(child["instance_id"]), "parent_id": parent_id,
                              "relation": "supported_by_candidate", "depth_gap_m": gap,
                              "footpoint_px": [float(foot_x), float(foot_y)]})
    return proposals
