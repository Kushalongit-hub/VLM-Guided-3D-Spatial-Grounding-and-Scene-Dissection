"""Soft support correction of a child center along its camera ray."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def correct_child_ray_support_layout(layout: dict, child_id: str, parent_id: str,
                                     up: np.ndarray, offset_m: float = 0.4,
                                     child_sigma_m: float = 0.4,
                                     relation_sigma_m: float = 0.2,
                                     max_shift_m: float = 0.5) -> dict:
    """Adjust only child depth using a soft center-height-gap prior.

    This post-hoc ablation is deliberately simpler than the frozen paired-ray
    method. It keeps the parent center fixed and restricts the child center to
    its original camera ray. The offset remains a supplied prior in metres.
    """
    axis = np.asarray(up, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("up must be a finite nonzero 3-vector")
    axis = axis / np.linalg.norm(axis)
    params = (offset_m, child_sigma_m, relation_sigma_m, max_shift_m)
    if not np.isfinite(params).all() or any(value <= 0 for value in params[1:]):
        raise ValueError("Offset must be finite; uncertainties and cap must be positive")
    result = deepcopy(layout)
    by_id = {str(obj["instance_id"]): obj for obj in result["objects"]}
    if len(by_id) != len(result["objects"]):
        raise ValueError("Duplicate object IDs")
    if child_id == parent_id or child_id not in by_id or parent_id not in by_id:
        raise ValueError("Unknown or identical child/parent IDs")
    child = np.asarray(by_id[child_id]["center_point_3d"], dtype=float)
    parent = np.asarray(by_id[parent_id]["center_point_3d"], dtype=float)
    if (child.shape != (3,) or parent.shape != (3,) or
            not np.isfinite(child).all() or not np.isfinite(parent).all() or
            child[2] <= 0 or parent[2] <= 0):
        raise ValueError("Centers must be finite points in front of the camera")
    ray = child / np.linalg.norm(child)
    projection = float(ray @ axis)
    innovation = offset_m - float((child - parent) @ axis)
    shift = float(np.clip(child_sigma_m**2 * projection * innovation /
                          (child_sigma_m**2 * projection**2 + relation_sigma_m**2),
                          -max_shift_m, max_shift_m))
    corrected = child + ray * shift
    if corrected[2] <= 0:
        raise ValueError("Correction moved the child behind the camera")
    by_id[child_id]["uncorrected_center_point_3d"] = child.tolist()
    by_id[child_id]["center_point_3d"] = corrected.tolist()
    by_id[child_id]["support_radial_shift_m"] = shift
    result["child_ray_support_correction"] = {
        "child_id": child_id, "parent_id": parent_id,
        "radial_shift_m": shift,
        "residual_m": float((corrected - parent) @ axis - offset_m),
        "offset_m": float(offset_m), "relation_sigma_m": float(relation_sigma_m),
        "max_shift_m": float(max_shift_m),
    }
    return result
