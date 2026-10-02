"""Support-constrained correction along camera rays, including depth."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def correct_radial_support_layout(layout: dict, child_id: str, parent_id: str,
                                  up: np.ndarray, offset_m: float = 0.4,
                                  child_sigma_m: float = 0.4,
                                  parent_sigma_m: float = 0.4,
                                  relation_sigma_m: float = 0.2,
                                  max_shift_m: float = 0.5) -> dict:
    """Move two estimated centers along camera rays to reduce a support gap.

    The weighted least-squares objective penalizes radial displacements and
    the residual child-minus-parent height gap. A shift cap avoids extreme
    changes when the rays are nearly orthogonal to camera up.
    """
    axis = np.asarray(up, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("up must be a finite nonzero 3-vector")
    axis /= np.linalg.norm(axis)
    values = (offset_m, child_sigma_m, parent_sigma_m, relation_sigma_m, max_shift_m)
    if not np.isfinite(values).all() or any(v <= 0 for v in values[1:]):
        raise ValueError("Offset must be finite; uncertainties and cap must be positive")
    result = deepcopy(layout)
    by_id = {str(obj["instance_id"]): obj for obj in result["objects"]}
    if child_id not in by_id or parent_id not in by_id or child_id == parent_id:
        raise ValueError("Unknown or identical child/parent IDs")
    child = np.asarray(by_id[child_id]["center_point_3d"], dtype=float)
    parent = np.asarray(by_id[parent_id]["center_point_3d"], dtype=float)
    if (child.shape != (3,) or parent.shape != (3,) or
            not np.isfinite(child).all() or not np.isfinite(parent).all() or
            child[2] <= 0 or parent[2] <= 0):
        raise ValueError("Centers must be finite points in front of the camera")
    child_ray = child / np.linalg.norm(child)
    parent_ray = parent / np.linalg.norm(parent)
    coefficients = np.array([float(child_ray @ axis), -float(parent_ray @ axis)])
    variance = np.square([child_sigma_m, parent_sigma_m])
    innovation = offset_m - float((child - parent) @ axis)
    denominator = float(np.sum(np.square(coefficients) * variance) + relation_sigma_m**2)
    shifts = np.clip(variance * coefficients * innovation / denominator,
                     -max_shift_m, max_shift_m)
    for obj, old, direction, shift in (
            (by_id[child_id], child, child_ray, shifts[0]),
            (by_id[parent_id], parent, parent_ray, shifts[1])):
        new = old + direction * shift
        if new[2] <= 0:
            raise ValueError("Correction moved an object behind the camera")
        obj["uncorrected_center_point_3d"] = old.tolist()
        obj["center_point_3d"] = new.tolist()
        obj["support_radial_shift_m"] = float(shift)
    new_child = np.asarray(by_id[child_id]["center_point_3d"])
    new_parent = np.asarray(by_id[parent_id]["center_point_3d"])
    result["radial_support_correction"] = {
        "child_id": child_id, "parent_id": parent_id,
        "radial_shifts_m": shifts.tolist(),
        "residual_m": float((new_child - new_parent) @ axis - offset_m),
        "offset_m": float(offset_m), "relation_sigma_m": float(relation_sigma_m),
        "max_shift_m": float(max_shift_m),
    }
    return result
