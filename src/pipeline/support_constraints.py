"""Uncertainty-weighted correction of object centers using support contacts.

Relations and contact offsets must be supplied independently of 3D ground truth.
This stage adjusts center heights only; it does not infer support from a 2D box.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def camera_up_from_rtilt(rtilt: np.ndarray) -> np.ndarray:
    """Convert SUN RGB-D's upright Z axis to camera X-right/Y-down/Z-forward."""
    matrix = np.asarray(rtilt, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Rtilt must be an invertible finite 3x3 matrix")
    # camera -> raw SUN depth frame: (X, Z, -Y)
    raw_up = np.linalg.solve(matrix, np.array([0.0, 0.0, 1.0]))
    camera_up = np.array([raw_up[0], -raw_up[2], raw_up[1]])
    return camera_up / np.linalg.norm(camera_up)


def correct_support_layout(layout: dict, relations: list[dict],
                           center_sigma_m: dict[str, float],
                           up: np.ndarray | None = None,
                           hard: bool = False) -> dict:
    """Fit center heights to child-on-parent offsets with Gaussian weights.

    Each relation has child_id, parent_id, offset_m (child center height minus
    parent center height), and sigma_m. Hard mode enforces exact offsets and is
    a comparison baseline. No relation is inferred by this function.
    """
    result = deepcopy(layout)
    objects = result["objects"]
    ids = [str(obj["instance_id"]) for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object instance_id")
    indices = {key: index for index, key in enumerate(ids)}
    axis = np.array([0.0, -1.0, 0.0] if up is None else up, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("Up direction must be a finite nonzero 3-vector")
    axis /= np.linalg.norm(axis)
    centers = np.array([obj["center_point_3d"] for obj in objects], dtype=float)
    if centers.shape != (len(objects), 3) or not np.isfinite(centers).all():
        raise ValueError("Object centers must be finite 3-vectors")
    sigmas = np.array([center_sigma_m[key] for key in ids], dtype=float)
    if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError("Center uncertainties must be positive and finite")
    rows, offsets, relation_sigmas = [], [], []
    seen = set()
    for relation in relations:
        child, parent = str(relation["child_id"]), str(relation["parent_id"])
        if child not in indices or parent not in indices or child == parent:
            raise ValueError("Support relation needs two distinct known objects")
        if (child, parent) in seen:
            raise ValueError("Duplicate support relation")
        seen.add((child, parent))
        offset, sigma = float(relation["offset_m"]), float(relation["sigma_m"])
        if not np.isfinite(offset) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Relation offset must be finite and uncertainty positive")
        row = np.zeros(len(ids))
        row[indices[child]], row[indices[parent]] = 1.0, -1.0
        rows.append(row)
        offsets.append(offset)
        relation_sigmas.append(sigma)
    if not rows:
        result["support_correction"] = {"relation_count": 0, "mode": "hard" if hard else "weighted"}
        return result
    a = np.stack(rows)
    target = np.array(offsets)
    old_heights = centers @ axis
    covariance = np.diag(sigmas ** 2)
    innovation = target - a @ old_heights
    relation_covariance = np.zeros((len(rows), len(rows))) if hard else np.diag(np.square(relation_sigmas))
    system = a @ covariance @ a.T + relation_covariance
    try:
        correction = covariance @ a.T @ np.linalg.solve(system, innovation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Conflicting or dependent hard support relations") from exc
    new_centers = centers + correction[:, None] * axis[None, :]
    for obj, old, new, change in zip(objects, centers, new_centers, correction):
        obj["uncorrected_center_point_3d"] = old.tolist()
        obj["center_point_3d"] = new.tolist()
        obj["support_height_shift_m"] = float(change)
    result["support_correction"] = {
        "mode": "hard" if hard else "weighted", "relation_count": len(rows),
        "up_camera": axis.tolist(), "residual_m": (a @ (new_centers @ axis) - target).tolist(),
    }
    return result
