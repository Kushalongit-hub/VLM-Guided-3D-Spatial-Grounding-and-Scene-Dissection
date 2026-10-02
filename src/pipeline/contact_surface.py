"""Estimate a local cup/mug contact height from monocular metric depth.

Development-stage method. It uses a surface strip immediately below the
child box, constrained to the parent box, and does not use reference 3D data.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .geometry import Intrinsics


def local_contact_target(depth: np.ndarray, child: dict, parent: dict,
                         camera: Intrinsics, up: np.ndarray) -> dict | None:
    """Return a child center height target, or None when no local surface exists."""
    if depth.ndim != 2:
        raise ValueError("Depth must be HxW")
    axis = np.asarray(up, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError("up must be a finite nonzero 3-vector")
    axis = axis / np.linalg.norm(axis)
    cx0, cy0, cx1, cy1 = map(int, child["bbox_2d"])
    px0, py0, px1, py1 = map(int, parent["bbox_2d"])
    height, width = depth.shape
    if not (0 <= cx0 < cx1 <= width and 0 <= cy0 < cy1 <= height and
            0 <= px0 < px1 <= width and 0 <= py0 < py1 <= height):
        raise ValueError("Invalid child or parent box")
    child_center = np.asarray(child["center_point_3d"], dtype=float)
    if child_center.shape != (3,) or not np.isfinite(child_center).all() or child_center[2] <= 0:
        raise ValueError("Invalid child center")
    box_width, box_height = cx1 - cx0, cy1 - cy0
    x0 = max(px0, cx0 - box_width // 2)
    x1 = min(px1, cx1 + box_width // 2)
    y0 = max(py0, cy1)
    y1 = min(py1, cy1 + max(4, int(round(0.4 * box_height))))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0) & (np.abs(patch - child_center[2]) <= 0.75)
    if np.count_nonzero(valid) < 12:
        return None
    local_y, local_x = np.nonzero(valid)
    z = patch[valid].astype(float)
    x = (local_x + x0 - camera.cx) * z / camera.fx
    y = (local_y + y0 - camera.cy) * z / camera.fy
    surface_height = float(np.median(np.column_stack((x, y, z)) @ axis))
    child_height = float(np.clip(box_height * child_center[2] / camera.fy, 0.05, 0.35))
    return {"target_child_height_m": surface_height + child_height / 2,
            "surface_height_m": surface_height, "child_height_estimate_m": child_height,
            "surface_pixel_count": int(np.count_nonzero(valid))}


def correct_child_to_local_surface(layout: dict, depth: np.ndarray, child_id: str,
                                   parent_id: str, up: np.ndarray,
                                   blend: float = 0.5) -> dict:
    """Blend a child center toward estimated local support-surface contact."""
    if not 0 <= blend <= 1:
        raise ValueError("blend must be in [0, 1]")
    result = deepcopy(layout)
    by_id = {str(item["instance_id"]): item for item in result["objects"]}
    if child_id not in by_id or parent_id not in by_id or child_id == parent_id:
        raise ValueError("Unknown or identical child/parent IDs")
    axis = np.asarray(up, dtype=float)
    axis /= np.linalg.norm(axis)
    target = local_contact_target(depth, by_id[child_id], by_id[parent_id],
                                  Intrinsics(**result["camera_intrinsics"]), axis)
    if target is None:
        result["local_contact_correction"] = {"applied": False, "reason": "no_valid_surface_strip"}
        return result
    child = by_id[child_id]
    center = np.asarray(child["center_point_3d"], dtype=float)
    shift = blend * (target["target_child_height_m"] - float(center @ axis))
    child["uncorrected_center_point_3d"] = center.tolist()
    child["center_point_3d"] = (center + shift * axis).tolist()
    child["support_height_shift_m"] = float(shift)
    result["local_contact_correction"] = {"applied": True, "blend": blend, **target}
    return result
