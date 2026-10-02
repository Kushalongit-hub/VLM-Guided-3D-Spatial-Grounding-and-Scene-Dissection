"""Geometry baseline using metric depth and pixel-space object regions.

Camera coordinates use X right, Y down, Z forward; distances are metres.
The visible point extent is not a full 3D object dimension estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not all(np.isfinite((self.fx, self.fy, self.cx, self.cy))):
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")


def lift_region(depth: np.ndarray, bbox: tuple[int, int, int, int], camera: Intrinsics,
                mask: np.ndarray | None = None) -> dict:
    """Lift a [xmin, ymin, xmax, ymax] box; report robust visible extents."""
    if depth.ndim != 2:
        raise ValueError("Depth must be an HxW array")
    height, width = depth.shape
    xmin, ymin, xmax, ymax = map(int, bbox)
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"Invalid pixel bbox: {bbox}")
    patch = depth[ymin:ymax, xmin:xmax]
    valid = np.isfinite(patch) & (patch > 0)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("Object mask must match depth image shape")
        valid &= mask[ymin:ymax, xmin:xmax].astype(bool)
    if not valid.any():
        raise ValueError("Object region has no valid metric depth")
    local_y, local_x = np.nonzero(valid)
    z = patch[valid].astype(np.float64)
    x = (local_x + xmin - camera.cx) * z / camera.fx
    y = (local_y + ymin - camera.cy) * z / camera.fy
    points = np.column_stack((x, y, z))
    lo, hi = np.percentile(points, [5, 95], axis=0)
    return {
        "center_point_3d": np.median(points, axis=0).tolist(),
        "visible_extent_3d": (hi - lo).tolist(),
        "valid_depth_pixels": int(points.shape[0]),
        "bbox_2d": [xmin, ymin, xmax, ymax],
    }


def build_layout(depth: np.ndarray, objects: list[dict], camera: Intrinsics,
                 masks: dict[str, np.ndarray] | None = None) -> dict:
    """Build a reproducible baseline layout from externally supplied boxes."""
    output = []
    for index, obj in enumerate(objects):
        instance_id = str(obj.get("instance_id", f"object_{index}"))
        mask = masks.get(instance_id) if masks is not None else None
        if masks is not None and mask is None:
            raise ValueError(f"Missing mask for {instance_id}")
        region = lift_region(depth, tuple(obj["bbox_2d"]), camera, mask=mask)
        entry = {"object": str(obj["label"]), "instance_id": instance_id, **region}
        entry["region_type"] = "mask" if mask is not None else "box"
        if "confidence" in obj:
            entry["detection_confidence"] = float(obj["confidence"])
        output.append(entry)
    return {
        "coordinate_system": "camera_x_right_y_down_z_forward_meters",
        "camera_intrinsics": vars(camera),
        "objects": output,
    }
