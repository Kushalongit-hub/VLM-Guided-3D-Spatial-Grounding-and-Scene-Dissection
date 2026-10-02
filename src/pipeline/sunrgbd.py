"""Convert one SUN RGB-D scene using VoteNet's extracted labels/calibration.

Inputs are the original encoded depth PNG and label/calibration TXT files
produced by VoteNet's SUN RGB-D v2 MATLAB extraction script.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


COORDINATE_SYSTEM = "camera_x_right_y_down_z_forward_meters"


def decode_depth_png(path: Path) -> np.ndarray:
    """Undo the SUN RGB-D 3-bit rotation and convert millimetres to metres."""
    packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if packed is None or packed.dtype != np.uint16 or packed.ndim != 2:
        raise ValueError("Expected a readable 16-bit single-channel SUN RGB-D PNG")
    decoded = ((packed.astype(np.uint32) >> 3) | ((packed.astype(np.uint32) << 13) & 0xFFFF))
    depth = decoded.astype(np.float32) / 1000.0
    depth[depth > 8.0] = 8.0  # Match the dataset toolbox read3dPoints.m.
    return depth


def read_calibration(path: Path) -> tuple[np.ndarray, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Calibration must contain Rtilt and K lines")
    rtilt = np.array([float(v) for v in lines[0].split()], dtype=float).reshape(3, 3, order="F")
    k = np.array([float(v) for v in lines[1].split()], dtype=float).reshape(3, 3, order="F")
    if not np.isfinite(rtilt).all() or not np.isfinite(k).all() or abs(np.linalg.det(rtilt)) < 1e-8:
        raise ValueError("Invalid calibration matrices")
    return rtilt, {"fx": float(k[0, 0]), "fy": float(k[1, 1]),
                   "cx": float(k[0, 2]), "cy": float(k[1, 2])}


def convert_labels(path: Path, rtilt: np.ndarray, image_shape: tuple[int, int]) -> tuple[list[dict], dict]:
    height, width = image_shape
    objects = []
    references = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 13:
            raise ValueError(f"Malformed object label at line {line_number}")
        label = parts[0]
        x, y, w, h = map(float, parts[1:5])
        xmin, ymin = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        xmax, ymax = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
        if xmin >= xmax or ymin >= ymax:
            continue  # Annotation is outside the image.
        upright_center = np.array([float(v) for v in parts[5:8]], dtype=float)
        if not np.isfinite(upright_center).all():
            raise ValueError(f"Invalid 3D center at line {line_number}")
        depth_frame = np.linalg.solve(rtilt, upright_center)
        camera_center = [float(depth_frame[0]), float(-depth_frame[2]), float(depth_frame[1])]
        instance_id = f"object_{line_number - 1}"
        objects.append({"label": label, "instance_id": instance_id,
                        "bbox_2d": [xmin, ymin, xmax, ymax]})
        references.append({"instance_id": instance_id, "center_point_3d": camera_center})
    return objects, {"coordinate_system": COORDINATE_SYSTEM, "objects": references}
