"""Visible mask contact evidence for a proposed child/support pair."""

from __future__ import annotations

import numpy as np


def visible_contact_fraction(child_mask: np.ndarray, parent_mask: np.ndarray,
                             vertical_radius_px: int = 8,
                             bottom_fraction: float = 0.2) -> dict:
    """Fraction of lower-contour child columns adjacent to parent-mask pixels.

    This tests visible 2D adjacency, not physical 3D contact. Occlusion,
    incomplete masks and vertical camera angle can make true supports score 0.
    """
    child = np.asarray(child_mask, dtype=bool)
    parent = np.asarray(parent_mask, dtype=bool)
    if child.ndim != 2 or parent.shape != child.shape:
        raise ValueError("Child and parent masks must share an HxW shape")
    if vertical_radius_px < 1 or not 0 < bottom_fraction <= 1:
        raise ValueError("Invalid radius or bottom fraction")
    ys, xs = np.nonzero(child)
    if xs.size == 0 or not parent.any():
        return {"contact_fraction": 0.0, "contact_columns": 0, "sampled_columns": 0}
    cutoff = float(np.quantile(ys, 1 - bottom_fraction))
    lower = ys >= cutoff
    sampled_x = np.unique(xs[lower])
    if sampled_x.size == 0:
        return {"contact_fraction": 0.0, "contact_columns": 0, "sampled_columns": 0}
    hits = 0
    height = child.shape[0]
    for x in sampled_x:
        bottom_y = int(np.max(ys[(xs == x) & lower]))
        y0 = max(0, bottom_y - vertical_radius_px)
        y1 = min(height, bottom_y + vertical_radius_px + 1)
        hits += bool(parent[y0:y1, x].any())
    return {"contact_fraction": float(hits / sampled_x.size),
            "contact_columns": int(hits), "sampled_columns": int(sampled_x.size)}
