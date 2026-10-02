"""Scene-level centroid evaluation for matched object instances."""

from __future__ import annotations

import numpy as np


def evaluate_centroids(prediction: dict, reference: dict) -> dict:
    """Compare camera-frame metric centers matched by unique instance_id.

    Both inputs use the layout's `objects` list. Missing predictions are counted
    separately and never silently excluded from the reported coverage.
    """
    if prediction.get("coordinate_system") != reference.get("coordinate_system"):
        raise ValueError("Prediction and reference coordinate systems differ")

    def index(layout: dict) -> dict:
        result = {}
        for obj in layout["objects"]:
            key = obj["instance_id"]
            if key in result:
                raise ValueError(f"Duplicate instance_id: {key}")
            result[key] = obj
        return result

    pred, truth = index(prediction), index(reference)
    if not truth:
        raise ValueError("Reference has no objects")
    errors = []
    per_object = []
    for key, target in truth.items():
        if key not in pred:
            continue
        actual = np.asarray(pred[key]["center_point_3d"], dtype=float)
        expected = np.asarray(target["center_point_3d"], dtype=float)
        if actual.shape != (3,) or expected.shape != (3,) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"Invalid center_point_3d for {key}")
        error = float(np.linalg.norm(actual - expected))
        errors.append(error)
        per_object.append({"instance_id": key, "centroid_error_m": error})
    return {
        "reference_count": len(truth),
        "matched_count": len(errors),
        "coverage": len(errors) / len(truth),
        "median_centroid_error_m": float(np.median(errors)) if errors else None,
        "p90_centroid_error_m": float(np.percentile(errors, 90)) if errors else None,
        "per_object": per_object,
    }
