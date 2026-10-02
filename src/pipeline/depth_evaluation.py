"""Pixel-aligned metric-depth evaluation on valid reference pixels."""

import numpy as np


def depth_metrics(predicted: np.ndarray, reference: np.ndarray,
                  min_depth: float = 0.1, max_depth: float = 7.99) -> dict:
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("Predicted and reference depths must have the same HxW shape")
    valid = (np.isfinite(predicted) & np.isfinite(reference) &
             (reference >= min_depth) & (reference <= max_depth) & (predicted > 0))
    if not np.any(valid):
        raise ValueError("No valid reference depth pixels")
    pred, truth = predicted[valid].astype(np.float64), reference[valid].astype(np.float64)
    error = np.abs(pred - truth)
    ratio = np.maximum(pred / truth, truth / pred)
    return {"valid_pixels": int(valid.sum()), "coverage": float(valid.mean()),
            "mae_m": float(error.mean()), "median_ae_m": float(np.median(error)),
            "rmse_m": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "abs_rel": float(np.mean(error / truth)),
            "delta_1": float(np.mean(ratio < 1.25))}
