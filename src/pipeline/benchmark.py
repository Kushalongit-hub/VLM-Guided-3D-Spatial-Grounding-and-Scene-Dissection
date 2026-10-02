"""Evaluate a frozen collection of scenes with metric depth and 3D centers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate_centroids
from .geometry import Intrinsics, build_layout


def run_manifest(manifest_path: Path) -> dict:
    """Run scene records from JSON, resolving paths relative to the manifest.

    Each record contains scene_id, depth, objects, reference, and intrinsics.
    The objects file supplies oracle 2D boxes; scores therefore measure the
    geometry stage only and must not be reported as end-to-end RGB performance.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest["scenes"]
    if not scenes:
        raise ValueError("Manifest contains no scenes")
    base = manifest_path.parent
    results = []
    all_errors = []
    total_ref = total_matched = 0
    seen_ids = set()
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        if scene_id in seen_ids:
            raise ValueError(f"Duplicate scene_id: {scene_id}")
        seen_ids.add(scene_id)
        depth = np.load(base / scene["depth"], allow_pickle=False)
        objects = json.loads((base / scene["objects"]).read_text(encoding="utf-8"))
        reference = json.loads((base / scene["reference"]).read_text(encoding="utf-8"))
        layout = build_layout(depth, objects, Intrinsics(**scene["intrinsics"]))
        score = evaluate_centroids(layout, reference)
        total_ref += score["reference_count"]
        total_matched += score["matched_count"]
        all_errors.extend(item["centroid_error_m"] for item in score["per_object"])
        results.append({"scene_id": scene_id, **score})
    return {
        "scene_count": len(results),
        "reference_count": total_ref,
        "matched_count": total_matched,
        "coverage": total_matched / total_ref,
        "median_centroid_error_m": float(np.median(all_errors)) if all_errors else None,
        "p90_centroid_error_m": float(np.percentile(all_errors, 90)) if all_errors else None,
        "scenes": results,
    }
