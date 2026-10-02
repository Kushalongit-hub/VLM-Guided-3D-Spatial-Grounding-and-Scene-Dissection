"""Paired box/mask geometry comparison from an existing fixed detector run."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    if source["summary"]["class_aware_matched_count"] <= 0:
        raise ValueError("Expected class-aware matched instances")
    rows = []
    by_scene = defaultdict(list)
    for scene in source["scenes"]:
        score = scene["class_aware"]
        errors = score["center_errors_m"]
        detections = {obj["instance_id"]: obj for obj in scene["detections"]}
        for match in score["matches"]:
            ref_id = match["reference_id"]
            box = errors["predicted_boxes"][ref_id]
            mask = errors["predicted_masks"][ref_id]
            row = {"scene_id": scene["scene_id"], "reference_id": ref_id,
                   "predicted_label": detections[match["prediction_id"]]["label"],
                   "box_iou": match["iou"], "box_error_m": box,
                   "mask_error_m": mask, "mask_minus_box_error_m": mask - box}
            rows.append(row)
            by_scene[scene["scene_id"]].append(row)
    if len(rows) != source["summary"]["class_aware_matched_count"]:
        raise ValueError("Matched object count differs from source summary")
    differences = np.asarray([row["mask_minus_box_error_m"] for row in rows])
    # Resample whole scenes; their matched-object counts stay intact.
    scenes = list(by_scene.values())
    rng = np.random.default_rng(20261002)
    draws = []
    for _ in range(10000):
        sampled = [row for index in rng.integers(len(scenes), size=len(scenes))
                   for row in scenes[index]]
        draws.append(float(np.mean([row["mask_minus_box_error_m"] for row in sampled])))
    report = {
        "setting": "post-hoc paired YOLO11n-seg box-vs-mask analysis of existing SUN RGB-D heldout20 report",
        "matched_object_count": len(rows), "scene_count_with_matches": len(scenes),
        "mean_box_error_m": float(np.mean([row["box_error_m"] for row in rows])),
        "mean_mask_error_m": float(np.mean([row["mask_error_m"] for row in rows])),
        "mean_mask_minus_box_error_m": float(differences.mean()),
        "median_mask_minus_box_error_m": float(np.median(differences)),
        "mask_better_count": int(np.sum(differences < 0)),
        "scene_bootstrap_95pct_interval_m": np.percentile(draws, [2.5, 97.5]).tolist(),
        "per_label": {
            label: {"count": len(matched),
                    "mean_mask_minus_box_error_m": float(np.mean([row["mask_minus_box_error_m"]
                                                              for row in matched])),
                    "mask_better_count": sum(row["mask_minus_box_error_m"] < 0 for row in matched)}
            for label in sorted({row["predicted_label"] for row in rows})
            for matched in [[row for row in rows if row["predicted_label"] == label]]
        },
        "note": "Only 2D class-aware IoU>=0.5 matches; same YOLO detection boxes and predicted metric depth for both methods. No new scenes or checkpoint; source report was previously inspected. Scene bootstrap is descriptive, not independent-room uncertainty.",
        "objects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "objects"}, indent=2))


if __name__ == "__main__":
    main()
