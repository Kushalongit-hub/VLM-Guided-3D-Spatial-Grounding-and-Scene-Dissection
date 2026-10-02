"""Evaluate frozen, visually checked cup/mug-on-table support candidates.

Uses oracle boxes and manually labeled relations. It does not evaluate a
relation detector or a fully automatic RGB-to-3D pipeline.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


def evaluate(manifest_path: Path, labels_path: Path, protocol_path: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(labels_path.read_text(encoding="utf-8"))
    if annotations["status"] != "rgb_and_2d_box_visual_labels_frozen_before_depth_inference":
        raise ValueError("Expected pre-inference RGB-only labels")
    scenes = {item["scene_id"]: item for item in manifest["scenes"]}
    labels = {item["scene_id"]: item for item in annotations["scenes"]}
    if len(scenes) != len(manifest["scenes"]) or len(labels) != len(annotations["scenes"]):
        raise ValueError("Duplicate scene ID")
    if scenes.keys() != labels.keys():
        raise ValueError("Manifest and frozen annotation scene sets differ")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path else annotations
    if protocol_path and protocol["status"] != "frozen_before_confirmation_images_or_depth_inference":
        raise ValueError("Expected a pre-frozen confirmation protocol")
    prior = protocol["center_sigma_m"]
    outcomes = []
    for scene_id, scene in scenes.items():
        annotation = labels[scene_id]
        support_label = annotation["supported_by_table"]
        if support_label is not None and not isinstance(support_label, bool):
            raise ValueError(f"Invalid support label in {scene_id}")
        root = manifest_path.parent
        scene_dir = root / scene_id
        objects = json.loads((root / scene["objects"]).read_text(encoding="utf-8"))
        by_id = {item["instance_id"]: item for item in objects}
        child, parent = annotation["child_id"], annotation["parent_id"]
        if child not in by_id or parent not in by_id or child == parent:
            raise ValueError(f"Invalid object IDs in {scene_id}")
        if by_id[child]["label"] not in ("mug", "cup") or by_id[parent]["label"] != "table":
            raise ValueError(f"Unexpected object labels in {scene_id}")
        if scene["depth"] != f"{scene_id}/predicted_depth.npy":
            raise ValueError("Evaluation requires predicted RGB depth")
        rtilt, _ = read_calibration(scene_dir / "calib.txt")
        up = camera_up_from_rtilt(rtilt)
        baseline = build_layout(np.load(root / scene["depth"], allow_pickle=False),
                                objects, Intrinsics(**scene["intrinsics"]))
        sigma = {item["instance_id"]: prior["cup_or_mug"] if item["instance_id"] == child
                 else prior["table"] if item["instance_id"] == parent else prior["other"]
                 for item in objects}
        relation = [{"child_id": child, "parent_id": parent,
                     "offset_m": protocol["offset_prior_m"],
                     "sigma_m": protocol["offset_sigma_m"]}] if support_label is True else []
        outputs = {"uncorrected": baseline,
                   "weighted": correct_support_layout(baseline, relation, sigma, up),
                   "hard": correct_support_layout(baseline, relation, sigma, up, hard=True)}
        reference = {item["instance_id"]: np.array(item["center_point_3d"])
                     for item in json.loads((root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
        results = {}
        for mode, output in outputs.items():
            predicted = {item["instance_id"]: np.array(item["center_point_3d"])
                         for item in output["objects"]}
            child_error = float(np.linalg.norm(predicted[child] - reference[child]))
            parent_error = float(np.linalg.norm(predicted[parent] - reference[parent]))
            gap = float(np.dot(predicted[child] - predicted[parent], up))
            reference_gap = float(np.dot(reference[child] - reference[parent], up))
            results[mode] = {"child_error_m": child_error, "parent_error_m": parent_error,
                             "pair_mean_error_m": (child_error + parent_error) / 2,
                             "height_gap_error_m": abs(gap - reference_gap)}
        outcomes.append({"scene_id": scene_id,
                         "supported_by_table": support_label,
                         "child_class": by_id[child]["label"], "child_id": child,
                         "parent_id": parent, "methods": results})
    positive = [item for item in outcomes if item["supported_by_table"] is True]
    negative = [item for item in outcomes if item["supported_by_table"] is False]
    uncertain = [item for item in outcomes if item["supported_by_table"] is None]
    if not positive:
        raise ValueError("No positive support pairs to evaluate")
    summary = {}
    for mode in ("uncorrected", "weighted", "hard"):
        summary[mode] = {
            "mean_child_error_m": float(np.mean([x["methods"][mode]["child_error_m"] for x in positive])),
            "median_child_error_m": float(np.median([x["methods"][mode]["child_error_m"] for x in positive])),
            "mean_pair_error_m": float(np.mean([x["methods"][mode]["pair_mean_error_m"] for x in positive])),
            "mean_height_gap_error_m": float(np.mean([x["methods"][mode]["height_gap_error_m"] for x in positive])),
        }
    summary["weighted"]["child_improved_count"] = sum(
        x["methods"]["weighted"]["child_error_m"] < x["methods"]["uncorrected"]["child_error_m"]
        for x in positive)
    summary["weighted"]["child_worsened_count"] = sum(
        x["methods"]["weighted"]["child_error_m"] > x["methods"]["uncorrected"]["child_error_m"]
        for x in positive)
    return {"setting": "oracle 2D boxes, predicted RGB depth, official calibration, manual RGB support labels",
            "candidate_count": len(outcomes), "positive_count": len(positive), "negative_count": len(negative),
            "uncertain_count": len(uncertain),
            "offset_prior_m": protocol["offset_prior_m"], "center_sigma_m": prior,
            "note": "Nearby frames may be correlated; no relation detection or independent test-set claim.",
            "summary_positive_pairs": summary, "scenes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        help="Pre-frozen parameters separate from visual relation labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.labels, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "positive_count",
                    "negative_count", "uncertain_count", "summary_positive_pairs")}, indent=2))


if __name__ == "__main__":
    main()
