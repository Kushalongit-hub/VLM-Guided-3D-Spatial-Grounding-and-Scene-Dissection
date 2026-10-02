"""Post-hoc interaction of SAM masks and child-only support correction."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dino-sam-report", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {row["scene_id"]: row for row in json.loads(
        args.labels.read_text(encoding="utf-8"))["scenes"]}
    dino = {row["scene_id"]: row for row in json.loads(
        args.dino_sam_report.read_text(encoding="utf-8"))["scenes"]}
    manifest = {row["scene_id"]: row for row in json.loads(
        (args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    rows = []
    for scene_id, annotation in labels.items():
        if annotation["supported_by_table"] is not True:
            continue
        scene = manifest[scene_id]
        matches = {row["label"]: row for row in dino[scene_id]["matches"]
                   if not row["sam_mask_empty_in_box"]}
        if not all(label in matches for label in ("lamp", "table")):
            rows.append({"scene_id": scene_id, "paired_detection": False})
            continue
        child_id = matches["lamp"]["prediction_id"]
        parent_id = matches["table"]["prediction_id"]
        ids = {child_id, parent_id}
        objects = [row for row in dino[scene_id]["detections"] if row["instance_id"] in ids]
        if len(objects) != 2:
            raise ValueError(f"Expected exactly two detected objects in {scene_id}")
        depth = np.load(args.root / scene_id / "predicted_depth.npy", allow_pickle=False)
        with np.load(args.masks_dir / f"{scene_id}.npz", allow_pickle=False) as stored:
            masks = {key: stored[key] for key in ids}
        camera = Intrinsics(**scene["intrinsics"])
        layouts = {"box": build_layout(depth, objects, camera),
                   "mask": build_layout(depth, objects, camera, masks=masks)}
        rtilt, _ = read_calibration(args.root / scene_id / "calib.txt")
        up = camera_up_from_rtilt(rtilt)
        truth = {row["instance_id"]: np.asarray(row["center_point_3d"])
                 for row in json.loads((args.root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
        child_truth = truth[annotation["child_id"]]
        parent_truth = truth[annotation["parent_id"]]
        results = {}
        for region_type, layout in layouts.items():
            corrected = correct_child_ray_support_layout(
                layout, child_id, parent_id, up, offset_m=0.4,
                child_sigma_m=0.4, relation_sigma_m=0.2, max_shift_m=0.5)
            for method, result in (("uncorrected", layout), ("child_ray_soft", corrected)):
                centers = {row["instance_id"]: np.asarray(row["center_point_3d"])
                           for row in result["objects"]}
                results[f"{region_type}_{method}"] = {
                    "child_error_m": float(np.linalg.norm(centers[child_id] - child_truth)),
                    "parent_error_m": float(np.linalg.norm(centers[parent_id] - parent_truth)),
                    "height_gap_error_m": float(abs((centers[child_id] - centers[parent_id]) @ up -
                                                    (child_truth - parent_truth) @ up)),
                }
        rows.append({"scene_id": scene_id, "paired_detection": True, "methods": results})
    matched = [row for row in rows if row["paired_detection"]]
    means = {name: {metric: float(np.mean([row["methods"][name][metric] for row in matched]))
                    for metric in ("child_error_m", "parent_error_m", "height_gap_error_m")}
             for name in matched[0]["methods"]} if matched else {}
    report = {"setting": "post-hoc mask and child-ray interaction on already viewed lamp/table positives",
              "visually_positive_count": len(rows), "paired_detector_coverage": len(matched),
              "methods": means, "scenes": rows,
              "note": "Same detected DINO boxes and predicted depth. Only positives with both objects matched at IoU>=0.5 are scored. Oracle boxes align predictions for scoring only. No fresh-holdout claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
