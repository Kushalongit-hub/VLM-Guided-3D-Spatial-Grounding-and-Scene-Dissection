"""Exploratory DINO-box versus DINO+SAM-mask 3D center comparison.

Use the same predicted depth and detected boxes in both arms. Oracle boxes
align predictions to references only for scoring. Previously viewed scenes.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.geometry import Intrinsics, lift_region
from src.pipeline.grounding_dino_detector import load_grounding_dino, detect_grounded_objects
from src.pipeline.sam_segmenter import load_sam, segment_detector_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grounding-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path,
                        help="Optional compressed SAM masks per scene for follow-up analysis")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    scenes = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]
    start = perf_counter()
    processor, detector, device = load_grounding_dino(args.grounding_checkpoint, args.device)
    predictions = {}
    for scene in scenes:
        scene_id = scene["scene_id"]
        with Image.open(args.root / scene_id / "image.jpg") as source:
            predictions[scene_id] = detect_grounded_objects(
                source.convert("RGB"), processor, detector, labels=("lamp", "table"),
                box_threshold=0.25, text_threshold=0.25)
        print(f"{scene_id}: {len(predictions[scene_id])} DINO boxes", flush=True)
    del detector
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    sam_processor, sam, _ = load_sam(args.sam_checkpoint, device)
    rows = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_dir = args.root / scene_id
        with Image.open(scene_dir / "image.jpg") as source:
            masks, sam_scores = segment_detector_boxes(
                source.convert("RGB"), predictions[scene_id], sam_processor, sam)
        if args.masks_dir:
            args.masks_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.masks_dir / f"{scene_id}.npz", **masks)
        objects = json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))
        references = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                      for obj in json.loads((args.root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
        depth = np.load(scene_dir / "predicted_depth.npy", allow_pickle=False)
        intrinsics = Intrinsics(**scene["intrinsics"])
        matched = []
        for label in ("lamp", "table"):
            target = next((obj for obj in objects if obj["label"] == label), None)
            if target is None:
                continue
            candidates = [(box_iou(target["bbox_2d"], obj["bbox_2d"]), obj)
                          for obj in predictions[scene_id] if obj["label"] == label]
            if not candidates:
                continue
            iou, detected = max(candidates, key=lambda item: item[0])
            if iou < 0.5:
                continue
            box = tuple(detected["bbox_2d"])
            mask = masks[detected["instance_id"]]
            xmin, ymin, xmax, ymax = box
            if not mask[ymin:ymax, xmin:xmax].any():
                matched.append({"label": label, "reference_id": target["instance_id"],
                                "prediction_id": detected["instance_id"], "box_iou": iou,
                                "sam_iou_estimate": sam_scores[detected["instance_id"]],
                                "sam_mask_empty_in_box": True})
                continue
            box_center = np.asarray(lift_region(depth, box, intrinsics)["center_point_3d"])
            mask_center = np.asarray(lift_region(depth, box, intrinsics, mask=mask)["center_point_3d"])
            true_center = references[target["instance_id"]]
            box_error = float(np.linalg.norm(box_center - true_center))
            mask_error = float(np.linalg.norm(mask_center - true_center))
            matched.append({"label": label, "reference_id": target["instance_id"],
                            "prediction_id": detected["instance_id"], "box_iou": iou,
                            "sam_iou_estimate": sam_scores[detected["instance_id"]],
                            "sam_mask_empty_in_box": False,
                            "box_error_m": box_error, "mask_error_m": mask_error,
                            "mask_minus_box_error_m": mask_error - box_error,
                            "mask_pixels_in_box": int(mask[ymin:ymax, xmin:xmax].sum())})
        rows.append({"scene_id": scene_id, "detections": predictions[scene_id], "matches": matched})
        print(f"{scene_id}: {len(matched)} matched lamp/table regions", flush=True)
    valid = [match for scene in rows for match in scene["matches"] if not match["sam_mask_empty_in_box"]]
    differences = np.asarray([row["mask_minus_box_error_m"] for row in valid])
    by_scene = defaultdict(list)
    for scene in rows:
        by_scene[scene["scene_id"]] = [match for match in scene["matches"]
                                      if not match["sam_mask_empty_in_box"]]
    occupied = [group for group in by_scene.values() if group]
    rng = np.random.default_rng(20261002)
    draws = []
    if occupied:
        for _ in range(10000):
            sampled = [row["mask_minus_box_error_m"]
                       for idx in rng.integers(len(occupied), size=len(occupied))
                       for row in occupied[idx]]
            draws.append(float(np.mean(sampled)))
    report = {
        "setting": "exploratory Grounding DINO lamp/table + SAM ViT-B on already inspected SUN RGB-D scenes",
        "box_threshold": 0.25, "text_threshold": 0.25, "matching_box_iou_min": 0.5,
        "scene_count": len(rows), "matched_count": sum(len(scene["matches"]) for scene in rows),
        "scored_count": len(valid), "empty_mask_count": sum(match["sam_mask_empty_in_box"]
                                                         for scene in rows for match in scene["matches"]),
        "mean_box_error_m": float(np.mean([row["box_error_m"] for row in valid])) if valid else None,
        "mean_mask_error_m": float(np.mean([row["mask_error_m"] for row in valid])) if valid else None,
        "mean_mask_minus_box_error_m": float(differences.mean()) if valid else None,
        "mask_better_count": int(np.sum(differences < 0)),
        "scene_bootstrap_95pct_interval_m": np.percentile(draws, [2.5, 97.5]).tolist() if draws else None,
        "per_label": {label: {"count": sum(row["label"] == label for row in valid),
                              "mean_mask_minus_box_error_m": float(np.mean([
                                  row["mask_minus_box_error_m"] for row in valid if row["label"] == label]))
                              if any(row["label"] == label for row in valid) else None}
                      for label in ("lamp", "table")},
        "wall_seconds": perf_counter() - start,
        "note": "Same DINO boxes and RGB depth in both arms. Oracle boxes match for scoring only. Previously scored scenes; no generalization claim. Unmatched regions omitted from error means.",
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
