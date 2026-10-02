"""Benchmark RGB depth + YOLO segmentation + 3D lifting on a frozen manifest.

Paired SUN RGB-D labels omit many objects. Therefore this reports annotation
coverage and matched-center errors, not detector precision, recall, or AP.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.depth_estimator import load_metric_model
from src.pipeline.detection_evaluation import SUN_TO_COCO, match_boxes
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.object_detector import parse_segment_result


def errors_for_matches(matches: list[dict], layout: dict, reference: dict) -> dict:
    predicted = {item["instance_id"]: np.asarray(item["center_point_3d"], dtype=float)
                 for item in layout["objects"]}
    truth = {item["instance_id"]: np.asarray(item["center_point_3d"], dtype=float)
             for item in reference["objects"]}
    return {pair["reference_id"]: float(np.linalg.norm(
        predicted[pair["prediction_id"]] - truth[pair["reference_id"]])) for pair in matches}


def summarize_errors(scenes: list[dict], match_key: str, method: str) -> dict:
    values = [error for scene in scenes for error in
              scene[match_key]["center_errors_m"][method].values()]
    return {"count": len(values), "median_m": float(np.median(values)) if values else None,
            "mean_m": float(np.mean(values)) if values else None,
            "p90_m": float(np.percentile(values, 90)) if values else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--segmenter-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")
    if not args.segmenter_weights.is_file():
        raise FileNotFoundError(args.segmenter_weights)
    config_dir = Path(__file__).resolve().parents[1] / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    import torch
    from ultralytics import YOLO

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("scenes"):
        raise ValueError("Manifest has no scenes")
    started_load = perf_counter()
    detector = YOLO(str(args.segmenter_weights))
    depth_model, device = load_metric_model(args.depth_checkpoint, device)
    model_load_seconds = perf_counter() - started_load
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    results = []
    for scene in manifest["scenes"]:
        scene_id = str(scene["scene_id"])
        root = args.manifest.parent
        image_path = root / scene_id / "image.jpg"
        reference_objects = json.loads((root / scene["objects"]).read_text(encoding="utf-8"))
        reference = json.loads((root / scene["reference"]).read_text(encoding="utf-8"))
        scene_started = perf_counter()
        clock = perf_counter()
        detected = detector.predict(source=str(image_path), conf=args.confidence,
                                    device=device, retina_masks=True, verbose=False)[0]
        objects, masks = parse_segment_result(detected)
        if device == "cuda":
            torch.cuda.synchronize()
        detection_seconds = perf_counter() - clock
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {image_path}")
        clock = perf_counter()
        depth = depth_model.infer_image(image, input_size=518)
        if device == "cuda":
            torch.cuda.synchronize()
        depth_seconds = perf_counter() - clock
        if depth.shape != image.shape[:2] or not np.isfinite(depth).all():
            raise ValueError(f"Invalid depth in {scene_id}")
        clock = perf_counter()
        camera = Intrinsics(**scene["intrinsics"])
        oracle_layout = build_layout(depth, reference_objects, camera)
        box_layout = build_layout(depth, objects, camera)
        mask_layout = build_layout(depth, objects, camera, masks=masks)
        agnostic = match_boxes(reference_objects, objects, class_aware=False)
        strict = match_boxes(reference_objects, objects, class_aware=True)
        truth_centers = {item["instance_id"]: np.asarray(item["center_point_3d"])
                         for item in reference["objects"]}
        oracle_centers = {item["instance_id"]: np.asarray(item["center_point_3d"])
                          for item in oracle_layout["objects"]}
        for score in (agnostic, strict):
            ids = [pair["reference_id"] for pair in score["matches"]]
            score["center_errors_m"] = {
                "oracle_boxes": {key: float(np.linalg.norm(oracle_centers[key] - truth_centers[key]))
                                 for key in ids},
                "predicted_boxes": errors_for_matches(score["matches"], box_layout, reference),
                "predicted_masks": errors_for_matches(score["matches"], mask_layout, reference),
            }
        geometry_seconds = perf_counter() - clock
        wall_seconds = perf_counter() - scene_started
        results.append({"scene_id": scene_id, "reference_count": len(reference_objects),
                        "detection_count": len(objects), "detections": objects,
                        "class_agnostic": agnostic, "class_aware": strict,
                        "seconds": {"detection": detection_seconds, "depth": depth_seconds,
                                    "geometry_and_matching": geometry_seconds,
                                    "sum": detection_seconds + depth_seconds + geometry_seconds,
                                    "wall": wall_seconds}})
        print(f"{scene_id}: {len(objects)} detections, "
              f"{strict['matched_count']}/{strict['eligible_count']} class-aware matches", flush=True)
    total_reference = sum(item["reference_count"] for item in results)
    total_agnostic = sum(item["class_agnostic"]["matched_count"] for item in results)
    eligible_strict = sum(item["class_aware"]["eligible_count"] for item in results)
    matched_strict = sum(item["class_aware"]["matched_count"] for item in results)
    summary = {
        "scene_count": len(results), "reference_count": total_reference,
        "class_agnostic_iou50_coverage": total_agnostic / total_reference,
        "class_agnostic_matched_count": total_agnostic,
        "class_aware_eligible_count": eligible_strict,
        "class_aware_matched_count": matched_strict,
        "class_aware_iou50_coverage": matched_strict / eligible_strict if eligible_strict else None,
        "class_aware_all_reference_coverage": matched_strict / total_reference,
        "class_aware_center_error": {mode: summarize_errors(results, "class_aware", mode)
                                     for mode in ("oracle_boxes", "predicted_boxes", "predicted_masks")},
        "class_agnostic_center_error": {mode: summarize_errors(results, "class_agnostic", mode)
                                        for mode in ("oracle_boxes", "predicted_boxes", "predicted_masks")},
        "model_load_seconds": model_load_seconds,
        "mean_detection_seconds": float(np.mean([x["seconds"]["detection"] for x in results])),
        "mean_depth_seconds": float(np.mean([x["seconds"]["depth"] for x in results])),
        "mean_geometry_and_matching_seconds": float(np.mean([x["seconds"]["geometry_and_matching"] for x in results])),
        "mean_sum_seconds": float(np.mean([x["seconds"]["sum"] for x in results])),
        "median_sum_seconds": float(np.median([x["seconds"]["sum"] for x in results])),
        "mean_sum_seconds_after_first": float(np.mean([x["seconds"]["sum"] for x in results[1:]]))
        if len(results) > 1 else None,
        "mean_wall_seconds_after_first": float(np.mean([x["seconds"]["wall"] for x in results[1:]]))
        if len(results) > 1 else None,
    }
    if device == "cuda":
        summary.update({"peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
    report = {"setting": "RGB to YOLO11n-seg to metric depth to 3D; official oracle calibration",
              "matching": "one-to-one 2D box IoU>=0.5; labels incomplete; no AP/precision claim",
              "class_mapping": SUN_TO_COCO, "confidence_threshold": args.confidence,
              "depth_checkpoint": args.depth_checkpoint.name,
              "segmenter_checkpoint": args.segmenter_weights.name,
              "device": device, "summary": summary, "scenes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
