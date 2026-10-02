"""Evaluate predicted RGB depth with oracle SUN RGB-D boxes and calibration."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.benchmark import run_manifest
from src.pipeline.depth_estimator import load_metric_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    model, device = load_metric_model(args.checkpoint, args.device)
    predicted = []
    durations = []
    for scene in source["scenes"]:
        image_path = args.manifest.parent / scene["scene_id"] / "image.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read scene image: {image_path}")
        started = perf_counter()
        depth = model.infer_image(image)
        durations.append(perf_counter() - started)
        depth_path = args.manifest.parent / scene["scene_id"] / "predicted_depth.npy"
        np.save(depth_path, depth.astype(np.float32))
        predicted.append({**scene, "depth": f"{scene['scene_id']}/predicted_depth.npy"})
    predicted_manifest = args.manifest.parent / "manifest_predicted_depth.json"
    predicted_manifest.write_text(json.dumps({"scenes": predicted}, indent=2), encoding="utf-8")
    report = run_manifest(predicted_manifest)
    report.update({"depth_source": "Depth-Anything-V2-Metric-Hypersim-Small",
                   "boxes_source": "official_oracle_2d_annotations",
                   "camera_source": "official_sunrgbd_calibration",
                   "device": device,
                   "mean_depth_inference_seconds": float(np.mean(durations))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scene_count", "reference_count",
          "matched_count", "median_centroid_error_m", "p90_centroid_error_m",
          "mean_depth_inference_seconds")}, indent=2))


if __name__ == "__main__":
    main()
