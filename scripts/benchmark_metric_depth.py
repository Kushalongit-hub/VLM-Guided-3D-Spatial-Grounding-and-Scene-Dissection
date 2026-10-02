"""Benchmark the indoor metric checkpoint on downloaded SUN RGB-D scenes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.depth_estimator import load_metric_model
from src.pipeline.depth_evaluation import depth_metrics
from src.pipeline.sunrgbd import decode_depth_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("data/sunrgbd_sample"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()
    manifest = json.loads((args.sample_dir / "source_manifest.json").read_text(encoding="utf-8"))
    model, device = load_metric_model(args.checkpoint, args.device)
    if device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
    scenes = []
    for entry in manifest["records"]:
        for key in ("image", "depth"):
            source = args.sample_dir / entry[key]["file"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry[key]["sha256"]:
                raise ValueError(f"Sample checksum mismatch: {source}")
        image = cv2.imread(str(args.sample_dir / entry["image"]["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image for row {entry['row_idx']}")
        reference = decode_depth_png(args.sample_dir / entry["depth"]["file"])
        started = perf_counter()
        predicted = model.infer_image(image, input_size=args.input_size)
        elapsed = perf_counter() - started
        metrics = depth_metrics(predicted, reference)
        metrics["row_idx"] = entry["row_idx"]
        metrics["inference_seconds"] = elapsed
        scenes.append(metrics)
        print(f"row {entry['row_idx']}: abs_rel={metrics['abs_rel']:.3f}, "
              f"mae={metrics['mae_m']:.3f} m, delta_1={metrics['delta_1']:.3f}")
    keys = ("mae_m", "median_ae_m", "rmse_m", "abs_rel", "delta_1", "coverage",
            "inference_seconds")
    summary = {"scene_count": len(scenes), "total_valid_pixels": sum(s["valid_pixels"] for s in scenes)}
    for key in keys:
        summary[f"mean_scene_{key}"] = float(np.mean([s[key] for s in scenes]))
    if device == "cuda":
        summary["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        summary["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
    report = {"dataset": manifest["dataset"], "dataset_revision": manifest["revision"],
              "split": manifest["split"], "model": "Depth-Anything-V2-Metric-Hypersim-Small",
              "device": device, "input_size": args.input_size,
              "valid_depth_range_m": [0.1, 7.99], "summary": summary, "scenes": scenes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
