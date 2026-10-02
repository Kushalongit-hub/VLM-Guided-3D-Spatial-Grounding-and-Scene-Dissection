"""Frozen Grounding DINO Tiny detector comparison on the SUN RGB-D desk set."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.grounding_dino_detector import detect_grounded_objects, load_grounding_dino


PROMPT = ["a cup", "a mug", "a desk", "a table"]
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--prompt-format", choices=("separate", "combined"), default="separate")
    args = parser.parse_args()
    if not 0 < args.box_threshold <= 1 or not 0 < args.text_threshold <= 1:
        parser.error("thresholds must be in (0, 1]")
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]
    labels = {row["scene_id"]: row for row in json.loads(args.labels.read_text(encoding="utf-8"))["scenes"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = perf_counter()
    processor, model, device = load_grounding_dino(args.checkpoint, device)
    load_seconds = perf_counter() - start
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for scene in manifest:
        scene_id = scene["scene_id"]
        image = Image.open(args.root / scene_id / "image.jpg").convert("RGB")
        start = perf_counter()
        detections = detect_grounded_objects(image, processor, model,
                                             box_threshold=args.box_threshold,
                                             text_threshold=args.text_threshold,
                                             prompt_format=args.prompt_format)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = perf_counter() - start
        annotation = labels[scene_id]
        relation_label = annotation.get("supported_by_desk", annotation.get("supported_by_table"))
        oracle = {row["instance_id"]: row for row in json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))}
        def best_match(instance_id: str, allowed: set[str]) -> dict:
            ranked = sorted(({"instance_id": det["instance_id"], "label": det["label"],
                              "iou": box_iou(oracle[instance_id]["bbox_2d"], det["bbox_2d"])}
                             for det in detections if det["label"] in allowed),
                            key=lambda row: row["iou"], reverse=True)
            return ranked[0] if ranked else {"instance_id": None, "label": None, "iou": 0.0}
        child = best_match(annotation["child_id"], {"cup", "mug"})
        parent = best_match(annotation["parent_id"], {"desk", "table"})
        rows.append({"scene_id": scene_id, "detections": detections, "seconds": seconds,
                     "label": relation_label, "child_match": child,
                     "parent_match": parent,
                     "pair_detected": child["iou"] >= IOU_THRESHOLD and parent["iou"] >= IOU_THRESHOLD})
        print(f"{scene_id}: {len(detections)} detections, child IoU={child['iou']:.3f}, parent IoU={parent['iou']:.3f}, {seconds:.2f}s", flush=True)
    positive = [row for row in rows if row["label"] is True]
    report = {"setting": f"Grounding DINO Tiny, official checkpoint, cup/mug/desk/table prompt; SUN RGB-D {args.root.name} set",
              "checkpoint": str(args.checkpoint), "prompt": PROMPT,
              "prompt_format": args.prompt_format,
              "box_threshold": args.box_threshold, "text_threshold": args.text_threshold,
              "iou_threshold": IOU_THRESHOLD, "device": device, "model_load_seconds": load_seconds,
              "median_inference_seconds_excluding_first": float(np.median([r["seconds"] for r in rows[1:]])),
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1048576 if device == "cuda" else None,
              "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1048576 if device == "cuda" else None,
              "positive_count": len(positive), "positive_pair_detected_count": sum(r["pair_detected"] for r in positive),
              "positive_pair_detection_coverage": sum(r["pair_detected"] for r in positive) / len(positive),
              "note": "Small selected positive set; no negative relation labels, precision/AP claim, or tuned threshold.",
              "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("positive_count", "positive_pair_detected_count", "positive_pair_detection_coverage", "median_inference_seconds_excluding_first", "peak_cuda_allocated_mib")}, indent=2))


if __name__ == "__main__":
    main()
