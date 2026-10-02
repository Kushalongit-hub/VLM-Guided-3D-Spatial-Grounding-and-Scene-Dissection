"""COCO object detection baseline with a local Ultralytics YOLO checkpoint."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def suppress_duplicate_masks(objects: list[dict], masks: dict[str, np.ndarray],
                             iou_threshold: float = 0.7) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Keep the highest-scoring instance when same-class masks largely overlap."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    kept, kept_masks = [], {}
    for obj in sorted(objects, key=lambda item: item["confidence"], reverse=True):
        instance_id = obj["instance_id"]
        candidate = masks[instance_id]
        duplicate = False
        for prior in kept:
            if prior["label"] != obj["label"]:
                continue
            previous = kept_masks[prior["instance_id"]]
            intersection = np.count_nonzero(candidate & previous)
            union = np.count_nonzero(candidate | previous)
            if union and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
            kept_masks[instance_id] = candidate
    return kept, kept_masks


def detect_objects(image_path: Path, weights: Path, confidence: float = 0.25,
                   device: str = "cpu") -> list[dict]:
    """Return pixel boxes and scores, sorted by detector confidence."""
    if not weights.is_file():
        raise FileNotFoundError(f"Detector checkpoint missing: {weights}")
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    config_dir = Path(__file__).resolve().parents[2] / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = YOLO(str(weights)).predict(source=str(image_path), conf=confidence,
                                        device=device, verbose=False)[0]
    height, width = result.orig_shape
    found = []
    if result.boxes is None:
        return found
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for index in np.argsort(-scores):
        xmin, ymin, xmax, ymax = xyxy[index]
        box = [max(0, int(np.floor(xmin))), max(0, int(np.floor(ymin))),
               min(width, int(np.ceil(xmax))), min(height, int(np.ceil(ymax)))]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        found.append({"label": str(result.names[int(classes[index])]),
                      "instance_id": f"det_{index}", "bbox_2d": box,
                      "confidence": float(scores[index])})
    return found


def segment_objects(image_path: Path, weights: Path, confidence: float = 0.25,
                   device: str = "cpu") -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return COCO instances and full-resolution binary masks."""
    if not weights.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint missing: {weights}")
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    config_dir = Path(__file__).resolve().parents[2] / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = YOLO(str(weights)).predict(source=str(image_path), conf=confidence,
                                        device=device, retina_masks=True, verbose=False)[0]
    return parse_segment_result(result)


def parse_segment_result(result) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Convert one Ultralytics result to our instance and mask schema."""
    if result.boxes is None or result.masks is None:
        return [], {}
    height, width = result.orig_shape
    all_masks = result.masks.data.cpu().numpy().astype(bool)
    if all_masks.shape[1:] != (height, width):
        raise ValueError("Segmentation masks do not match original image size")
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    objects, masks = [], {}
    for index in np.argsort(-scores):
        xmin, ymin, xmax, ymax = xyxy[index]
        box = [max(0, int(np.floor(xmin))), max(0, int(np.floor(ymin))),
               min(width, int(np.ceil(xmax))), min(height, int(np.ceil(ymax)))]
        if box[0] >= box[2] or box[1] >= box[3] or not all_masks[index].any():
            continue
        instance_id = f"seg_{index}"
        objects.append({"label": str(result.names[int(classes[index])]),
                        "instance_id": instance_id, "bbox_2d": box,
                        "confidence": float(scores[index])})
        masks[instance_id] = all_masks[index]
    return suppress_duplicate_masks(objects, masks)
