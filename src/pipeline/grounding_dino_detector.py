"""Open-vocabulary object boxes from a local Grounding DINO checkpoint."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


DEFAULT_LABELS = ("cup", "mug", "desk", "table")


def suppress_prompt_duplicates(detections: list[dict], iou_threshold: float = 0.7) -> list[dict]:
    """Collapse overlapping synonyms while preserving distinct instances."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    from src.pipeline.detection_evaluation import box_iou

    groups = ({"cup", "mug"}, {"desk", "table"})
    kept = []
    for candidate in sorted(detections, key=lambda row: row["confidence"], reverse=True):
        synonym_group = next((group for group in groups if candidate["label"] in group),
                             {candidate["label"]})
        if any(prior["label"] in synonym_group and
               box_iou(candidate["bbox_2d"], prior["bbox_2d"]) >= iou_threshold
               for prior in kept):
            continue
        kept.append(candidate)
    return kept


def load_grounding_dino(checkpoint: Path, device: str = "auto"):
    """Load local weights and processor; no network access is needed."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Grounding DINO checkpoint directory missing: {checkpoint}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        checkpoint, local_files_only=True).to(device).eval()
    return processor, model, device


def _canonical_label(raw: str, labels: tuple[str, ...]) -> str | None:
    words = raw.lower().strip(" .").split()
    for label in labels:
        if label in words:
            return label
    return None


def detect_grounded_objects(image: Image.Image, processor, model,
                            labels: tuple[str, ...] = DEFAULT_LABELS,
                            box_threshold: float = 0.25,
                            text_threshold: float = 0.25,
                            prompt_format: str = "separate") -> list[dict]:
    """Detect prompted categories and return clipped pixel-space boxes."""
    import torch

    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be nonempty and unique")
    if not 0 < box_threshold <= 1 or not 0 < text_threshold <= 1:
        raise ValueError("thresholds must be in (0, 1]")
    if prompt_format not in {"separate", "combined"}:
        raise ValueError("prompt_format must be separate or combined")
    if image.mode != "RGB":
        image = image.convert("RGB")
    device = next(model.parameters()).device
    phrases = [f"a {label.replace('_', ' ')}" for label in labels]
    text_input = [phrases] if prompt_format == "separate" else ". ".join(phrases) + "."
    inputs = processor(images=image, text=text_input, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        output, inputs.input_ids, threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
    detections = []
    for idx, (box, score, raw_label) in enumerate(zip(
            result["boxes"], result["scores"], result["text_labels"])):
        label = _canonical_label(str(raw_label), labels)
        if label is None:
            continue
        xyxy = box.tolist()
        xmin = max(0, min(image.width - 1, int(np.floor(xyxy[0]))))
        ymin = max(0, min(image.height - 1, int(np.floor(xyxy[1]))))
        xmax = max(xmin + 1, min(image.width, int(np.ceil(xyxy[2]))))
        ymax = max(ymin + 1, min(image.height, int(np.ceil(xyxy[3]))))
        detections.append({"instance_id": f"gdino_{idx}", "label": label,
                           "raw_label": str(raw_label), "confidence": float(score),
                           "bbox_2d": [xmin, ymin, xmax, ymax]})
    return suppress_prompt_duplicates(detections)
