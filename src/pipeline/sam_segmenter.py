"""Local SAM ViT-B masks prompted by detector boxes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_sam(checkpoint: Path, device: str = "auto"):
    """Load a local Hugging Face SAM checkpoint without network requests."""
    import torch
    from transformers import SamModel, SamProcessor

    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Local SAM checkpoint missing: {checkpoint}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = SamProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = SamModel.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    return processor, model, device


def segment_detector_boxes(image: Image.Image, objects: list[dict],
                           processor, model) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return one full-resolution mask and SAM IoU estimate per input box.

    Masks are kept in the original image coordinates. Geometry later intersects
    each mask with its detector box, so outside-box SAM pixels do not enter 3D.
    Empty masks remain empty; callers decide whether to skip or fall back.
    """
    import torch

    if not objects:
        return {}, {}
    if image.mode != "RGB":
        image = image.convert("RGB")
    height, width = image.height, image.width
    ids = [str(obj["instance_id"]) for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object instance_id")
    boxes = []
    for obj in objects:
        box = np.asarray(obj["bbox_2d"], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError(f"Invalid detector box: {obj['bbox_2d']}")
        boxes.append(box.tolist())
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    processed = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    masks_array = processed[0]
    if masks_array.shape[0] != len(objects):
        raise ValueError("SAM returned a different number of box masks")
    masks, scores = {}, {}
    for index, instance_id in enumerate(ids):
        mask = np.asarray(masks_array[index, 0], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("SAM mask shape differs from source image")
        masks[instance_id] = mask
        scores[instance_id] = float(outputs.iou_scores[0, index, 0].detach().cpu())
    return masks, scores
