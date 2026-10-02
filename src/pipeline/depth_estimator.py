"""Local Depth Anything V2 Small indoor metric checkpoint inference."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


SMALL_CONFIG = {"encoder": "vits", "features": 64,
                "out_channels": [48, 96, 192, 384], "max_depth": 20.0}


def load_metric_model(checkpoint: Path, device: str = "auto"):
    """Load the official Hypersim Small metric checkpoint from a local file."""
    import torch

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Metric checkpoint missing: {checkpoint}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this PyTorch installation")
    vendor_root = Path(__file__).resolve().parents[1] / "depth_anything" / "metric_depth"
    sys.path.insert(0, str(vendor_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.pop(0)
    model = DepthAnythingV2(**SMALL_CONFIG)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), device


def infer_metric_depth(image_path: Path, checkpoint: Path, device: str = "auto",
                       input_size: int = 518) -> tuple[np.ndarray, str]:
    """Return an HxW depth array in metres for the original image dimensions."""
    if input_size < 14:
        raise ValueError("input_size must be at least 14")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read RGB image: {image_path}")
    model, selected_device = load_metric_model(checkpoint, device)
    depth = model.infer_image(image, input_size=input_size)
    if depth.shape != image.shape[:2] or not np.isfinite(depth).all():
        raise ValueError("Model returned invalid depth")
    return depth.astype(np.float32), selected_device
