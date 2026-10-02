"""Run Depth Anything V2 Small indoor metric inference on one RGB image."""

import argparse
from pathlib import Path
import sys

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.depth_estimator import infer_metric_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official Depth Anything V2 Metric Hypersim Small .pth file")
    parser.add_argument("--output", type=Path, required=True, help="Output metric depth .npy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--preview", type=Path, help="Optional 0–20 m colorized PNG preview")
    args = parser.parse_args()
    depth, device = infer_metric_depth(args.image, args.checkpoint, args.device, args.input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, depth)
    if args.preview:
        scaled = np.clip(depth / 20.0 * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), color):
            raise OSError(f"Could not write preview: {args.preview}")
    print(f"Saved {depth.shape[1]}x{depth.shape[0]} metric depth to {args.output} using {device}")


if __name__ == "__main__":
    main()
