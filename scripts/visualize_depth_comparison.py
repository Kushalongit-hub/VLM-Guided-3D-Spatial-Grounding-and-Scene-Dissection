"""Render RGB, sensor depth, predicted depth, and absolute error side by side."""

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.depth_estimator import infer_metric_depth
from src.pipeline.sunrgbd import decode_depth_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-depth", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {args.image}")
    reference = decode_depth_png(args.reference_depth)
    prediction, _ = infer_metric_depth(args.image, args.checkpoint, args.device)
    valid = (reference >= 0.1) & (reference <= 7.99)
    error = np.abs(prediction - reference)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB input")
    depth_cmap = plt.get_cmap("turbo").copy()
    depth_cmap.set_bad("#333333")
    for axis, data, title in ((axes[1], reference, "Sensor depth (m)"),
                              (axes[2], prediction, "Predicted depth (m)")):
        artist = axis.imshow(np.ma.masked_where(~valid, data), cmap=depth_cmap, vmin=0, vmax=8)
        axis.set_title(title)
    fig.colorbar(artist, ax=[axes[1], axes[2]], shrink=0.65, label="metres")
    artist = axes[3].imshow(np.ma.masked_where(~valid, error), cmap="magma", vmin=0, vmax=2)
    axes[3].set_title("Absolute error (m)")
    fig.colorbar(artist, ax=axes[3], shrink=0.65, label="metres")
    for axis in axes:
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
