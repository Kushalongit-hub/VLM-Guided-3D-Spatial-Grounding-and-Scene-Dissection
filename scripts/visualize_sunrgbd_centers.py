"""Inspect annotated boxes and projected reference/predicted 3D centers."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["scenes"] if item["scene_id"] == args.scene_dir.name)
    camera = Intrinsics(**record["intrinsics"])
    objects = json.loads((args.scene_dir / "objects.json").read_text(encoding="utf-8"))
    reference = json.loads((args.scene_dir / "reference.json").read_text(encoding="utf-8"))
    depth = np.load(args.scene_dir / "predicted_depth.npy", allow_pickle=False)
    predicted = build_layout(depth, objects, camera)
    rgb = cv2.imread(str(args.scene_dir / "image.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("Scene image missing")
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    for obj, pred, truth in zip(objects, predicted["objects"], reference["objects"]):
        x0, y0, x1, y1 = obj["bbox_2d"]
        axis.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                 edgecolor="yellow", linewidth=1.5))
        error = np.linalg.norm(np.asarray(pred["center_point_3d"]) -
                               np.asarray(truth["center_point_3d"]))
        axis.text(x0, max(10, y0-3), f"{obj['label']} {error:.2f}m", color="yellow",
                  fontsize=9, weight="bold")
        for center, marker, color in ((truth["center_point_3d"], "o", "red"),
                                      (pred["center_point_3d"], "x", "cyan")):
            x, y, z = center
            axis.scatter(camera.fx*x/z+camera.cx, camera.fy*y/z+camera.cy,
                         marker=marker, c=color, s=55, linewidths=2)
    axis.set_title("Yellow: oracle 2D boxes; red: reference 3D center; cyan: predicted 3D center")
    axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
