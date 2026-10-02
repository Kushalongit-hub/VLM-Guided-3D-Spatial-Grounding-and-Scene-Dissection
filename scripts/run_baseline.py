"""Run the metric-depth geometry baseline on a saved depth map and boxes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True, help="HxW metric depth .npy file")
    parser.add_argument("--objects", type=Path, required=True, help="JSON list of label/bbox_2d objects")
    parser.add_argument("--output", type=Path, required=True)
    for key in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{key}", type=float, required=True)
    args = parser.parse_args()
    depth = np.load(args.depth, allow_pickle=False)
    objects = json.loads(args.objects.read_text(encoding="utf-8"))
    if not isinstance(objects, list):
        raise ValueError("Objects JSON must be a list")
    camera = Intrinsics(args.fx, args.fy, args.cx, args.cy)
    layout = build_layout(depth, objects, camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    print(f"Wrote {len(layout['objects'])} objects to {args.output}")


if __name__ == "__main__":
    main()
