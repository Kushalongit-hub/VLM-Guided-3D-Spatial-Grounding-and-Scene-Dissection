"""Prepare one official SUN RGB-D depth PNG and VoteNet v2 annotation pair."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.sunrgbd import convert_labels, decode_depth_png, read_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--depth-png", type=Path, required=True)
    parser.add_argument("--label-txt", type=Path, required=True)
    parser.add_argument("--calib-txt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    depth = decode_depth_png(args.depth_png)
    rtilt, intrinsics = read_calibration(args.calib_txt)
    objects, reference = convert_labels(args.label_txt, rtilt, depth.shape)
    if not objects:
        raise ValueError("No annotated objects overlap the depth image")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "depth.npy", depth)
    (args.output_dir / "objects.json").write_text(json.dumps(objects, indent=2), encoding="utf-8")
    (args.output_dir / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    manifest = {"scenes": [{"scene_id": args.scene_id, "depth": "depth.npy",
                            "objects": "objects.json", "reference": "reference.json",
                            "intrinsics": intrinsics}]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(objects)} annotated objects in {args.output_dir}")


if __name__ == "__main__":
    main()
