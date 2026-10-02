"""Exploratory, manually labeled mug-on-table support case from SUN RGB-D.

The relation and 0.40 m offset are image/category priors, not fitted to 3D labels.
This single case is a plumbing check, not evidence of general improvement.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scene.name != "sunrgbd_002591":
        raise ValueError("The manually checked support annotation applies only to sunrgbd_002591")
    rtilt, intrinsics = read_calibration(args.scene / "calib.txt")
    objects = json.loads((args.scene / "objects.json").read_text(encoding="utf-8"))
    labels = {obj["instance_id"]: obj["label"] for obj in objects}
    if labels.get("object_1") != "table" or labels.get("object_2") != "mug":
        raise ValueError("Unexpected object IDs or labels; recheck the support annotation")
    layout = build_layout(np.load(args.scene / "predicted_depth.npy"), objects,
                          Intrinsics(**intrinsics))
    relation = {"child_id": "object_2", "parent_id": "object_1",
                "offset_m": 0.40, "sigma_m": 0.20}
    sigmas = {"object_0": 0.20, "object_1": 0.20, "object_2": 0.40}
    up = camera_up_from_rtilt(rtilt)
    methods = {
        "uncorrected": layout,
        "weighted": correct_support_layout(layout, [relation], sigmas, up),
        "hard": correct_support_layout(layout, [relation], sigmas, up, hard=True),
    }
    reference = {obj["instance_id"]: np.array(obj["center_point_3d"])
                 for obj in json.loads((args.scene / "reference.json").read_text(encoding="utf-8"))["objects"]}
    report = {
        "scene_id": args.scene.name,
        "setting": "oracle 2D boxes; predicted RGB depth; official calibration; manually identified support",
        "assumption": "0.40 m mug-center-above-table-center offset; prior chosen without reading 3D labels",
        "center_sigma_m": sigmas,
        "relation": relation,
        "methods": {},
    }
    for name, output in methods.items():
        report["methods"][name] = {
            obj["instance_id"]: {
                "center_point_3d": obj["center_point_3d"],
                "centroid_error_m": float(np.linalg.norm(np.array(obj["center_point_3d"]) - reference[obj["instance_id"]]))
            } for obj in output["objects"] if obj["instance_id"] in ("object_1", "object_2")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: round(item["centroid_error_m"], 3)
                             for key, item in values.items()}
                      for name, values in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
