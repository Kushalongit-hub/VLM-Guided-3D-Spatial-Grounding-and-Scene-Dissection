"""Post-hoc audit of annotated center-height gaps; never used for inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt


ROOT = Path(__file__).resolve().parents[1]
SETS = (("laptop", "laptop_table6"), ("bottle", "bottle_table6"))
OUTPUT = ROOT / "results/sunrgbd_laptop_bottle_true_gap_diagnostic.json"


def main() -> None:
    groups = []
    for child_class, name in SETS:
        data = ROOT / "data/sunrgbd_official" / name
        labels = json.loads((ROOT / "results" / f"sunrgbd_{name}_rgb_labels.json").read_text())
        rows = []
        for annotation in labels["scenes"]:
            if annotation["supported_by_paired_table"] is not True:
                continue
            folder = data / annotation["scene_id"]
            objects = json.loads((folder / "objects.json").read_text())
            centers = {obj["instance_id"]: np.asarray(obj["center_point_3d"], dtype=float)
                       for obj in json.loads((folder / "reference.json").read_text())["objects"]}
            child = next(obj for obj in objects if obj["label"] == child_class)
            parent = next(obj for obj in objects if obj["label"] == "table")
            rtilt, _ = read_calibration(folder / "calib.txt")
            up = camera_up_from_rtilt(rtilt)
            gap = float((centers[child["instance_id"]] - centers[parent["instance_id"]]) @ up)
            rows.append({"scene_id": annotation["scene_id"], "true_center_gap_m": gap,
                         "fixed_offset_error_m": abs(gap - 0.4)})
        groups.append({"child_class": child_class, "positive_count": len(rows),
                       "mean_true_center_gap_m": float(np.mean([r["true_center_gap_m"] for r in rows])),
                       "mean_fixed_offset_error_m": float(np.mean([r["fixed_offset_error_m"] for r in rows])),
                       "scenes": rows})
    result = {"setting": "post-hoc ground-truth gap diagnostic on already scored sets",
              "note": "Uses annotated 3D centers and visually positive pair labels; no inference or tuning. Center gap is not physical contact height.",
              "groups": groups}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({group["child_class"]: {"mean_gap_m": group["mean_true_center_gap_m"],
                    "mean_abs_fixed_offset_error_m": group["mean_fixed_offset_error_m"]}
                    for group in groups}, indent=2))


if __name__ == "__main__":
    main()
