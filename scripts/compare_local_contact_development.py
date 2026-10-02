"""Compare a local contact-height estimate on previously inspected scenes."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.contact_surface import correct_child_to_local_surface
from src.pipeline.radial_support import correct_radial_support_layout
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


def main() -> None:
    root = Path("data/sunrgbd_official")
    groups = [
        (root / "support_candidates", Path("results/sunrgbd_support_candidates_labels.json")),
        (root / "support_confirmation", Path("results/sunrgbd_support_confirmation_labels.json")),
    ]
    rows = []
    for directory, labels_path in groups:
        manifest = {row["scene_id"]: row for row in json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["scenes"]
        for label in labels:
            if label["supported_by_table"] is not True:
                continue
            scene_id = label["scene_id"]
            scene = manifest[scene_id]
            depth = np.load(directory / scene_id / "predicted_depth.npy", allow_pickle=False)
            objects = json.loads((directory / scene["objects"]).read_text(encoding="utf-8"))
            reference = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                         for obj in json.loads((directory / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
            child_id, parent_id = label["child_id"], label["parent_id"]
            rtilt, _ = read_calibration(directory / scene_id / "calib.txt")
            up = camera_up_from_rtilt(rtilt)
            layout = build_layout(depth, objects, Intrinsics(**scene["intrinsics"]))
            centers = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                       for obj in layout["objects"]}
            by_id = {obj["instance_id"]: obj for obj in objects}
            def depth_iqr(instance_id):
                x0, y0, x1, y1 = by_id[instance_id]["bbox_2d"]
                values = depth[y0:y1, x0:x1]
                values = values[np.isfinite(values) & (values > 0)]
                return float(np.percentile(values, 75) - np.percentile(values, 25))
            sigmas = {obj["instance_id"]: 0.4 if obj["instance_id"] == child_id else
                      0.8 if obj["instance_id"] == parent_id else 0.2 for obj in objects}
            relation = [{"child_id": child_id, "parent_id": parent_id,
                         "offset_m": 0.4, "sigma_m": 0.2}]
            def error(output):
                center = next(np.asarray(obj["center_point_3d"]) for obj in output["objects"]
                              if obj["instance_id"] == child_id)
                return float(np.linalg.norm(center - reference[child_id]))
            methods = {"uncorrected": error(layout),
                       "weighted_fixed": error(correct_support_layout(layout, relation, sigmas, up)),
                       "hard_fixed": error(correct_support_layout(layout, relation, sigmas, up, hard=True))}
            metadata = {}
            for blend in (0.25, 0.5, 0.75, 1.0):
                output = correct_child_to_local_surface(layout, depth, child_id, parent_id, up, blend)
                methods[f"local_{blend}"] = error(output)
                metadata = output["local_contact_correction"]
            for parent_sigma in (0.1, 0.2, 0.4, 0.8):
                output = correct_radial_support_layout(
                    layout, child_id, parent_id, up, parent_sigma_m=parent_sigma)
                methods[f"radial_parent_sigma_{parent_sigma}"] = error(output)
            rows.append({"scene_id": scene_id, "methods": methods,
                         "predicted_gap_m": float(np.dot(centers[child_id] - centers[parent_id], up)),
                         "reference_gap_m": float(np.dot(reference[child_id] - reference[parent_id], up)),
                         "predicted_child_height_m": float(np.dot(centers[child_id], up)),
                         "parent_depth_iqr_m": depth_iqr(parent_id),
                         "child_depth_iqr_m": depth_iqr(child_id),
                         "parent_center_error_m": float(np.linalg.norm(centers[parent_id] - reference[parent_id])),
                         "local_contact": metadata})
    summary = {name: {"mean_child_error_m": float(np.mean([row["methods"][name] for row in rows])),
                      "median_child_error_m": float(np.median([row["methods"][name] for row in rows]))}
               for name in rows[0]["methods"]}
    report = {"setting": "development scenes only; oracle boxes and relations; predicted depth",
              "positive_count": len(rows), "local_estimate_available_count": sum(row["local_contact"]["applied"] for row in rows),
              "summary": summary, "scenes": rows}
    output = Path("results/local_contact_development.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("positive_count", "local_estimate_available_count", "summary")}, indent=2))


if __name__ == "__main__":
    main()
