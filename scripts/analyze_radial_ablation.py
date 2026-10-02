"""Post-hoc stronger controls for the frozen camera-ray support evaluations.

This analysis never changes a frozen protocol or its primary report. It uses
the same annotated positive pairs and predicted depth, but the child-only
controls were specified after the original results were viewed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.radial_support import correct_radial_support_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


def child_only_ray(child: np.ndarray, parent: np.ndarray, up: np.ndarray,
                   offset_m: float, child_sigma_m: float,
                   relation_sigma_m: float, max_shift_m: float,
                   hard: bool = False) -> np.ndarray:
    """Adjust only the child depth along its original camera ray."""
    ray = child / np.linalg.norm(child)
    projection = float(ray @ up)
    innovation = offset_m - float((child - parent) @ up)
    if hard:
        shift = innovation / projection if abs(projection) > 1e-9 else 0.0
    else:
        shift = (child_sigma_m**2 * projection * innovation /
                 (child_sigma_m**2 * projection**2 + relation_sigma_m**2))
    shift = float(np.clip(shift, -max_shift_m, max_shift_m))
    return child + ray * shift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = protocol["radial_correction"]
    parent_class = protocol.get("parent_class", "table")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["scenes"]
    manifest = {row["scene_id"]: row for row in json.loads(
        (args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    rows = []
    for annotation in labels:
        if annotation[f"supported_by_{parent_class}"] is not True:
            continue
        scene_id = annotation["scene_id"]
        scene = manifest[scene_id]
        child_id, parent_id = annotation["child_id"], annotation["parent_id"]
        rtilt, _ = read_calibration(args.root / scene_id / "calib.txt")
        up = camera_up_from_rtilt(rtilt)
        depth = np.load(args.root / scene_id / "predicted_depth.npy", allow_pickle=False)
        objects = json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))
        layout = build_layout(depth, objects, Intrinsics(**scene["intrinsics"]))
        centers = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                   for obj in layout["objects"]}
        child, parent = centers[child_id], centers[parent_id]
        reference = json.loads((args.root / scene["reference"]).read_text(encoding="utf-8"))
        truth = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                 for obj in reference["objects"]}
        height_config = protocol["correction"]
        relation = [{"child_id": child_id, "parent_id": parent_id,
                     "offset_m": height_config["offset_prior_m"],
                     "sigma_m": height_config["offset_sigma_m"]}]
        sigmas = {key: (height_config["center_sigma_m"]["cup_or_mug"] if key == child_id else
                        height_config["center_sigma_m"]["table"] if key == parent_id else
                        height_config["center_sigma_m"]["other"])
                  for key in centers}
        hard_layout = correct_support_layout(layout, relation, sigmas, up, hard=True)
        radial_layout = correct_radial_support_layout(layout, child_id, parent_id, up, **config)
        child_only_layout = correct_child_ray_support_layout(
            layout, child_id, parent_id, up,
            offset_m=config["offset_m"], child_sigma_m=config["child_sigma_m"],
            relation_sigma_m=config["relation_sigma_m"], max_shift_m=config["max_shift_m"])
        methods = {
            "uncorrected": child,
            "height_exact": next(np.asarray(o["center_point_3d"]) for o in hard_layout["objects"]
                                 if o["instance_id"] == child_id),
            "paired_radial": next(np.asarray(o["center_point_3d"]) for o in radial_layout["objects"]
                                  if o["instance_id"] == child_id),
            "child_only_ray_soft": next(np.asarray(o["center_point_3d"])
                                        for o in child_only_layout["objects"]
                                        if o["instance_id"] == child_id),
            "child_only_ray_exact": child_only_ray(child, parent, up,
                                                   config["offset_m"], config["child_sigma_m"],
                                                   config["relation_sigma_m"], config["max_shift_m"], hard=True),
        }
        rows.append({"scene_id": scene_id, "child_error_m": {
            name: float(np.linalg.norm(point - truth[child_id]))
            for name, point in methods.items()}})
    means = {name: float(np.mean([row["child_error_m"][name] for row in rows]))
             for name in rows[0]["child_error_m"]}
    comparisons = {}
    for other in ("height_exact", "child_only_ray_soft", "child_only_ray_exact"):
        differences = np.asarray([row["child_error_m"]["paired_radial"] -
                                  row["child_error_m"][other] for row in rows])
        rng = np.random.default_rng(20261002)
        draws = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        comparisons[f"paired_radial_minus_{other}"] = {
            "mean_m": float(differences.mean()),
            "scene_bootstrap_95pct_interval_m": np.percentile(draws, [2.5, 97.5]).tolist(),
            "paired_radial_better_count": int(np.sum(differences < 0)),
        }
    report = {"status": "post_hoc_stronger_baselines_not_frozen_primary_evaluation",
              "parent_class": parent_class, "positive_count": len(rows),
              "mean_child_error_m": means, "comparisons": comparisons,
              "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("positive_count", "mean_child_error_m", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
