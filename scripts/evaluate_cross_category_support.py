"""Score a frozen cross-category support transfer on oracle boxes and RGB depth."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.radial_support import correct_radial_support_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_images_depth_or_3d_reference_inspection":
        raise ValueError("Protocol was not frozen")
    repo = Path(__file__).resolve().parents[1]
    checks = {
        "selection_sha256": args.selection,
        "child_ray_code_sha256": repo / "src/pipeline/child_ray_support.py",
        "paired_ray_code_sha256": repo / "src/pipeline/radial_support.py",
        "height_correction_code_sha256": repo / "src/pipeline/support_constraints.py",
        "depth_checkpoint_sha256": repo / "checkpoints/depth_anything_v2_metric_hypersim_vits.pth",
    }
    for name, path in checks.items():
        if sha(path) != protocol[name]:
            raise ValueError(f"Frozen hash mismatch: {path}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["split"] != "test":
        raise ValueError("Expected official test-split selection")
    scenes = {row["scene_id"]: row for row in selection["scenes"]}
    labels_data = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels_data["status"] != "rgb_and_2d_box_labels_frozen_before_depth_or_3d_reference":
        raise ValueError("Expected pre-depth RGB-only labels")
    labels = {row["scene_id"]: row for row in labels_data["scenes"]}
    manifest = {row["scene_id"]: row for row in json.loads(
        (args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    if scenes.keys() != labels.keys() or scenes.keys() != manifest.keys():
        raise ValueError("Selection, labels and manifest scene IDs differ")
    rows = []
    for scene_id, selection_row in scenes.items():
        scene = manifest[scene_id]
        annotation = labels[scene_id]
        child_id, parent_id = annotation["child_id"], annotation["parent_id"]
        oracle = json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))
        objects_by_id = {row["instance_id"]: row for row in oracle}
        if (objects_by_id[child_id]["label"] != selection_row["candidate_child_class"] or
                objects_by_id[parent_id]["label"] != selection_row["candidate_parent_class"]):
            raise ValueError(f"Wrong paired classes in {scene_id}")
        row = {"scene_id": scene_id, "relation_label": annotation["supported_by_table"],
               "sensor": selection_row["sensor"], "sequence_name": selection_row["sequence_name"]}
        if row["relation_label"] is True:
            predicted_depth = np.load(args.root / scene_id / "predicted_depth.npy", allow_pickle=False)
            layout = build_layout(predicted_depth, oracle, Intrinsics(**scene["intrinsics"]))
            rtilt, _ = read_calibration(args.root / scene_id / "calib.txt")
            up = camera_up_from_rtilt(rtilt)
            hcfg = protocol["methods"]["height_exact"]
            relation = [{"child_id": child_id, "parent_id": parent_id,
                         "offset_m": hcfg["offset_m"], "sigma_m": hcfg["relation_sigma_m"]}]
            sigmas = {obj["instance_id"]: (hcfg["child_sigma_m"] if obj["instance_id"] == child_id else
                       hcfg["parent_sigma_m"] if obj["instance_id"] == parent_id else 0.2)
                      for obj in layout["objects"]}
            methods = {
                "uncorrected": layout,
                "height_exact": correct_support_layout(layout, relation, sigmas, up, hard=True),
                "paired_radial": correct_radial_support_layout(
                    layout, child_id, parent_id, up, **protocol["methods"]["paired_radial"]),
                "child_only_ray_soft": correct_child_ray_support_layout(
                    layout, child_id, parent_id, up, **protocol["methods"]["child_only_ray_soft"]),
            }
            truth = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                     for obj in json.loads((args.root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
            true_gap = float((truth[child_id] - truth[parent_id]) @ up)
            row["methods"] = {}
            for name, output in methods.items():
                centers = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                           for obj in output["objects"]}
                child_error = float(np.linalg.norm(centers[child_id] - truth[child_id]))
                parent_error = float(np.linalg.norm(centers[parent_id] - truth[parent_id]))
                row["methods"][name] = {
                    "child_error_m": child_error,
                    "parent_error_m": parent_error,
                    "pair_mean_error_m": (child_error + parent_error) / 2,
                    "height_gap_error_m": float(abs((centers[child_id] - centers[parent_id]) @ up - true_gap)),
                }
        rows.append(row)
    positive = [row for row in rows if row["relation_label"] is True]
    summary = {}
    if positive:
        for name in protocol["methods"]:
            summary[name] = {metric: float(np.mean([row["methods"][name][metric] for row in positive]))
                             for metric in ("child_error_m", "parent_error_m", "pair_mean_error_m",
                                            "height_gap_error_m")}
    comparisons = {}
    for other in ("paired_radial", "uncorrected", "height_exact"):
        if not positive:
            continue
        differences = np.asarray([row["methods"]["child_only_ray_soft"]["child_error_m"] -
                                  row["methods"][other]["child_error_m"] for row in positive])
        rng = np.random.default_rng(20261002)
        draws = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        comparisons[f"child_only_minus_{other}_child_error_m"] = {
            "mean": float(differences.mean()),
            "scene_bootstrap_95pct_interval": np.percentile(draws, [2.5, 97.5]).tolist(),
            "child_only_better_count": int(np.sum(differences < 0)),
        }
    report = {"setting": "frozen SUN RGB-D official test-split lamp/table category transfer",
              "candidate_count": len(rows), "positive_count": len(positive),
              "negative_count": sum(row["relation_label"] is False for row in rows),
              "uncertain_count": sum(row["relation_label"] is None for row in rows),
              "oracle_positive": summary, "comparisons": comparisons,
              "note": "Category transfer with fixed cup-derived 0.4m prior and oracle boxes/relations. Small scene bootstrap is descriptive; index spacing does not guarantee room independence.",
              "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "positive_count", "negative_count",
          "uncertain_count", "oracle_positive", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
