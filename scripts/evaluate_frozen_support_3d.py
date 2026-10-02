"""Evaluate predeclared 3D support correction on a frozen SUN RGB-D subset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.radial_support import correct_radial_support_layout
from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout
from src.pipeline.support_proposals import propose_support_relations


def _method_errors(layout: dict, child: str, parent: str, truth: dict,
                   up: np.ndarray, protocol: dict) -> dict:
    config = protocol["correction"]
    sigmas = {item["instance_id"]: config["center_sigma_m"]["cup_or_mug"]
              if item["instance_id"] == child else
              config["center_sigma_m"]["table"] if item["instance_id"] == parent else
              config["center_sigma_m"]["other"] for item in layout["objects"]}
    relation = [{"child_id": child, "parent_id": parent,
                 "offset_m": config["offset_prior_m"],
                 "sigma_m": config["offset_sigma_m"]}]
    outputs = {"uncorrected": layout,
               "weighted": correct_support_layout(layout, relation, sigmas, up),
               "hard": correct_support_layout(layout, relation, sigmas, up, hard=True)}
    if "radial_correction" in protocol:
        config = protocol["radial_correction"]
        outputs["radial"] = correct_radial_support_layout(
            layout, child, parent, up,
            offset_m=config["offset_m"], child_sigma_m=config["child_sigma_m"],
            parent_sigma_m=config["parent_sigma_m"],
            relation_sigma_m=config["relation_sigma_m"],
            max_shift_m=config["max_shift_m"])
    values = {}
    for method, output in outputs.items():
        centers = {row["instance_id"]: np.asarray(row["center_point_3d"])
                   for row in output["objects"]}
        values[method] = {
            "child_error_m": float(np.linalg.norm(centers[child] - truth["child"])),
            "parent_error_m": float(np.linalg.norm(centers[parent] - truth["parent"])),
            "height_gap_error_m": float(abs(np.dot(centers[child] - centers[parent], up) -
                                            np.dot(truth["child"] - truth["parent"], up))),
        }
        values[method]["pair_mean_error_m"] = (values[method]["child_error_m"] +
                                                values[method]["parent_error_m"]) / 2
    return values


def _paired_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    methods = tuple(rows[0].keys())
    result = {"count": len(rows), "methods": {}}
    for method in methods:
        result["methods"][method] = {
            metric: float(np.mean([row[method][metric] for row in rows]))
            for metric in ("child_error_m", "parent_error_m", "pair_mean_error_m", "height_gap_error_m")}
    rng = np.random.default_rng(20261002)
    for other in ("uncorrected", "hard"):
        differences = np.asarray([row["weighted"]["child_error_m"] - row[other]["child_error_m"]
                                  for row in rows])
        samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        result[f"weighted_minus_{other}_child_error_m"] = {
            "mean": float(differences.mean()),
            "bootstrap_95pct_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
            "weighted_better_count": int(np.sum(differences < 0)),
        }
    if "radial" in methods:
        for other in ("uncorrected", "hard", "weighted"):
            differences = np.asarray([row["radial"]["child_error_m"] - row[other]["child_error_m"]
                                      for row in rows])
            samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
            result[f"radial_minus_{other}_child_error_m"] = {
                "mean": float(differences.mean()),
                "bootstrap_95pct_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
                "radial_better_count": int(np.sum(differences < 0)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_images_depth_or_3d_reference_inspection":
        raise ValueError("Expected a frozen protocol")
    if hashlib.sha256(args.selection.read_bytes()).hexdigest() != protocol["selection_sha256"]:
        raise ValueError("Selection does not match frozen protocol")
    repo = Path(__file__).resolve().parents[1]
    for key, name in (
        ("support_constraints_code_sha256", "src/pipeline/support_constraints.py"),
        ("support_proposals_code_sha256", "src/pipeline/support_proposals.py"),
        ("grounding_dino_adapter_code_sha256", "src/pipeline/grounding_dino_detector.py"),
    ):
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen code hash mismatch: {name}")
    if "radial_correction" in protocol and hashlib.sha256(
            (repo / "src/pipeline/radial_support.py").read_bytes()).hexdigest() != protocol["radial_support_code_sha256"]:
        raise ValueError("Frozen radial-support code hash mismatch")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["split"] != "train":
        raise ValueError("Expected disjoint SUN RGB-D training split selection")
    manifest = {row["scene_id"]: row for row in json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    labels_data = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels_data["status"] != "rgb_and_2d_box_labels_frozen_before_depth_or_3d_reference":
        raise ValueError("Expected RGB-only relation labels")
    labels = {row["scene_id"]: row for row in labels_data["scenes"]}
    detector_report = json.loads(args.detections.read_text(encoding="utf-8"))
    checkpoint = Path(detector_report["checkpoint"])
    weights = (checkpoint if checkpoint.is_absolute() else repo / checkpoint) / "model.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != protocol["grounding_dino_weights_sha256"]:
        raise ValueError("Detector weights differ from frozen protocol")
    if (detector_report["box_threshold"] != protocol["detector"]["box_threshold"] or
            detector_report["text_threshold"] != protocol["detector"]["text_threshold"] or
            detector_report["prompt_format"] != protocol["detector"]["prompt_format"]):
        raise ValueError("Detector configuration differs from frozen protocol")
    detections = {row["scene_id"]: row["detections"] for row in detector_report["scenes"]}
    if manifest.keys() != labels.keys() or manifest.keys() != detections.keys():
        raise ValueError("Scene IDs differ among manifest, labels, and detections")
    rows = []
    parent_class = protocol.get("parent_class", "table")
    support_key = f"supported_by_{parent_class}"
    for scene_id, scene in manifest.items():
        annotation = labels[scene_id]
        root = args.root
        oracle = json.loads((root / scene["objects"]).read_text(encoding="utf-8"))
        by_id = {row["instance_id"]: row for row in oracle}
        child, parent = annotation["child_id"], annotation["parent_id"]
        if by_id[child]["label"] not in ("cup", "mug") or by_id[parent]["label"] != parent_class:
            raise ValueError(f"Wrong paired object class in {scene_id}")
        rtilt, _ = read_calibration(root / scene_id / "calib.txt")
        up = camera_up_from_rtilt(rtilt)
        reference = {row["instance_id"]: np.asarray(row["center_point_3d"])
                     for row in json.loads((root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
        truth = {"child": reference[child], "parent": reference[parent]}
        depth = np.load(root / scene_id / "predicted_depth.npy", allow_pickle=False)
        camera = Intrinsics(**scene["intrinsics"])
        oracle_layout = build_layout(depth, oracle, camera)
        predicted = detections[scene_id]
        pred_by_id = {item["instance_id"]: item for item in predicted}
        detected_layout = build_layout(depth, predicted, camera)
        proposals = propose_support_relations(detected_layout)
        def best(ref: str, allowed: set[str]) -> tuple[str | None, float]:
            candidates = [(row["instance_id"], box_iou(by_id[ref]["bbox_2d"], row["bbox_2d"]))
                          for row in predicted if row["label"] in allowed]
            return max(candidates, key=lambda x: x[1]) if candidates else (None, 0.0)
        pred_child, child_iou = best(child, {"cup", "mug"})
        pred_parent, parent_iou = best(parent, {"table", "desk"})
        covered = child_iou >= protocol["matching"]["box_iou_min"] and parent_iou >= protocol["matching"]["box_iou_min"]
        matched_proposals = []
        for proposal in proposals:
            proposal_child = pred_by_id[proposal["child_id"]]
            proposal_parent = pred_by_id[proposal["parent_id"]]
            child_overlap = box_iou(by_id[child]["bbox_2d"], proposal_child["bbox_2d"])
            parent_overlap = box_iou(by_id[parent]["bbox_2d"], proposal_parent["bbox_2d"])
            if (child_overlap >= protocol["matching"]["box_iou_min"] and
                    parent_overlap >= protocol["matching"]["box_iou_min"]):
                matched_proposals.append((child_overlap + parent_overlap, proposal))
        pair_proposed = bool(matched_proposals)
        row = {"scene_id": scene_id, "label": annotation[support_key],
               "child_iou": child_iou, "parent_iou": parent_iou,
               "pair_detected": covered, "pair_proposed": pair_proposed,
               "proposal_count": len(proposals)}
        if annotation[support_key] is True:
            row["oracle_methods"] = _method_errors(oracle_layout, child, parent, truth, up, protocol)
            if pair_proposed:
                matched = max(matched_proposals, key=lambda value: value[0])[1]
                row["automatic_methods"] = _method_errors(
                    detected_layout, matched["child_id"], matched["parent_id"], truth, up, protocol)
        rows.append(row)
    positive = [row for row in rows if row["label"] is True]
    negative = [row for row in rows if row["label"] is False]
    automatic = [row for row in positive if "automatic_methods" in row]
    report = {"setting": "frozen SUN RGB-D train-complement scenes, RGB-predicted depth, official intrinsics",
              "positive_count": len(positive), "negative_count": len(negative),
              "uncertain_count": sum(row["label"] is None for row in rows),
              "positive_detected_pair_count": sum(row["pair_detected"] for row in positive),
              "positive_proposed_pair_count": len(automatic),
              "negative_proposed_pair_count": sum(row["pair_proposed"] for row in negative),
              "oracle_positive": _paired_summary([row["oracle_methods"] for row in positive]),
              "automatic_matched_positive": _paired_summary([row["automatic_methods"] for row in automatic]),
              "note": "Automatic matched-pair 3D errors use oracle 2D boxes only to align an emitted proposal to its annotated objects for scoring. They are conditional on a matched proposal; unmatched positives are counted separately. Other proposals lack exhaustive labels. Bootstrap resamples scenes and does not correct room dependence.",
              "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("positive_count", "negative_count",
          "positive_proposed_pair_count", "negative_proposed_pair_count",
          "oracle_positive", "automatic_matched_positive")}, indent=2))


if __name__ == "__main__":
    main()
