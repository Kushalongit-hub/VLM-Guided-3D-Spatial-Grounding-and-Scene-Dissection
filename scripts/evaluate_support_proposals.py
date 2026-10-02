"""Diagnostic check of box/depth support proposals on inspected SUN RGB-D pairs.

This selection was inspected while designing the heuristic. Scores are
resubstitution diagnostics, not independent generalization estimates.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.support_proposals import propose_support_relations


def evaluate_group(root: Path, labels_path: Path,
                   label_field: str = "supported_by_table") -> list[dict]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = {item["scene_id"]: item for item in manifest["scenes"]}
    if set(scenes) != {item["scene_id"] for item in labels["scenes"]}:
        raise ValueError("Labels and manifest scene IDs differ")
    rows = []
    for annotation in labels["scenes"]:
        scene_id = annotation["scene_id"]
        scene = scenes[scene_id]
        objects = json.loads((root / scene["objects"]).read_text(encoding="utf-8"))
        layout = build_layout(np.load(root / scene_id / "predicted_depth.npy", allow_pickle=False),
                              objects, Intrinsics(**scene["intrinsics"]))
        proposals = propose_support_relations(layout)
        pair_found = any(item["child_id"] == annotation["child_id"] and
                         item["parent_id"] == annotation["parent_id"] for item in proposals)
        rows.append({"scene_id": scene_id, "label": annotation[label_field],
                     "candidate_pair_proposed": pair_found, "proposals": proposals})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--pilot-labels", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--confirmation-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate_group(args.pilot_root, args.pilot_labels)
    rows.extend(evaluate_group(args.confirmation_root, args.confirmation_labels))
    known = [row for row in rows if row["label"] is not None]
    tp = sum(row["label"] is True and row["candidate_pair_proposed"] for row in known)
    tn = sum(row["label"] is False and not row["candidate_pair_proposed"] for row in known)
    fp = sum(row["label"] is False and row["candidate_pair_proposed"] for row in known)
    fn = sum(row["label"] is True and not row["candidate_pair_proposed"] for row in known)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    report = {
        "setting": "oracle 2D boxes, RGB-predicted depth, previously inspected mug/cup-table pairs",
        "warning": "Heuristic designed after inspecting these scenes; diagnostic only, not held-out accuracy.",
        "positive_count": tp + fn, "negative_count": tn + fp,
        "uncertain_count": len(rows) - len(known),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_on_labeled_candidate_pairs": precision,
        "recall_on_labeled_candidate_pairs": recall,
        "f1_on_labeled_candidate_pairs": 2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall else None,
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("positive_count", "negative_count",
          "uncertain_count", "tp", "tn", "fp", "fn")}, indent=2))


if __name__ == "__main__":
    main()
