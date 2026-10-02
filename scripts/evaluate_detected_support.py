"""Measure automatic support-pair coverage using saved detector predictions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.support_proposals import propose_support_relations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-name", default="YOLO11n-seg")
    args = parser.parse_args()
    manifest = {item["scene_id"]: item for item in
                json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    labels = {item["scene_id"]: item for item in
              json.loads(args.labels.read_text(encoding="utf-8"))["scenes"]}
    detector = {item["scene_id"]: item for item in
                json.loads(args.detections.read_text(encoding="utf-8"))["scenes"]}
    if manifest.keys() != labels.keys() or manifest.keys() != detector.keys():
        raise ValueError("Manifest, labels, and detections must cover identical scene IDs")
    rows = []
    for scene_id, scene in manifest.items():
        annotation = labels[scene_id]
        predictions = detector[scene_id]["detections"]
        depth = np.load(args.root / scene_id / "predicted_depth.npy", allow_pickle=False)
        layout = build_layout(depth, predictions, Intrinsics(**scene["intrinsics"]))
        proposals = propose_support_relations(layout)
        pred_by_id = {item["instance_id"]: item for item in predictions}
        truth_by_id = {item["instance_id"]: item for item in
                       json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))}
        child_box = truth_by_id[annotation["child_id"]]["bbox_2d"]
        parent_box = truth_by_id[annotation["parent_id"]]["bbox_2d"]
        pair_matches = []
        for proposal in proposals:
            child_iou = box_iou(child_box, pred_by_id[proposal["child_id"]]["bbox_2d"])
            parent_iou = box_iou(parent_box, pred_by_id[proposal["parent_id"]]["bbox_2d"])
            if child_iou >= 0.5 and parent_iou >= 0.5:
                pair_matches.append({**proposal, "child_iou": child_iou, "parent_iou": parent_iou})
        rows.append({"scene_id": scene_id, "label": annotation.get("supported_by_desk", annotation.get("supported_by_table")),
                     "detector_count": len(predictions), "proposal_count": len(proposals),
                     "matched_pair": bool(pair_matches), "matched_pairs": pair_matches})
    positive = [row for row in rows if row["label"] is True]
    negative = [row for row in rows if row["label"] is False]
    report = {
        "setting": f"{args.detector_name} detections, predicted metric depth, heuristic support proposals; IoU>=0.5 for both annotated objects",
        "positive_count": len(positive),
        "positive_pair_matched_count": sum(row["matched_pair"] for row in positive),
        "positive_pair_coverage": sum(row["matched_pair"] for row in positive) / len(positive)
        if positive else None,
        "negative_count": len(negative),
        "negative_pair_matched_count": sum(row["matched_pair"] for row in negative),
        "uncertain_count": sum(row["label"] is None for row in rows),
        "note": "Pair matches require IoU>=0.5 for both annotated objects. Other emitted proposals are not exhaustively labeled; this is not detector precision. Official intrinsics supplied.",
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
          ("positive_count", "positive_pair_matched_count", "positive_pair_coverage")}, indent=2))


if __name__ == "__main__":
    main()
