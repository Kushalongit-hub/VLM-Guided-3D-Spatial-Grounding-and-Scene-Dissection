"""Evaluate frozen cup/mug-on-desk support proposals after RGB-only labeling."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_support_proposals import evaluate_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "support_proposals.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol["proposal_code_sha256"]:
        raise ValueError("Proposal code changed after protocol freeze")
    if hashlib.sha256(args.selection.read_bytes()).hexdigest() != protocol["selection_sha256"]:
        raise ValueError("Selection changed after protocol freeze")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels["status"] != "rgb_box_labels_frozen_before_predicted_depth_inference":
        raise ValueError("Expected RGB-only labels frozen before model inference")
    rows = evaluate_group(args.root, args.labels, "supported_by_desk")
    known = [row for row in rows if row["label"] is not None]
    tp = sum(row["label"] is True and row["candidate_pair_proposed"] for row in known)
    tn = sum(row["label"] is False and not row["candidate_pair_proposed"] for row in known)
    fp = sum(row["label"] is False and row["candidate_pair_proposed"] for row in known)
    fn = sum(row["label"] is True and not row["candidate_pair_proposed"] for row in known)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    report = {
        "setting": "frozen desk candidates; official oracle boxes and calibration; RGB-predicted depth",
        "selection_sha256": protocol["selection_sha256"],
        "proposal_code_sha256": protocol["proposal_code_sha256"],
        "positive_count": tp + fn, "negative_count": tn + fp,
        "uncertain_count": len(rows) - len(known),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "candidate_pair_precision": precision, "candidate_pair_recall": recall,
        "candidate_pair_f1": 2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall else None,
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
          ("positive_count", "negative_count", "uncertain_count", "tp", "tn", "fp", "fn")}, indent=2))


if __name__ == "__main__":
    main()
