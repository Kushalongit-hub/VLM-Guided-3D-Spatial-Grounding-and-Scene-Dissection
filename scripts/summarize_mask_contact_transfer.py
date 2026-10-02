"""Summarize the fixed visible-mask contact rule on two existing scene sets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "results/sunrgbd_support_radial15_sam_pairs.json",
    ROOT / "results/sunrgbd_support_train15_sam_pairs.json",
)
OUTPUT = ROOT / "results/sunrgbd_mask_contact_transfer_summary.json"


def main() -> None:
    totals = {"positive": {"detected_pairs": 0, "retained": 0},
              "negative": {"detected_pairs": 0, "retained": 0}}
    per_set = []
    for path in INPUTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        counts = {"positive": {"detected_pairs": 0, "retained": 0},
                  "negative": {"detected_pairs": 0, "retained": 0}}
        for scene in report["scenes"]:
            if not (scene["child_matched"] and scene["parent_matched"]):
                continue
            if scene["relation_label"] not in (True, False):
                continue
            label = "positive" if scene["relation_label"] else "negative"
            retained = scene.get("mask_contact", {}).get("contact_fraction", 0) > 0
            counts[label]["detected_pairs"] += 1
            counts[label]["retained"] += int(retained)
            totals[label]["detected_pairs"] += 1
            totals[label]["retained"] += int(retained)
        per_set.append({"input": str(path.relative_to(ROOT)), **counts})
    result = {
        "setting": "diagnostic transfer on previously inspected scenes; no independent test claim",
        "rule": "visible child/parent SAM mask adjacency fraction > 0, 8px vertical radius",
        "denominator": "annotated pairs with both detector boxes matched at IoU >= 0.5",
        "per_set": per_set,
        "combined": totals,
        "caveat": "Pair labels and matching use annotation; whole-scene false positives, missed detections, and unknown supports are excluded.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["combined"], indent=2))


if __name__ == "__main__":
    main()
