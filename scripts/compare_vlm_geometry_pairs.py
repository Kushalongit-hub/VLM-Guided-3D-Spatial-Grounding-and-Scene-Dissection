"""Compare VLM YES decisions with mask-gated geometry on identical matched pairs."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/smolvlm_detected_vs_geometry9.json"


def main() -> None:
    vlm = json.loads((ROOT / "results/smolvlm_detected_pair9.json").read_text())
    if vlm["status"] != "complete":
        raise ValueError("Detected-pair VLM report is incomplete")
    transfer = {}
    for name in ("laptop_table6", "bottle_table6"):
        report = json.loads((ROOT / "results" / f"sunrgbd_{name}_transfer.json").read_text())
        transfer.update({row["scene_id"]: row for row in report["scenes"]})
    rows = []
    for pair in vlm["matched"]:
        scene = transfer[pair["scene_id"]]
        if pair["rgb_relation"] != scene["visible_relation"]:
            raise ValueError("RGB label changed")
        rows.append({"scene_id": pair["scene_id"], "rgb_relation": pair["rgb_relation"],
                     "geometry_mask_contact_accept": scene["paired_candidate_after_gate"],
                     "vlm_yes_accept": pair["parsed_answer"] == "YES",
                     "vlm_answer": pair["parsed_answer"]})
    counts = {}
    for method in ("geometry_mask_contact_accept", "vlm_yes_accept"):
        counts[method] = {
            "positive_accepted": sum(row["rgb_relation"] is True and row[method] for row in rows),
            "positive_total": sum(row["rgb_relation"] is True for row in rows),
            "negative_accepted": sum(row["rgb_relation"] is False and row[method] for row in rows),
            "negative_total": sum(row["rgb_relation"] is False for row in rows),
        }
    result = {"setting": "post-hoc same nine annotation-matched detected pairs",
              "claim_limits": "Pairs were selected through oracle IoU matching on previously inspected scenes; whole-scene relation precision and missed-detection coverage are not captured.",
              "counts": counts, "scenes": rows}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
