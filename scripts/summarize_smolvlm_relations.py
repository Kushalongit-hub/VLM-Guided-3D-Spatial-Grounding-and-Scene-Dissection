"""Summarize fixed-prompt VLM pair judgments without rerunning the model."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/smolvlm_laptop_bottle_relation12.json"
OUTPUT = ROOT / "results/smolvlm_relation12_summary.json"
ANSWERS = ("YES", "NO", "UNCERTAIN", "UNPARSEABLE")


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["status"] != "complete" or len(report["scenes"]) != 12:
        raise ValueError("Expected complete frozen 12-scene VLM report")
    groups = {}
    for child_class in ("laptop", "bottle", "all"):
        rows = [row for row in report["scenes"] if child_class == "all" or row["child_class"] == child_class]
        groups[child_class] = {
            "positive": {answer: sum(row["rgb_relation"] is True and row["parsed_answer"] == answer
                                     for row in rows) for answer in ANSWERS},
            "negative": {answer: sum(row["rgb_relation"] is False and row["parsed_answer"] == answer
                                     for row in rows) for answer in ANSWERS},
        }
    result = {
        "setting": "exploratory oracle-box VLM relation judgments on previously inspected scenes",
        "source": str(SOURCE.relative_to(ROOT)),
        "groups": groups,
        "interpretation": "If YES is used as the only accept signal, 6/8 RGB-positive pairs are accepted and 1/4 RGB-negative pairs is accepted. UNCERTAIN and unparseable answers abstain. This is not an independent or end-to-end score.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups["all"], indent=2))


if __name__ == "__main__":
    main()
