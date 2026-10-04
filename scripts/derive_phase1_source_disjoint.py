"""Derive a source-group-disjoint final subset before any final image inspection."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_phase1_split import source_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=Path("results/sunrgbd_phase1_development20_selection.json"))
    parser.add_argument("--final", type=Path, default=Path("results/sunrgbd_phase1_final40_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("results/sunrgbd_phase1_final_source_disjoint30_selection.json"))
    args = parser.parse_args()
    development = json.loads(args.development.read_text(encoding="utf-8"))
    final = json.loads(args.final.read_text(encoding="utf-8"))
    dev_groups = {source_group(row) for row in development["scenes"]}
    kept = [row for row in final["scenes"] if source_group(row) not in dev_groups]
    excluded = [row["scene_id"] for row in final["scenes"] if source_group(row) in dev_groups]
    result = {
        "status": "pre_annotation_metadata_only_derived_selection",
        "parent_final_selection": str(args.final).replace("\\", "/"),
        "parent_final_sha256": hashlib.sha256(args.final.read_bytes()).hexdigest(),
        "development_selection_sha256": hashlib.sha256(args.development.read_bytes()).hexdigest(),
        "rule": "Keep final scenes whose sequence_name parent path is absent from development",
        "source_group_caveat": "Parent path is a conservative source proxy; independently verified room IDs unavailable",
        "excluded_scene_ids": excluded,
        "category_counts": {f"{child}+{parent}": count for (child, parent), count in
                            Counter((row["candidate_child_class"], row["candidate_parent_class"])
                                    for row in kept).items()},
        "scenes": kept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Retained {len(kept)} of {len(final['scenes'])} final scenes; excluded {len(excluded)} shared-source scenes")


if __name__ == "__main__":
    main()
