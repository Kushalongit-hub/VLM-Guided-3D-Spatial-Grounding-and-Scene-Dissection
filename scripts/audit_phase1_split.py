"""Audit frozen Phase 1 membership and conservative source-group overlap."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def source_group(scene):
    # SUN RGB-D sequenceName ends in a frame/image identifier. The parent path
    # is a conservative proxy for a shared scan; some sensors use broader roots.
    return str(Path(scene["sequence_name"]).parent).replace("\\", "/")


def audit(development, final):
    dev = development["scenes"]
    test = final["scenes"]
    dev_ids = [row["scene_id"] for row in dev]
    test_ids = [row["scene_id"] for row in test]
    groups = defaultdict(lambda: {"development": [], "final": []})
    for split, rows in (("development", dev), ("final", test)):
        for row in rows:
            groups[source_group(row)][split].append(row["scene_id"])
    shared = {name: members for name, members in sorted(groups.items())
              if members["development"] and members["final"]}
    return {
        "development_count": len(dev),
        "final_count": len(test),
        "duplicate_development_ids": sorted(id for id, count in Counter(dev_ids).items() if count > 1),
        "duplicate_final_ids": sorted(id for id, count in Counter(test_ids).items() if count > 1),
        "overlapping_scene_ids": sorted(set(dev_ids) & set(test_ids)),
        "shared_source_group_count": len(shared),
        "shared_source_groups": shared,
        "group_definition": "parent path of SUN RGB-D sequence_name; conservative proxy, not independently verified room identity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=Path("results/sunrgbd_phase1_development20_selection.json"))
    parser.add_argument("--final", type=Path, default=Path("results/sunrgbd_phase1_final40_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("results/sunrgbd_phase1_split_audit.json"))
    args = parser.parse_args()
    result = audit(json.loads(args.development.read_text(encoding="utf-8")),
                   json.loads(args.final.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "shared_source_groups"}, indent=2))


if __name__ == "__main__":
    main()
