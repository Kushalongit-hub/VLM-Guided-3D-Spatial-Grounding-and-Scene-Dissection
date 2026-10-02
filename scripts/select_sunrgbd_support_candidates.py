"""Freeze mug/cup-and-surface candidate scenes before image or depth inspection.

Co-occurring category labels are only candidates, never support ground truth.
The script reads split membership and 2D/3D annotation *availability* but no
3D center, size, depth, or image pixels.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/sunrgbd_official"))
    parser.add_argument("--split", choices=("test", "train"), default="test",
                        help="SUN RGB-D official test split or its complement")
    parser.add_argument("--exclude", type=Path, required=True,
                        help="Existing 20-scene selection, excluded to separate development and evaluation")
    parser.add_argument("--additional-exclude", type=Path, action="append", default=[],
                        help="Previously frozen candidate selection to exclude; may repeat")
    parser.add_argument("--parent-class", choices=("table", "desk", "coffee_table"),
                        default="table")
    parser.add_argument("--child-class", choices=("cup_or_mug", "lamp", "laptop", "bottle", "bowl", "book"),
                        default="cup_or_mug")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-index-gap", type=int, default=0,
                        help="Exclude nearby prior indices and sample one per nearby-index cluster")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.min_index_gap < 0:
        parser.error("--min-index-gap must be nonnegative")
    root = args.metadata_dir
    paths = {name: root / name for name in
             ("SUNRGBDMeta3DBB_v2.mat", "SUNRGBDMeta2DBB_v2.mat", "sunrgbd_testing_images.txt")}
    split = set(paths["sunrgbd_testing_images.txt"].read_text(encoding="utf-8").splitlines())
    prior = json.loads(args.exclude.read_text(encoding="utf-8"))
    excluded = {int(item["metadata_index"]) for item in prior["scenes"]}
    for exclusion in args.additional_exclude:
        extra = json.loads(exclusion.read_text(encoding="utf-8"))
        excluded.update(int(item["metadata_index"]) for item in extra["scenes"])
    three = loadmat(paths["SUNRGBDMeta3DBB_v2.mat"], squeeze_me=True,
                    struct_as_record=False)["SUNRGBDMeta"]
    two = loadmat(paths["SUNRGBDMeta2DBB_v2.mat"], squeeze_me=True,
                  struct_as_record=False)["SUNRGBDMeta2DBB"]
    eligible = []
    for index, (scene3, scene2) in enumerate(zip(three, two)):
        if (any(abs(index - prior_index) < args.min_index_gap for prior_index in excluded)
                if args.min_index_gap else index in excluded):
            continue
        if not bool(scene3.valid) or scene3.sequenceName != scene2.sequenceName:
            continue
        rgb = "SUNRGBD/" + str(scene3.rgbpath).split("SUNRGBD/", 1)[-1]
        if (rgb in split) != (args.split == "test"):
            continue
        count3 = Counter(str(obj.classname) for obj in np.atleast_1d(scene3.groundtruth3DBB))
        count2 = Counter(str(obj.classname) for obj in np.atleast_1d(scene2.groundtruth2DBB)
                         if bool(obj.has3dbox))
        child_classes = ("mug", "cup") if args.child_class == "cup_or_mug" else (args.child_class,)
        children = [name for name in child_classes if count3[name] == count2[name] == 1]
        if count3[args.parent_class] != 1 or count2[args.parent_class] != 1 or len(children) != 1:
            continue
        eligible.append({"metadata_index": index, "scene_id": f"sunrgbd_{index + 1:06d}",
                         "candidate_child_class": children[0], "candidate_parent_class": args.parent_class,
                         "sensor": str(scene3.sensorType), "sequence_name": str(scene3.sequenceName)})
    generator = random.Random(args.seed)
    if args.min_index_gap:
        clusters = []
        for item in eligible:
            if not clusters or item["metadata_index"] - clusters[-1][-1]["metadata_index"] >= args.min_index_gap:
                clusters.append([])
            clusters[-1].append(item)
        pool = [generator.choice(cluster) for cluster in clusters]
    else:
        pool = eligible
    if len(pool) < args.count:
        raise ValueError(f"Only {len(pool)} spaced candidates; requested {args.count}")
    selected = sorted(generator.sample(pool, args.count),
                      key=lambda item: item["metadata_index"])
    result = {
        "status": "frozen_candidates_not_relation_labels", "selection_seed": args.seed,
        "split": args.split,
        "eligible_count": len(eligible), "spaced_pool_count": len(pool),
        "min_index_gap": args.min_index_gap, "selection_rule":
        f"unique paired {args.child_class} and {args.parent_class} classes in SUN RGB-D {args.split} split",
        "excluded_selection_sha256": hashlib.sha256(args.exclude.read_bytes()).hexdigest(),
        "additional_exclusion_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in args.additional_exclude],
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in paths.items()},
        "scenes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"eligible_count": len(eligible), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
