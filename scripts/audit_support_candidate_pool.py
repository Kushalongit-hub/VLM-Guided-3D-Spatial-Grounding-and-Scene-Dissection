"""Count unused paired category candidates without reading images or 3D values."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/sunrgbd_official"))
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--min-index-gap", type=int, default=30)
    args = parser.parse_args()
    root = args.metadata_dir
    three = loadmat(root / "SUNRGBDMeta3DBB_v2.mat", squeeze_me=True,
                    struct_as_record=False)["SUNRGBDMeta"]
    two = loadmat(root / "SUNRGBDMeta2DBB_v2.mat", squeeze_me=True,
                  struct_as_record=False)["SUNRGBDMeta2DBB"]
    test = set((root / "sunrgbd_testing_images.txt").read_text(encoding="utf-8").splitlines())
    excluded = {int(scene["metadata_index"]) for path in args.exclude
                for scene in json.loads(path.read_text(encoding="utf-8"))["scenes"]}
    children = ("cup", "mug", "bottle", "bowl", "book", "lamp", "laptop")
    parents = ("table", "desk", "coffee_table", "counter", "night_stand")
    groups = defaultdict(list)
    for index, (scene3, scene2) in enumerate(zip(three, two)):
        if any(abs(index - old) < args.min_index_gap for old in excluded):
            continue
        if not bool(scene3.valid) or scene3.sequenceName != scene2.sequenceName:
            continue
        rgb = "SUNRGBD/" + str(scene3.rgbpath).split("SUNRGBD/", 1)[-1]
        split = "test" if rgb in test else "train"
        counts3 = Counter(str(obj.classname) for obj in np.atleast_1d(scene3.groundtruth3DBB))
        counts2 = Counter(str(obj.classname) for obj in np.atleast_1d(scene2.groundtruth2DBB)
                          if bool(obj.has3dbox))
        for child in children:
            if counts3[child] != 1 or counts2[child] != 1:
                continue
            for parent in parents:
                if counts3[parent] == counts2[parent] == 1:
                    groups[(split, child, parent)].append(index)
    rows = []
    for (split, child, parent), indices in sorted(groups.items()):
        groups_count = 0
        previous = None
        for index in indices:
            if previous is None or index - previous >= args.min_index_gap:
                groups_count += 1
            previous = index
        rows.append({"split": split, "child": child, "parent": parent,
                     "candidate_count": len(indices), "index_groups": groups_count})
    print(json.dumps(sorted(rows, key=lambda x: (-x["index_groups"], -x["candidate_count"])), indent=2))


if __name__ == "__main__":
    main()
