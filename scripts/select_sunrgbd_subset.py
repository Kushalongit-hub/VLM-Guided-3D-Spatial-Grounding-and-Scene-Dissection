"""Freeze a sensor-balanced SUN RGB-D test subset before model evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.io import loadmat


SENSORS = ("kv1", "kv2", "xtion", "realsense")


def pairable_count(three, two) -> int:
    boxes3 = np.atleast_1d(three.groundtruth3DBB)
    boxes2 = np.atleast_1d(two.groundtruth2DBB)
    counts3 = Counter(str(item.classname) for item in boxes3)
    counts2 = Counter(str(item.classname) for item in boxes2 if bool(item.has3dbox))
    return sum(counts3[name] == 1 and counts2[name] == 1 for name in counts3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/sunrgbd_official"))
    parser.add_argument("--per-sensor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_sensor < 1:
        parser.error("--per-sensor must be positive")
    split_path = args.metadata_dir / "sunrgbd_testing_images.txt"
    test_paths = set(split_path.read_text(encoding="utf-8").splitlines())
    meta3_path = args.metadata_dir / "SUNRGBDMeta3DBB_v2.mat"
    meta2_path = args.metadata_dir / "SUNRGBDMeta2DBB_v2.mat"
    meta3 = loadmat(meta3_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
    meta2 = loadmat(meta2_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta2DBB"]
    eligible = {sensor: [] for sensor in SENSORS}
    for index, (three, two) in enumerate(zip(meta3, meta2)):
        rgb_path = str(three.rgbpath)
        rgb_path = "SUNRGBD/" + rgb_path.split("SUNRGBD/", 1)[-1]
        sensor = str(three.sensorType)
        if rgb_path not in test_paths or sensor not in eligible or not bool(three.valid):
            continue
        if three.sequenceName != two.sequenceName:
            continue
        count = pairable_count(three, two)
        if count >= 3:
            eligible[sensor].append({"metadata_index": index, "scene_id": f"sunrgbd_{index+1:06d}",
                                     "sensor": sensor, "pairable_objects": count,
                                     "sequence_name": str(three.sequenceName)})
    generator = random.Random(args.seed)
    selected = []
    for sensor in SENSORS:
        if len(eligible[sensor]) < args.per_sensor:
            raise ValueError(f"Only {len(eligible[sensor])} eligible {sensor} scenes")
        selected.extend(generator.sample(eligible[sensor], args.per_sensor))
    selected.sort(key=lambda scene: scene["metadata_index"])
    result = {"split": "sunrgbd_testing_images.txt", "selection_seed": args.seed,
              "per_sensor": args.per_sensor, "minimum_pairable_objects": 3,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (split_path, meta3_path, meta2_path)},
              "scenes": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scene_count": len(selected), "eligible_by_sensor":
                      {sensor: len(eligible[sensor]) for sensor in SENSORS},
                      "selected_indices": [scene["metadata_index"] for scene in selected]}, indent=2))


if __name__ == "__main__":
    main()
