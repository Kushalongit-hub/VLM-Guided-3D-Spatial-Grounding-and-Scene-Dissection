"""Summarize paired sensor-depth and RGB-depth object-center benchmarks."""

import argparse
import json
from pathlib import Path

import numpy as np


def index_errors(report: dict) -> dict:
    return {(scene["scene_id"], obj["instance_id"]): obj["centroid_error_m"]
            for scene in report["scenes"] for obj in scene["per_object"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sensor = json.loads(args.sensor.read_text(encoding="utf-8"))
    predicted = json.loads(args.predicted.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    lookup = {entry["scene_id"]: entry["sensor"] for entry in selection["scenes"]}
    a, b = index_errors(sensor), index_errors(predicted)
    if set(a) != set(b) or {scene for scene, _ in a} != set(lookup):
        raise ValueError("Reports do not match the frozen selected objects/scenes")
    groups = {}
    scene_differences = {}
    for scene_id, sensor_type in lookup.items():
        keys = [key for key in a if key[0] == scene_id]
        sensor_errors = [a[key] for key in keys]
        predicted_errors = [b[key] for key in keys]
        scene_differences[scene_id] = float(np.mean(predicted_errors) - np.mean(sensor_errors))
        group = groups.setdefault(sensor_type, {"sensor_errors": [], "predicted_errors": [],
                                               "scene_count": 0})
        group["sensor_errors"].extend(sensor_errors)
        group["predicted_errors"].extend(predicted_errors)
        group["scene_count"] += 1
    by_sensor = {}
    for sensor_type, group in sorted(groups.items()):
        by_sensor[sensor_type] = {
            "scene_count": group["scene_count"],
            "object_count": len(group["sensor_errors"]),
            "sensor_median_error_m": float(np.median(group["sensor_errors"])),
            "predicted_median_error_m": float(np.median(group["predicted_errors"])),
            "sensor_mean_error_m": float(np.mean(group["sensor_errors"])),
            "predicted_mean_error_m": float(np.mean(group["predicted_errors"])),
        }
    scene_delta = np.array(list(scene_differences.values()))
    generator = np.random.default_rng(27)
    bootstrap = np.mean(generator.choice(scene_delta, size=(10000, len(scene_delta)), replace=True), axis=1)
    report = {
        "scene_count": len(lookup), "object_count": len(a),
        "sensor_median_error_m": sensor["median_centroid_error_m"],
        "predicted_median_error_m": predicted["median_centroid_error_m"],
        "improved_objects": sum(b[key] < a[key] for key in a),
        "worsened_objects": sum(b[key] > a[key] for key in a),
        "mean_scene_error_difference_predicted_minus_sensor_m": float(scene_delta.mean()),
        "bootstrap_95pct_ci_scene_mean_difference_m": [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
        "by_sensor": by_sensor, "per_scene_mean_difference_m": scene_differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scene_mean_difference_m"}, indent=2))


if __name__ == "__main__":
    main()
