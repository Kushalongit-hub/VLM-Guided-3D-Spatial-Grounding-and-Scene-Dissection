"""Post-hoc sensitivity to nearby SUN RGB-D metadata frames."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = {row["scene_id"]: row for row in
                json.loads(args.selection.read_text(encoding="utf-8"))["scenes"]}
    scenes = [row for row in json.loads(args.report.read_text(encoding="utf-8"))["scenes"]
              if row["label"] is True]
    values = []
    for row in scenes:
        info = metadata[row["scene_id"]]
        delta = (row["oracle_methods"]["radial"]["child_error_m"] -
                 row["oracle_methods"]["hard"]["child_error_m"])
        values.append({"scene_id": row["scene_id"], "metadata_index": info["metadata_index"],
                       "sensor": info["sensor"], "sequence_name": info["sequence_name"],
                       "radial_minus_hard_m": delta})
    values.sort(key=lambda row: row["metadata_index"])
    groups = []
    for item in values:
        if not groups or item["metadata_index"] - groups[-1][-1]["metadata_index"] >= 50:
            groups.append([])
        groups[-1].append(item)
    clusters = [{"scene_ids": [item["scene_id"] for item in group],
                 "mean_radial_minus_hard_m": float(np.mean([item["radial_minus_hard_m"] for item in group]))}
                for group in groups]
    group_values = np.asarray([group["mean_radial_minus_hard_m"] for group in clusters])
    rng = np.random.default_rng(20261002)
    sample = rng.choice(group_values, size=(10000, len(group_values)), replace=True).mean(axis=1)
    non_b3do = [row["radial_minus_hard_m"] for row in values if "/b3dodata/" not in row["sequence_name"]]
    report = {"status": "post_hoc_sensitivity_not_predeclared",
              "group_rule": "Consecutive positive metadata indices separated by less than 50 form one group; no RGB or outcome information used to form groups",
              "scene_count": len(values), "group_count": len(clusters),
              "scene_weighted_mean_m": float(np.mean([row["radial_minus_hard_m"] for row in values])),
              "equal_group_mean_m": float(group_values.mean()),
              "equal_group_bootstrap_95pct_interval_m": np.percentile(sample, [2.5, 97.5]).tolist(),
              "excluding_b3dodata_scene_count": len(non_b3do),
              "excluding_b3dodata_mean_m": float(np.mean(non_b3do)),
              "groups": clusters}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
