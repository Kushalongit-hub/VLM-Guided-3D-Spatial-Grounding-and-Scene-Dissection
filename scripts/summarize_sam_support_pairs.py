"""Paired descriptive uncertainty for the post-hoc SAM support study."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [row for row in source["scenes"] if "support_child_error_m" in row]
    if not rows:
        raise ValueError("No positive paired support rows")
    pairs = (
        ("mask_uncorrected", "box_uncorrected"),
        ("mask_child_ray_soft", "box_child_ray_soft"),
        ("mask_child_ray_soft", "mask_uncorrected"),
        ("box_child_ray_soft", "box_uncorrected"),
    )
    comparisons = {}
    for left, right in pairs:
        differences = np.asarray([row["support_child_error_m"][left] -
                                  row["support_child_error_m"][right] for row in rows])
        rng = np.random.default_rng(20261002)
        draws = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        comparisons[f"{left}_minus_{right}"] = {
            "mean_m": float(differences.mean()),
            "better_count": int(np.sum(differences < 0)),
            "scene_bootstrap_95pct_interval_m": np.percentile(draws, [2.5, 97.5]).tolist(),
        }
    report = {"setting": "post-hoc paired SAM/DINO support interaction on prior table scenes",
              "positive_paired_count": len(rows), "comparisons": comparisons,
              "note": "Scenes were already scored in earlier research work. Bootstrap resamples images, not independent rooms; nearby frames may be correlated. These are descriptive intervals, not a fresh held-out result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
