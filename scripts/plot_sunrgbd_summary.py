"""Plot the frozen object-center benchmark by SUN RGB-D sensor type."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    sensors = ("kv1", "kv2", "xtion", "realsense")
    labels = ("Kinect v1", "Kinect v2", "Xtion", "RealSense")
    x = np.arange(len(sensors))
    width = 0.36
    sensor = [report["by_sensor"][name]["sensor_median_error_m"] for name in sensors]
    predicted = [report["by_sensor"][name]["predicted_median_error_m"] for name in sensors]
    counts = [report["by_sensor"][name]["object_count"] for name in sensors]
    fig, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    bars_a = axis.bar(x-width/2, sensor, width, label="Sensor depth", color="#2f80b7")
    bars_b = axis.bar(x+width/2, predicted, width, label="Predicted depth", color="#e18938")
    axis.bar_label(bars_a, fmt="%.2f", padding=3)
    axis.bar_label(bars_b, fmt="%.2f", padding=3)
    axis.set_xticks(x, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    axis.set_ylabel("Median 3D center error (m)")
    axis.set_ylim(0, max(predicted) * 1.23)
    axis.set_title("SUN RGB-D: same oracle 2D boxes and camera calibration")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
