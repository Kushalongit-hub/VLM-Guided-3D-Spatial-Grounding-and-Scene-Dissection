import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.pipeline.benchmark import run_manifest


class BenchmarkTests(unittest.TestCase):
    def test_manifest_aggregates_real_metric_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "depth.npy", np.full((3, 3), 2.0))
            (root / "objects.json").write_text(json.dumps([
                {"label": "box", "instance_id": "a", "bbox_2d": [0, 0, 3, 3]}
            ]), encoding="utf-8")
            (root / "reference.json").write_text(json.dumps({
                "coordinate_system": "camera_x_right_y_down_z_forward_meters",
                "objects": [{"instance_id": "a", "center_point_3d": [0, 0, 2]}],
            }), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"scenes": [{
                "scene_id": "one", "depth": "depth.npy", "objects": "objects.json",
                "reference": "reference.json",
                "intrinsics": {"fx": 2, "fy": 2, "cx": 1, "cy": 1},
            }]}), encoding="utf-8")
            result = run_manifest(root / "manifest.json")
            self.assertEqual(result["coverage"], 1.0)
            self.assertEqual(result["median_centroid_error_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
