import unittest

import numpy as np

from src.pipeline.radial_support import correct_radial_support_layout


class RadialSupportTests(unittest.TestCase):
    def test_reduces_height_residual_and_changes_depth_without_mutating_input(self):
        layout = {"objects": [
            {"instance_id": "cup", "center_point_3d": [0.0, -0.2, 2.0]},
            {"instance_id": "table", "center_point_3d": [0.0, 0.0, 2.2]},
        ]}
        up = np.array([0.0, -0.8, 0.6])
        old_gap = np.dot(np.array([0.0, -0.2, 2.0]) - np.array([0.0, 0.0, 2.2]), up)
        result = correct_radial_support_layout(layout, "cup", "table", up,
                                               offset_m=0.4, max_shift_m=0.5)
        cup = np.array(result["objects"][0]["center_point_3d"])
        table = np.array(result["objects"][1]["center_point_3d"])
        self.assertLess(abs(np.dot(cup - table, up) - 0.4), abs(old_gap - 0.4))
        self.assertNotEqual(float(cup[2]), 2.0)
        self.assertEqual(layout["objects"][0]["center_point_3d"], [0.0, -0.2, 2.0])

    def test_caps_large_radial_shift(self):
        layout = {"objects": [
            {"instance_id": "cup", "center_point_3d": [0.0, 0.0, 2.0]},
            {"instance_id": "table", "center_point_3d": [0.0, 0.0, 2.0]},
        ]}
        result = correct_radial_support_layout(layout, "cup", "table",
                                               np.array([0.0, 0.0, 1.0]),
                                               offset_m=2.0, max_shift_m=0.1)
        self.assertTrue(all(abs(x) <= 0.1 for x in
                            result["radial_support_correction"]["radial_shifts_m"]))


if __name__ == "__main__":
    unittest.main()
