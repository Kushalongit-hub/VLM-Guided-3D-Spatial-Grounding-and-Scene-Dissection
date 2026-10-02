import unittest

import numpy as np

from src.pipeline.child_ray_support import correct_child_ray_support_layout


class ChildRaySupportTests(unittest.TestCase):
    def test_moves_child_on_camera_ray_and_preserves_parent(self):
        layout = {"objects": [
            {"instance_id": "cup", "center_point_3d": [0.4, -0.1, 2.0]},
            {"instance_id": "desk", "center_point_3d": [0.1, 0.2, 2.2]},
        ]}
        corrected = correct_child_ray_support_layout(
            layout, "cup", "desk", np.array([0.0, -1.0, 0.0]), offset_m=0.4)
        original = np.array(layout["objects"][0]["center_point_3d"])
        moved = np.array(corrected["objects"][0]["center_point_3d"])
        self.assertEqual(corrected["objects"][1], layout["objects"][1])
        self.assertEqual(layout["objects"][0]["center_point_3d"], [0.4, -0.1, 2.0])
        self.assertLess(np.linalg.norm(np.cross(original, moved)), 1e-12)
        self.assertNotEqual(corrected["child_ray_support_correction"]["radial_shift_m"], 0.0)

    def test_duplicate_ids_fail(self):
        layout = {"objects": [
            {"instance_id": "same", "center_point_3d": [0, 0, 1]},
            {"instance_id": "same", "center_point_3d": [0, 0, 2]},
        ]}
        with self.assertRaises(ValueError):
            correct_child_ray_support_layout(layout, "same", "other", np.array([0, -1, 0]))
