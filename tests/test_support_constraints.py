import unittest

import numpy as np

from src.pipeline.support_constraints import camera_up_from_rtilt, correct_support_layout


class SupportConstraintTests(unittest.TestCase):
    def setUp(self):
        self.layout = {"objects": [
            {"instance_id": "table", "center_point_3d": [0, 0, 2]},
            {"instance_id": "mug", "center_point_3d": [0, -0.3, 2]},
        ]}
        self.relation = [{"child_id": "mug", "parent_id": "table", "offset_m": 0.5,
                          "sigma_m": 0.1}]

    def test_weighted_update_and_unchanged_input(self):
        result = correct_support_layout(self.layout, self.relation,
                                        {"table": 0.1, "mug": 0.3})
        heights = [-obj["center_point_3d"][1] for obj in result["objects"]]
        self.assertAlmostEqual(heights[0], -0.01818181818)
        self.assertAlmostEqual(heights[1], 0.46363636363)
        self.assertEqual(self.layout["objects"][1]["center_point_3d"], [0, -0.3, 2])

    def test_hard_contact_and_camera_up(self):
        result = correct_support_layout(self.layout, self.relation,
                                        {"table": 0.1, "mug": 0.3}, hard=True)
        heights = [-obj["center_point_3d"][1] for obj in result["objects"]]
        self.assertAlmostEqual(heights[1] - heights[0], 0.5)
        np.testing.assert_allclose(camera_up_from_rtilt(np.eye(3)), [0, -1, 0])

    def test_invalid_relation(self):
        with self.assertRaises(ValueError):
            correct_support_layout(self.layout, [{**self.relation[0], "parent_id": "missing"}],
                                   {"table": 0.1, "mug": 0.3})


if __name__ == "__main__":
    unittest.main()
