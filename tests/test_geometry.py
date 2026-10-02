import unittest

import numpy as np

from src.pipeline.geometry import Intrinsics, build_layout, lift_region


class GeometryTests(unittest.TestCase):
    def test_constant_plane_projects_to_metric_coordinates(self):
        depth = np.full((3, 3), 2.0)
        result = lift_region(depth, (0, 0, 3, 3), Intrinsics(2, 2, 1, 1))
        np.testing.assert_allclose(result["center_point_3d"], [0, 0, 2])
        np.testing.assert_allclose(result["visible_extent_3d"], [2, 2, 0])

    def test_invalid_depth_is_excluded(self):
        depth = np.array([[0.0, np.nan], [2.0, 2.0]])
        layout = build_layout(depth, [{"label": "box", "bbox_2d": [0, 0, 2, 2]}], Intrinsics(1, 1, 0, 0))
        self.assertEqual(layout["objects"][0]["valid_depth_pixels"], 2)

    def test_rejects_invalid_box(self):
        with self.assertRaises(ValueError):
            lift_region(np.ones((2, 2)), (0, 0, 3, 2), Intrinsics(1, 1, 0, 0))

    def test_mask_excludes_background_depth(self):
        depth = np.full((4, 4), 10.0)
        depth[1:3, 1:3] = 2.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        camera = Intrinsics(4, 4, 1.5, 1.5)
        box_result = lift_region(depth, (0, 0, 4, 4), camera)
        mask_result = lift_region(depth, (0, 0, 4, 4), camera, mask=mask)
        self.assertEqual(box_result["center_point_3d"][2], 10.0)
        self.assertEqual(mask_result["center_point_3d"][2], 2.0)
        self.assertEqual(mask_result["valid_depth_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
