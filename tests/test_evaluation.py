import unittest

from src.pipeline.evaluation import evaluate_centroids


class EvaluationTests(unittest.TestCase):
    def test_matching_and_coverage(self):
        frame = "camera_x_right_y_down_z_forward_meters"
        prediction = {"coordinate_system": frame, "objects": [
            {"instance_id": "a", "center_point_3d": [0, 0, 2]},
        ]}
        reference = {"coordinate_system": frame, "objects": [
            {"instance_id": "a", "center_point_3d": [0, 0, 1]},
            {"instance_id": "b", "center_point_3d": [0, 0, 3]},
        ]}
        result = evaluate_centroids(prediction, reference)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["median_centroid_error_m"], 1.0)

    def test_duplicate_ids_are_rejected(self):
        frame = "camera_x_right_y_down_z_forward_meters"
        obj = {"instance_id": "a", "center_point_3d": [0, 0, 1]}
        with self.assertRaises(ValueError):
            evaluate_centroids({"coordinate_system": frame, "objects": [obj, obj]},
                               {"coordinate_system": frame, "objects": [obj]})


if __name__ == "__main__":
    unittest.main()
