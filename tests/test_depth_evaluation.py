import unittest

import numpy as np

from src.pipeline.depth_evaluation import depth_metrics


class DepthEvaluationTests(unittest.TestCase):
    def test_censored_and_missing_reference_pixels_are_excluded(self):
        prediction = np.array([[2.0, 2.0], [5.0, 4.0]])
        reference = np.array([[1.0, 0.0], [8.0, 4.0]])
        result = depth_metrics(prediction, reference)
        self.assertEqual(result["valid_pixels"], 2)
        self.assertAlmostEqual(result["mae_m"], 0.5)
        self.assertAlmostEqual(result["abs_rel"], 0.5)
        self.assertAlmostEqual(result["delta_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
