import unittest

import numpy as np

from src.pipeline.object_detector import suppress_duplicate_masks


class MaskSuppressionTests(unittest.TestCase):
    def test_same_class_duplicate_is_suppressed_without_removing_distinct_instance(self):
        left = np.zeros((5, 5), dtype=bool)
        left[0:2, 0:2] = True
        right = np.zeros((5, 5), dtype=bool)
        right[3:5, 3:5] = True
        objects = [
            {"instance_id": "low", "label": "chair", "confidence": 0.3},
            {"instance_id": "right", "label": "chair", "confidence": 0.6},
            {"instance_id": "high", "label": "chair", "confidence": 0.9},
        ]
        kept, masks = suppress_duplicate_masks(
            objects, {"low": left.copy(), "right": right, "high": left})
        self.assertEqual([obj["instance_id"] for obj in kept], ["high", "right"])
        self.assertEqual(set(masks), {"high", "right"})


if __name__ == "__main__":
    unittest.main()
