import unittest

from src.pipeline.grounding_dino_detector import suppress_prompt_duplicates


class GroundingDinoPostprocessTests(unittest.TestCase):
    def test_synonym_duplicates_collapse_but_separate_instances_remain(self):
        rows = [
            {"instance_id": "a", "label": "cup", "confidence": 0.8, "bbox_2d": [10, 10, 30, 30]},
            {"instance_id": "b", "label": "mug", "confidence": 0.7, "bbox_2d": [11, 11, 30, 30]},
            {"instance_id": "c", "label": "cup", "confidence": 0.6, "bbox_2d": [50, 50, 70, 70]},
            {"instance_id": "d", "label": "table", "confidence": 0.9, "bbox_2d": [0, 0, 100, 100]},
        ]
        self.assertEqual([row["instance_id"] for row in suppress_prompt_duplicates(rows)],
                         ["d", "a", "c"])


if __name__ == "__main__":
    unittest.main()
