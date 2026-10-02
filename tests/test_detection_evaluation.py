import unittest

from src.pipeline.detection_evaluation import box_iou, match_boxes


class DetectionEvaluationTests(unittest.TestCase):
    def test_iou_and_one_to_one_matching(self):
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        truth = [{"instance_id": "a", "label": "chair", "bbox_2d": [0, 0, 10, 10]},
                 {"instance_id": "b", "label": "chair", "bbox_2d": [0, 0, 10, 10]}]
        prediction = [{"instance_id": "p", "label": "chair", "bbox_2d": [0, 0, 10, 10]}]
        score = match_boxes(truth, prediction, class_aware=True)
        self.assertEqual(score["matched_count"], 1)
        self.assertEqual(score["coverage"], 0.5)

    def test_class_aware_denominator_excludes_unmapped_labels(self):
        truth = [{"instance_id": "a", "label": "chair", "bbox_2d": [0, 0, 10, 10]},
                 {"instance_id": "b", "label": "desk", "bbox_2d": [20, 20, 30, 30]}]
        prediction = [{"instance_id": "p", "label": "bed", "bbox_2d": [0, 0, 10, 10]}]
        strict = match_boxes(truth, prediction, class_aware=True)
        agnostic = match_boxes(truth, prediction)
        self.assertEqual((strict["eligible_count"], strict["matched_count"]), (1, 0))
        self.assertEqual((agnostic["eligible_count"], agnostic["matched_count"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
