import unittest

from PIL import Image

from src.pipeline.sam_segmenter import segment_detector_boxes


class SamSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (10, 8))

    def test_empty_detections_do_not_load_or_call_model(self):
        self.assertEqual(segment_detector_boxes(self.image, [], None, None), ({}, {}))

    def test_invalid_and_duplicate_detector_boxes_fail_before_model(self):
        bad = [{"instance_id": "a", "bbox_2d": [0, 0, 11, 8]}]
        with self.assertRaises(ValueError):
            segment_detector_boxes(self.image, bad, None, None)
        duplicate = [{"instance_id": "a", "bbox_2d": [0, 0, 5, 5]},
                     {"instance_id": "a", "bbox_2d": [5, 0, 10, 5]}]
        with self.assertRaises(ValueError):
            segment_detector_boxes(self.image, duplicate, None, None)
