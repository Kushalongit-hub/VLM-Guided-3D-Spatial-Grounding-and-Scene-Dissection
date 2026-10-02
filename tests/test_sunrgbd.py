import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.pipeline.sunrgbd import convert_labels, decode_depth_png, read_calibration


class SunRgbdTests(unittest.TestCase):
    def test_depth_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            millimetres = np.array([[0, 2000], [4500, 8000]], dtype=np.uint16)
            packed = ((millimetres << 3) | (millimetres >> 13)).astype(np.uint16)
            self.assertTrue(cv2.imwrite(str(path), packed))
            np.testing.assert_allclose(decode_depth_png(path), millimetres / 1000)

    def test_label_center_and_box_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = "1 0 0 0 1 0 0 0 1"
            (root / "calib.txt").write_text(identity + "\n" +
                "2 0 0 0 2 0 1 1 1\n", encoding="utf-8")
            (root / "label.txt").write_text("table 1 1 2 2 0 2 -1 1 1 1 1 0\n", encoding="utf-8")
            rotation, intrinsics = read_calibration(root / "calib.txt")
            objects, reference = convert_labels(root / "label.txt", rotation, (4, 4))
            self.assertEqual(intrinsics, {"fx": 2, "fy": 2, "cx": 1, "cy": 1})
            self.assertEqual(objects[0]["bbox_2d"], [1, 1, 3, 3])
            self.assertEqual(reference["objects"][0]["center_point_3d"], [0, 1, 2])

    def test_non_identity_rotation_applies_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            swap_xy = "0 1 0 1 0 0 0 0 1"
            (root / "calib.txt").write_text(swap_xy + "\n" +
                "2 0 0 0 2 0 1 1 1\n", encoding="utf-8")
            (root / "label.txt").write_text("chair 0 0 1 1 1 0 0 1 1 1 1 0\n", encoding="utf-8")
            rotation, intrinsics = read_calibration(root / "calib.txt")
            objects, reference = convert_labels(root / "label.txt", rotation, (4, 4))
            self.assertEqual(intrinsics, {"fx": 2, "fy": 2, "cx": 1, "cy": 1})
            self.assertEqual(objects[0]["bbox_2d"], [0, 0, 1, 1])
            self.assertEqual(reference["objects"][0]["center_point_3d"], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
