import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.pipeline.depth_estimator import infer_metric_depth, load_metric_model


class DepthEstimatorTests(unittest.TestCase):
    def test_missing_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_metric_model(Path(directory) / "missing.pth")

    def test_invalid_image_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                infer_metric_depth(Path(directory) / "missing.jpg",
                                   Path(directory) / "missing.pth")


if __name__ == "__main__":
    unittest.main()
