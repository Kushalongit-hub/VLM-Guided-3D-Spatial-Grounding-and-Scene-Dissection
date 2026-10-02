import unittest
from types import SimpleNamespace

import numpy as np

from scripts.fetch_sunrgbd_official_subset import archive_member
from scripts.select_sunrgbd_subset import pairable_count


class SunRgbdSelectionTests(unittest.TestCase):
    def test_doubled_slash_metadata_path_resolves_to_archive_member(self):
        original = "/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0210//image/NYU0210.jpg"
        self.assertEqual(archive_member(original),
                         "SUNRGBD/kv1/NYUdata/NYU0210/image/NYU0210.jpg")

    def test_repeated_classes_are_not_counted_as_unambiguous_pairs(self):
        three = SimpleNamespace(groundtruth3DBB=np.array([
            SimpleNamespace(classname="chair"), SimpleNamespace(classname="chair"),
            SimpleNamespace(classname="table")], dtype=object))
        two = SimpleNamespace(groundtruth2DBB=np.array([
            SimpleNamespace(classname="chair", has3dbox=True),
            SimpleNamespace(classname="chair", has3dbox=True),
            SimpleNamespace(classname="table", has3dbox=True)], dtype=object))
        self.assertEqual(pairable_count(three, two), 1)


if __name__ == "__main__":
    unittest.main()
