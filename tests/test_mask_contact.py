import unittest

import numpy as np

from src.pipeline.mask_contact import visible_contact_fraction


class MaskContactTests(unittest.TestCase):
    def test_visible_support_and_separated_surface(self):
        child = np.zeros((30, 30), dtype=bool)
        child[5:15, 10:20] = True
        parent = np.zeros_like(child)
        parent[15:25, 8:22] = True
        touching = visible_contact_fraction(child, parent, vertical_radius_px=2)
        self.assertEqual(touching["contact_fraction"], 1.0)
        parent[:] = False
        parent[23:29, 8:22] = True
        separated = visible_contact_fraction(child, parent, vertical_radius_px=2)
        self.assertEqual(separated["contact_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
