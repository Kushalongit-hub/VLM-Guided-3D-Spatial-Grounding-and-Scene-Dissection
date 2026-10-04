import unittest

from src.pipeline.generic_support_proposals import propose_labeled_support_relations
from src.pipeline.support_proposals import propose_support_relations


class GenericSupportProposalTests(unittest.TestCase):
    def test_new_child_category_uses_same_geometry_without_changing_frozen_default(self):
        layout = {"objects": [
            {"instance_id": "child", "object": "bottle", "bbox_2d": [10, 10, 20, 30],
             "center_point_3d": [0.0, 0.0, 2.0]},
            {"instance_id": "parent", "object": "table", "bbox_2d": [0, 25, 40, 50],
             "center_point_3d": [0.0, 0.0, 2.2]},
        ]}
        self.assertEqual(propose_support_relations(layout), [])
        generic = propose_labeled_support_relations(layout, {"bottle"})
        self.assertEqual([(p["child_id"], p["parent_id"]) for p in generic],
                         [("child", "parent")])
        layout["objects"][0]["object"] = "cup"
        self.assertEqual(propose_labeled_support_relations(layout, {"cup", "mug"}),
                         propose_support_relations(layout))

    def test_configured_child_still_requires_depth_consistency(self):
        layout = {"objects": [
            {"instance_id": "child", "object": "laptop", "bbox_2d": [10, 10, 20, 30],
             "center_point_3d": [0.0, 0.0, 1.0]},
            {"instance_id": "parent", "object": "table", "bbox_2d": [0, 25, 40, 50],
             "center_point_3d": [0.0, 0.0, 2.0]},
        ]}
        self.assertEqual(propose_labeled_support_relations(layout, {"laptop"}), [])


if __name__ == "__main__":
    unittest.main()
