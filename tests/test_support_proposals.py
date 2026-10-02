import unittest

from src.pipeline.support_proposals import propose_support_relations


def obj(key, label, box, depth):
    return {"instance_id": key, "object": label, "bbox_2d": box,
            "center_point_3d": [0, 0, depth]}


class SupportProposalTests(unittest.TestCase):
    def test_selects_nearest_depth_parent(self):
        layout = {"objects": [
            obj("cup", "cup", [4, 2, 6, 4], 2.0),
            obj("near", "table", [0, 3, 10, 9], 2.2),
            obj("far", "counter", [0, 3, 10, 9], 2.6),
        ]}
        result = propose_support_relations(layout)
        self.assertEqual([(x["child_id"], x["parent_id"]) for x in result], [("cup", "near")])

    def test_rejects_rear_surface_and_wrong_footpoint(self):
        layout = {"objects": [
            obj("cup", "mug", [4, 2, 6, 4], 3.0),
            obj("front", "table", [0, 3, 10, 9], 1.0),
            obj("side", "table", [10, 3, 20, 9], 3.0),
        ]}
        self.assertEqual(propose_support_relations(layout), [])


if __name__ == "__main__":
    unittest.main()
