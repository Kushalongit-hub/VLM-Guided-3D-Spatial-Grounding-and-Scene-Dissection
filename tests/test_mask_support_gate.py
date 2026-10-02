import unittest

import numpy as np

from src.pipeline.mask_support_gate import gate_support_proposals


class MaskSupportGateTests(unittest.TestCase):
    def test_contact_keeps_proposal_and_separation_abstains(self):
        child = np.zeros((30, 30), dtype=bool)
        child[5:15, 10:20] = True
        parent = np.zeros_like(child)
        parent[15:25, 8:22] = True
        proposal = [{"child_id": "cup", "parent_id": "table", "relation": "candidate"}]
        kept, evidence = gate_support_proposals(proposal, {"cup": child, "table": parent})
        self.assertEqual(len(kept), 1)
        self.assertTrue(evidence[0]["accepted"])
        self.assertEqual(proposal[0], {"child_id": "cup", "parent_id": "table", "relation": "candidate"})
        parent[:] = False
        parent[25:29, 8:22] = True
        kept, evidence = gate_support_proposals(proposal, {"cup": child, "table": parent})
        self.assertEqual(kept, [])
        self.assertFalse(evidence[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
