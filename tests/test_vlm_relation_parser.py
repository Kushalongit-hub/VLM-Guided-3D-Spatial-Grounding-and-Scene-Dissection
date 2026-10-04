import unittest

from src.pipeline.vlm_support import gate_vlm_proposals, parse_answer


class VlmRelationParserTests(unittest.TestCase):
    def test_accepts_only_explicit_leading_judgment(self):
        self.assertEqual(parse_answer("YES"), "YES")
        self.assertEqual(parse_answer("  no, because they are separated"), "NO")
        self.assertEqual(parse_answer("UNCERTAIN."), "UNCERTAIN")
        self.assertEqual(parse_answer("It looks like yes"), "UNPARSEABLE")
        self.assertEqual(parse_answer("YESNO"), "UNPARSEABLE")

    def test_vlm_gate_abstains_and_rejects_misaligned_answers(self):
        proposals = [{"child_id": "a", "parent_id": "b"}, {"child_id": "c", "parent_id": "d"}]
        judgments = [{"child_id": "c", "parent_id": "d", "parsed_answer": "UNCERTAIN"},
                     {"child_id": "a", "parent_id": "b", "parsed_answer": "YES"}]
        self.assertEqual(gate_vlm_proposals(proposals, judgments), [proposals[0]])
        with self.assertRaises(ValueError):
            gate_vlm_proposals(proposals, judgments[:1])


if __name__ == "__main__":
    unittest.main()
