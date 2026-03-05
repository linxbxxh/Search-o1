import unittest
from scripts.voting import build_voting_payload


class VotingTest(unittest.TestCase):
    def test_vote_majority(self):
        raw = ["Final Answer: 42", "answer 42", "final answer 7"]
        payload = build_voting_payload(raw, evidence_stats=[{}, {}, {}], mode="majority")
        self.assertIn("candidates", payload)
        self.assertIn("vote", payload)
        self.assertTrue(payload["vote"]["winner"])


if __name__ == '__main__':
    unittest.main()
