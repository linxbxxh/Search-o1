import unittest
from scripts.critic import evaluate_candidate


class CriticTest(unittest.TestCase):
    def test_critic_trigger_and_schema(self):
        out = evaluate_candidate("I am not sure. maybe maybe", min_score_to_pass=0.9)
        self.assertIn("score", out)
        self.assertIn("issues", out)
        self.assertIn("revise_required", out)
        self.assertTrue(out["revise_required"])


if __name__ == '__main__':
    unittest.main()
