import unittest
from scripts.planner import build_planner_output


class PlannerTest(unittest.TestCase):
    def test_planner_json_shape(self):
        def fake_search(_q):
            return [{"url": "u1", "snippet": "evidence snippet"}]

        out = build_planner_output("What is ATP and where is it produced?", search_fn=fake_search, context_budget_tokens=64)
        self.assertIn("task_type", out)
        self.assertIn("subquestions", out)
        self.assertIn("initial_evidence_brief", out)
        self.assertTrue(out["subquestions"][0]["id"].startswith("SQ"))


if __name__ == '__main__':
    unittest.main()
