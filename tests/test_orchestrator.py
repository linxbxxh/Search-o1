import unittest
from types import SimpleNamespace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from orchestrator import Orchestrator, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY


class DummySP:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class OrchestratorTest(unittest.TestCase):
    def _args(self, **overrides):
        base = dict(
            blocklist_patterns='ignore previous,reveal key,system prompt',
            max_injected_chars=200,
            enable_planner=True,
            planner_pre_retrieve=False,
            max_plan_queries=3,
            plan_summary_max_chars=300,
            enable_critic=True,
            critic_min_context_chars=200,
            critic_max_chars=120,
            critic_mode='event',
            seed=42,
            enable_sc=True,
            sc_n=3,
            sc_vote_mode='evidence_constrained',
            sc_share_retrieval=True,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_extract_between(self):
        orch = Orchestrator(self._args(), llm=None, tokenizer=None, sampling_params_cls=DummySP, log_fn=lambda _: None)
        txt = f"abc {BEGIN_SEARCH_QUERY}hello world{END_SEARCH_QUERY} zzz"
        self.assertEqual(orch.extract_between(txt, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY), 'hello world')

    def test_planner_schema_valid(self):
        orch = Orchestrator(self._args(), llm=None, tokenizer=None, sampling_params_cls=DummySP, log_fn=lambda _: None)
        block = orch.planner_block_for_question('What is ATP?', lambda q: [{'url': 'u1', 'snippet': 'ATP is energy currency'}])
        self.assertIn('[PLANNER_STRUCTURED]', block)
        self.assertIn('"plan"', block)
        self.assertIn('"queries"', block)
        self.assertIn('"constraints"', block)

    def test_critic_schema_valid(self):
        orch = Orchestrator(self._args(), llm=None, tokenizer=None, sampling_params_cls=DummySP, log_fn=lambda _: None)
        out = orch.critic_decision('goal', 'I am not sure', 'no evidence')
        self.assertIn(out['status'], ['pass', 'revise', 'search_more', 'abort'])
        self.assertIsInstance(out['reasons'], list)
        self.assertIn('patch', out)

    def test_sc_vote_deterministic_on_fixed_evidence(self):
        orch = Orchestrator(self._args(sc_vote_mode='evidence_constrained'), llm=None, tokenizer=None, sampling_params_cls=DummySP, log_fn=lambda _: None)
        cands = ['Answer: ATP synthase', 'Final: ATP synthase', 'answer glucose']
        winner = orch.vote_candidates(cands, evidence_summary='ATP synthase is supported by evidence.')
        self.assertIn('ATP synthase'.lower(), winner.lower())


if __name__ == '__main__':
    unittest.main()
