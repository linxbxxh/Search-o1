import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ma_rag_port import MARAGController


def test_stops_immediately_on_consensus():
    calls = {"retrieve": 0}

    def generate(prompt, n, round_id):
        return ["reason\n<answer>A</answer>"] * n

    def retrieve(query, top_k):
        calls["retrieve"] += 1
        return []

    result = MARAGController(generate, retrieve, num_workers=3, num_rounds=4).run("q", "A. x\nB. y")
    assert result.final_answer == "A"
    assert result.stopped_reason == "consensus"
    assert len(result.rounds) == 1
    assert calls["retrieve"] == 0


def test_conflict_triggers_query_retrieval_then_consensus():
    calls = {"retrieve": 0}

    def generate(prompt, n, round_id):
        if round_id == 1:
            return ["<answer>A</answer>", "<answer>B</answer>", "<answer>A</answer>"]
        assert "Documents" in prompt
        assert "Previous assistant answer" in prompt
        return ["verified\n<answer>B</answer>"] * n

    def query_gen(question, options, answers):
        return "[Query 1] disputed mechanism\n[Query 2] decisive fact"

    def retrieve(query, top_k):
        calls["retrieve"] += 1
        return [{"id": query, "title": query, "content": "evidence"}]

    result = MARAGController(
        generate,
        retrieve,
        query_generate_fn=query_gen,
        num_workers=3,
        num_rounds=3,
    ).run("q", "A. x\nB. y")

    assert result.final_answer == "B"
    assert result.stopped_reason == "consensus"
    assert len(result.rounds) == 2
    assert result.rounds[0].conflict_queries == ["disputed mechanism", "decisive fact"]
    assert calls["retrieve"] == 2


def test_round_budget_falls_back_to_majority():
    def generate(prompt, n, round_id):
        return ["<answer>A</answer>", "<answer>B</answer>", "<answer>A</answer>"]

    result = MARAGController(generate, lambda q, k: [], num_workers=3, num_rounds=1).run("q")
    assert result.final_answer == "A"
    assert result.stopped_reason == "round_budget"


def test_parse_queries_deduplicates():
    parsed = MARAGController.parse_queries(
        "[Query 1] alpha beta\n[Query 2] alpha beta\n[Query 3] gamma", limit=4
    )
    assert parsed == ["alpha beta", "gamma"]


def test_official_entropy_order_is_high_to_low():
    seen_second_round_prompt = {"text": ""}

    def generate(prompt, n, round_id):
        if round_id == 1:
            return [
                {"text": "low entropy\n<answer>A</answer>", "confidence": 0.10},
                {"text": "high entropy\n<answer>B</answer>", "confidence": 0.90},
                {"text": "mid entropy\n<answer>A</answer>", "confidence": 0.50},
            ]
        seen_second_round_prompt["text"] = prompt
        return ["<answer>A</answer>"] * n

    result = MARAGController(
        generate,
        lambda q, k: [],
        query_generate_fn=lambda q, o, a: ["fact"],
        num_workers=3,
        num_rounds=2,
        entropy_order_mode="official_code",
    ).run("q")

    assert result.rounds[0].ranked_candidate_indices == [1, 2, 0]
    prompt = seen_second_round_prompt["text"]
    assert prompt.index("high entropy") < prompt.index("mid entropy") < prompt.index("low entropy")


def test_low_first_entropy_order_is_low_to_high():
    def generate(prompt, n, round_id):
        if round_id == 1:
            return [
                {"text": "low\n<answer>A</answer>", "confidence": 0.10},
                {"text": "high\n<answer>B</answer>", "confidence": 0.90},
                {"text": "mid\n<answer>A</answer>", "confidence": 0.50},
            ]
        return ["<answer>A</answer>"] * n

    result = MARAGController(
        generate,
        lambda q, k: [],
        query_generate_fn=lambda q, o, a: ["fact"],
        num_workers=3,
        num_rounds=2,
        entropy_order_mode="low_first",
    ).run("q")

    assert result.rounds[0].ranked_candidate_indices == [0, 2, 1]
