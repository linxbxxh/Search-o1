import json
import re
from collections import Counter
from typing import Any, Dict, List


def normalize_answer(text: str) -> str:
    line = text.strip().split("\n")[-1].strip().lower()
    line = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def aggregate_votes(candidates: List[Dict[str, Any]], mode: str = "majority") -> Dict[str, Any]:
    if not candidates:
        return {"candidates": [], "vote": {"winner": "", "distribution": {}, "mode": mode}}

    scores = Counter()
    for c in candidates:
        ans = c["answer_norm"]
        if mode == "evidence_weighted":
            ev = c.get("evidence_stats", {})
            weight = 1.0 + float(ev.get("coverage", 0.0)) + float(ev.get("critic_pass", 0.0))
        else:
            weight = 1.0
        scores[ans] += weight

    winner, _ = scores.most_common(1)[0]
    return {
        "candidates": candidates,
        "vote": {
            "winner": winner,
            "distribution": {k: round(v, 4) for k, v in scores.items()},
            "mode": mode,
        },
    }


def build_voting_payload(raw_answers: List[str], evidence_stats: List[Dict[str, Any]], mode: str = "majority") -> Dict[str, Any]:
    candidates = []
    for i, raw in enumerate(raw_answers):
        candidates.append(
            {
                "answer_norm": normalize_answer(raw),
                "raw_answer": raw,
                "evidence_stats": evidence_stats[i] if i < len(evidence_stats) else {},
            }
        )
    return aggregate_votes(candidates, mode=mode)


def voting_to_context_block(voting_json: Dict[str, Any], budget_tokens: int = 128) -> str:
    compact = json.dumps(voting_json, ensure_ascii=False)
    compact = " ".join(compact.split()[:budget_tokens])
    return f"\n[VOTING_JSON]\n{compact}\n[/VOTING_JSON]\n"
