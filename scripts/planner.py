import json
import re
from typing import Callable, Dict, List, Any, Tuple


def _token_budget_trim(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _simple_subquestions(question: str, max_subquestions: int = 3) -> List[str]:
    parts = [p.strip() for p in re.split(r"[?;。！？]", question) if p.strip()]
    if not parts:
        return [question.strip()]
    return parts[:max_subquestions]


def _keywords(text: str, top_n: int = 4) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    stop = {"the", "a", "an", "is", "are", "what", "how", "why", "to", "of", "in", "for", "and", "or"}
    words = [t for t in tokens if t not in stop and len(t) > 2]
    uniq = []
    seen = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq[:top_n]


def build_planner_output(
    question: str,
    search_fn: Callable[[str], List[Dict[str, Any]]],
    context_budget_tokens: int = 256,
    max_subquestions: int = 3,
    queries_per_subq: int = 2,
) -> Dict[str, Any]:
    subqs = _simple_subquestions(question, max_subquestions=max_subquestions)
    planner_subqs = []
    evidence = []
    seen_sources = set()

    brief_budget = max(64, context_budget_tokens // max(1, len(subqs)))

    for i, sq in enumerate(subqs, start=1):
        kw = _keywords(sq)
        queries = [sq]
        if kw:
            queries.append(" ".join(kw))
        queries = queries[:queries_per_subq]
        sq_id = f"SQ{i}"
        planner_subqs.append(
            {
                "id": sq_id,
                "question": sq,
                "queries": queries,
                "evidence_needed": "facts and definitions directly supporting the subquestion",
            }
        )

        supports = []
        for q in queries:
            for doc in search_fn(q):
                source_id = doc.get("url") or doc.get("source_id") or doc.get("name") or "unknown"
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                snippet = _token_budget_trim((doc.get("snippet") or "").replace("\n", " ").strip(), brief_budget)
                if not snippet:
                    continue
                supports.append({"source_id": source_id, "snippet": snippet})
                if len(supports) >= 2:
                    break
            if len(supports) >= 2:
                break

        evidence.append(
            {
                "subq_id": sq_id,
                "claim": _token_budget_trim(f"Key evidence for: {sq}", 32),
                "support": supports,
            }
        )

    return {
        "task_type": "reasoning_with_retrieval",
        "subquestions": planner_subqs,
        "initial_evidence_brief": evidence,
    }


def planner_to_context_block(planner_json: Dict[str, Any], context_budget_tokens: int = 256) -> Tuple[str, Dict[str, Any]]:
    compact = json.dumps(planner_json, ensure_ascii=False)
    compact = _token_budget_trim(compact, context_budget_tokens)
    return f"\n[PLANNER_JSON]\n{compact}\n[/PLANNER_JSON]\n", planner_json
