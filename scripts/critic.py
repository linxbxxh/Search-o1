import json
import re
from typing import Any, Dict, List, Tuple


UNCERTAINTY_MARKERS = ["not sure", "uncertain", "maybe", "i think", "可能", "不确定"]


def evaluate_candidate(output_text: str, min_score_to_pass: float = 0.65) -> Dict[str, Any]:
    lowered = output_text.lower()
    has_search_result = "<|begin_search_result|>" in lowered
    has_answer = bool(re.search(r"final answer|answer", lowered))
    uncertainty = any(marker in lowered for marker in UNCERTAINTY_MARKERS)

    faithfulness = 0.75 if has_search_result else 0.45
    logic = 0.75 if has_answer else 0.5
    clarity = 0.7 if len(output_text.strip()) > 40 else 0.4

    issues: List[Dict[str, str]] = []
    if not has_search_result:
        issues.append({"type": "missing_evidence", "where": "global", "fix": "Add evidence-backed statements from retrieved results."})
    if uncertainty:
        issues.append({"type": "contradiction", "where": "global", "fix": "Resolve uncertain or self-contradictory claims with explicit support."})
    if not has_answer:
        issues.append({"type": "format", "where": "ending", "fix": "Provide a clear final answer section."})

    avg = (faithfulness + logic + clarity) / 3.0
    revise_required = avg < min_score_to_pass or len(issues) > 0

    return {
        "score": {
            "faithfulness": round(faithfulness, 3),
            "logic": round(logic, 3),
            "clarity": round(clarity, 3),
        },
        "issues": issues,
        "revise_required": revise_required,
    }


def critic_to_context_block(critic_json: Dict[str, Any], budget_tokens: int = 128) -> Tuple[str, Dict[str, Any]]:
    compact = json.dumps(critic_json, ensure_ascii=False)
    compact_tokens = compact.split()[:budget_tokens]
    compact = " ".join(compact_tokens)
    block = (
        "\n[CRITIC_JSON]\n"
        f"{compact}\n"
        "Please revise your previous answer using the issue list. Keep evidence-grounded reasoning.\n"
        "[/CRITIC_JSON]\n"
    )
    return block, critic_json


def should_run_critic(intermediate_text: str, enabled: bool) -> bool:
    if not enabled:
        return False
    lowered = intermediate_text.lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS) or "<|begin_search_result|>" not in lowered
