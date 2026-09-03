"""Standalone runner for the faithful MA-RAG port.

No existing Search-o1 / TRACE-o1 file is modified. This runner adds vLLM token-
entropy extraction so the MA-RAG entropy variant can be reproduced more faithfully.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bing_search import extract_relevant_info, web_search
from ma_rag_port import MARAGController


def parse_args():
    p = argparse.ArgumentParser(description="Run vanilla MA-RAG-style multi-round reasoning on TRACE/Search-o1 data")
    p.add_argument("--dataset_path", required=True, help="JSON list containing Question/question fields")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--subset_num", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=5)
    p.add_argument("--num_rounds", type=int, default=5)
    p.add_argument("--queries_per_conflict", type=int, default=4)
    p.add_argument("--docs_per_query", type=int, default=2)
    p.add_argument("--search_top_k", type=int, default=10)
    p.add_argument("--search_provider", default="ddgs", choices=["ddgs", "google", "bing"])
    p.add_argument("--bing_subscription_key", default="None")
    p.add_argument("--bing_endpoint", default="https://api.bing.microsoft.com/v7.0/search")
    p.add_argument("--google_api_key", default="None")
    p.add_argument("--google_cse_id", default="None")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k_sampling", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entropy_top_k", type=int, default=20, help="Number of vLLM top-logprobs used for token entropy")
    p.add_argument(
        "--entropy_order_mode",
        default="official_code",
        choices=["official_code", "low_first", "none"],
        help="official_code matches released MA-RAG code (high entropy first); low_first follows prompt semantics",
    )
    p.add_argument(
        "--entropy_scope",
        default="post_think",
        choices=["post_think", "all"],
        help="Use tokens after the final </think> when available, matching upstream behavior",
    )
    return p.parse_args()


def question_of(item: Dict[str, Any]) -> str:
    return str(item.get("Question", item.get("question", ""))).strip()


def options_of(item: Dict[str, Any]) -> str:
    opts = item.get("options", item.get("Options", item.get("option", "")))
    if isinstance(opts, dict):
        return "\n".join(f"{k}. {v}" for k, v in opts.items())
    if isinstance(opts, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {v}" for i, v in enumerate(opts))
    return str(opts or "")


def _logprob_value(obj: Any) -> float:
    """Read a log-probability from vLLM Logprob objects or numeric fallbacks."""
    if hasattr(obj, "logprob"):
        return float(obj.logprob)
    if isinstance(obj, dict) and "logprob" in obj:
        return float(obj["logprob"])
    return float(obj)


def entropy_from_top_logprobs(token_logprobs: Any) -> float:
    """Entropy of a top-logprob distribution, renormalized over the returned set.

    This mirrors scipy.stats.entropy(exp(top_logprobs)) used in MA-RAG. Because
    vLLM returns a truncated top-k distribution, probabilities are explicitly
    renormalized before entropy is computed.
    """
    if not token_logprobs:
        return float("nan")
    values = token_logprobs.values() if isinstance(token_logprobs, dict) else token_logprobs
    lps = [_logprob_value(x) for x in values]
    if not lps:
        return float("nan")
    m = max(lps)
    weights = [math.exp(lp - m) for lp in lps]
    z = sum(weights)
    if z <= 0:
        return float("nan")
    probs = [w / z for w in weights]
    return -sum(p * math.log(p) for p in probs if p > 0)


def find_last_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
    """Return the index immediately after the last pattern occurrence, or 0."""
    if not pattern or len(pattern) > len(sequence):
        return 0
    last = -1
    plen = len(pattern)
    for i in range(len(sequence) - plen + 1):
        if list(sequence[i : i + plen]) == list(pattern):
            last = i + plen
    return max(0, last)


def entropy_metadata(output: Any, tokenizer: Any, scope: str = "post_think") -> Dict[str, Any]:
    """Extract all-token and post-</think> entropy statistics from a vLLM output."""
    token_ids = list(getattr(output, "token_ids", []) or [])
    token_logprobs = list(getattr(output, "logprobs", []) or [])
    usable = min(len(token_ids), len(token_logprobs))
    token_ids = token_ids[:usable]
    token_logprobs = token_logprobs[:usable]
    token_entropies = [entropy_from_top_logprobs(x) for x in token_logprobs]

    valid_all = [x for x in token_entropies if math.isfinite(x)]
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    post_think_start = find_last_subsequence(token_ids, think_end_ids)
    post_values = [x for x in token_entropies[post_think_start:] if math.isfinite(x)]

    selected = post_values if scope == "post_think" and post_values else valid_all
    mean_selected = sum(selected) / len(selected) if selected else None
    mean_all = sum(valid_all) / len(valid_all) if valid_all else None
    mean_post = sum(post_values) / len(post_values) if post_values else None

    return {
        "mean_token_entropy": mean_selected,
        "entropy_scope": "post_think" if scope == "post_think" and post_values else "all",
        "all_mean_token_entropy": mean_all,
        "post_think_mean_token_entropy": mean_post,
        "num_entropy_tokens": len(selected),
        "num_all_entropy_tokens": len(valid_all),
        "post_think_start_token": post_think_start,
        "token_entropies": token_entropies,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    cache_path = out_dir / "search_cache.json"

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.subset_num >= 0:
        data = data[: args.subset_num]

    search_cache: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            search_cache = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        gpu_memory_utilization=0.95,
    )

    def llm_outputs(prompt: str, n: int, temperature: float, max_tokens: int, seed: int, with_logprobs: bool):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        request = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k_sampling,
            seed=seed,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
            logprobs=args.entropy_top_k if with_logprobs else None,
        )
        return llm.generate([chat], sampling_params=request)[0].outputs

    def generate_fn(prompt: str, n: int, round_id: int):
        outputs = llm_outputs(
            prompt,
            n=n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed + round_id,
            with_logprobs=True,
        )
        records = []
        for out in outputs:
            meta = entropy_metadata(out, tokenizer, scope=args.entropy_scope)
            records.append(
                {
                    "text": out.text,
                    # Controller calls this field confidence for compatibility, but
                    # it is mean token entropy (lower = more confident).
                    "confidence": meta["mean_token_entropy"],
                    "entropy": meta,
                }
            )
        return records

    def query_generate_fn(question: str, options: str, answers: List[str]):
        prompt = MARAGController.build_conflict_query_prompt(question, options, answers)
        outputs = llm_outputs(prompt, n=1, temperature=0.0, max_tokens=1024, seed=args.seed, with_logprobs=False)
        return outputs[0].text

    def retrieve_fn(query: str, top_k: int):
        if query not in search_cache:
            raw = web_search(
                query,
                provider=args.search_provider,
                subscription_key=None if args.bing_subscription_key == "None" else args.bing_subscription_key,
                endpoint=args.bing_endpoint,
                market="en-US",
                language="en",
                max_results=args.search_top_k,
                google_api_key=None if args.google_api_key == "None" else args.google_api_key,
                google_cse_id=None if args.google_cse_id == "None" else args.google_cse_id,
            )
            search_cache[query] = raw
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(search_cache, f, ensure_ascii=False, indent=2)
        relevant = extract_relevant_info(search_cache[query])[:top_k]
        return [
            {
                "id": x.get("url", f"{query}:{i}"),
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "content": x.get("snippet", ""),
            }
            for i, x in enumerate(relevant)
        ]

    controller = MARAGController(
        generate_fn=generate_fn,
        retrieve_fn=retrieve_fn,
        query_generate_fn=query_generate_fn,
        num_workers=args.num_workers,
        num_rounds=args.num_rounds,
        queries_per_conflict=args.queries_per_conflict,
        docs_per_query=args.docs_per_query,
        entropy_order_mode=args.entropy_order_mode,
    )

    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line).get("sample_id"))

    config = vars(args).copy()
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for idx, item in enumerate(data):
        sample_id = item.get("id", item.get("sample_id", idx))
        if sample_id in completed:
            continue
        question = question_of(item)
        if not question:
            continue
        result = controller.run(question, options_of(item))
        record = {
            "sample_id": sample_id,
            "question": question,
            "gold": item.get("answer", item.get("Answer", item.get("answer_idx"))),
            "entropy_order_mode": args.entropy_order_mode,
            "entropy_scope": args.entropy_scope,
            **result.to_dict(),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{idx + 1}/{len(data)}] id={sample_id} answer={result.final_answer} "
            f"rounds={len(result.rounds)} stop={result.stopped_reason}"
        )


if __name__ == "__main__":
    main()
