"""Standalone runner for the faithful MA-RAG port.

No existing Search-o1 / TRACE-o1 file is imported or modified except reusable search
helpers. This is intentionally a baseline runner; stage-aligned conflict routing will
be implemented separately after the vanilla behavior is measured.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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

    def llm_text(prompt: str, n: int, temperature: float, max_tokens: int, seed: int) -> List[str]:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        outputs = llm.generate(
            [chat],
            sampling_params=SamplingParams(
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=args.top_p,
                top_k=args.top_k_sampling,
                seed=seed,
                stop=[tokenizer.eos_token],
                include_stop_str_in_output=True,
            ),
        )[0].outputs
        return [o.text for o in outputs]

    def generate_fn(prompt: str, n: int, round_id: int):
        texts = llm_text(prompt, n=n, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed + round_id)
        return [{"text": t} for t in texts]

    def query_generate_fn(question: str, options: str, answers: List[str]):
        prompt = MARAGController.build_conflict_query_prompt(question, options, answers)
        return llm_text(prompt, n=1, temperature=0.0, max_tokens=1024, seed=args.seed)[0]

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
    )

    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line).get("sample_id"))

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
            **result.to_dict(),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx + 1}/{len(data)}] id={sample_id} answer={result.final_answer} rounds={len(result.rounds)} stop={result.stopped_reason}")


if __name__ == "__main__":
    main()
