# MA-RAG faithful port for TRACE-o1

This branch adds an **isolated vanilla MA-RAG baseline**. Existing Search-o1 / TRACE-o1 files are intentionally left unchanged.

## Why this branch exists

Before implementing stage-aligned conflict resolution, we need a clean answer to two questions:

1. Does MA-RAG's conflict-to-consensus loop help on the TRACE-o1 benchmark suite?
2. How much of the gain comes from conflict-driven retrieval itself, before any TRACE-specific modification?

The implementation follows the public `NJU-RL/MA-RAG` entropy-version control flow at the algorithmic level:

1. Generate `N` independent answers in a round.
2. If all normalized final answers agree, stop.
3. Otherwise ask an LLM to summarize disagreements and emit 1-4 retrieval queries.
4. Retrieve evidence for those conflict-derived queries.
5. Feed retrieved evidence plus previous candidate answers into the next round.
6. Repeat until consensus or the round budget is exhausted.
7. On budget exhaustion, return the final-round majority answer.

The upstream entropy implementation additionally ranks previous answers using mean token entropy. The first TRACE port keeps a confidence callback in the controller but the standalone vLLM runner does **not yet claim entropy parity**, because Search-o1's current vLLM path does not expose the same per-token entropy structure used by MA-RAG's OpenAI-compatible service. This is logged as a known parity gap rather than silently approximated.

## New files only

- `scripts/ma_rag_port.py`: reusable conflict-to-consensus controller.
- `scripts/run_ma_rag_port.py`: standalone vLLM + existing web-search runner.
- `tests/test_ma_rag_port.py`: deterministic controller tests.
- `MA_RAG_PORT.md`: this note.

No pre-existing source file is modified in this branch.

## Example

```bash
python scripts/run_ma_rag_port.py \
  --dataset_path /path/to/gpqa.json \
  --output_dir outputs/ma_rag/gpqa \
  --model_path /path/to/Qwen3-4B-Instruct-2507 \
  --num_workers 5 \
  --num_rounds 5 \
  --queries_per_conflict 4 \
  --docs_per_query 2 \
  --search_provider ddgs
```

The dataset may use either `Question`/`Answer` or `question`/`answer`; option dictionaries/lists are supported.

## Logging

Each JSONL result stores:

- all candidate texts and normalized predictions for every round;
- disagreement score `1 - majority_count / N`;
- whether the round reached unanimity;
- conflict-derived queries;
- retrieved documents;
- final answer and stop reason (`consensus` or `round_budget`).

These fields are intentionally sufficient for the next research step: classify whether unresolved conflicts are evidence, reasoning, or residual-answer conflicts without rerunning the baseline.

## Deliberately not included yet

The following are **not** part of this vanilla port:

- TRACE Planner or Critic integration;
- stage-specific conflict classification;
- Search vs Revise vs Sample routing;
- adaptive K;
- DAS;
- learned router / RL;
- MedCorp + MedCPT retrieval service parity.

Those should be introduced as separate commits/ablations only after the vanilla MA-RAG baseline is measured.

## Upstream reference

- Paper/code: `NJU-RL/MA-RAG`, *From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG* (ICML 2026).
- Main reference implementation inspected: `ma_rag_entropy.py`.
