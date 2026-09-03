# MA-RAG faithful port for TRACE-o1

This branch adds an **isolated vanilla MA-RAG baseline**. Existing Search-o1 / TRACE-o1 files are intentionally left unchanged.

## Why this branch exists

Before implementing stage-aligned conflict resolution, we need a clean answer to two questions:

1. Does MA-RAG's conflict-to-consensus loop help on the TRACE-o1 benchmark suite?
2. How much of the gain comes from conflict-driven retrieval itself, before any TRACE-specific modification?

The implementation follows the public `NJU-RL/MA-RAG` entropy-version control flow:

1. Generate `N` independent answers in a round.
2. If all normalized final answers agree, stop.
3. Otherwise ask an LLM to summarize disagreements and emit 1-4 retrieval queries.
4. Retrieve evidence for those conflict-derived queries.
5. Compute mean token entropy for each previous answer from top-k log-probabilities.
6. Rank previous answers according to the selected entropy-order mode and feed them, together with retrieved evidence, into the next round.
7. Repeat until consensus or the round budget is exhausted.
8. On budget exhaustion, return the final-round majority answer.

## Entropy parity

The standalone vLLM runner now requests `logprobs=20` by default (`--entropy_top_k 20`) and computes per-token entropy from the returned top-logprob distribution:

```text
p_i = exp(logp_i) / sum_j exp(logp_j)
H_t = -sum_i p_i log p_i
H(candidate) = mean_t H_t
```

This mirrors the upstream MA-RAG implementation, which applies `scipy.stats.entropy` to exponentiated top-logprobs.

For reasoning models that emit `<think> ... </think>`, the default `--entropy_scope post_think` uses only tokens after the final `</think>` when that marker can be located in the generated token sequence. If no post-think tokens are available, it safely falls back to all generated tokens. `--entropy_scope all` is also available as an ablation.

Each candidate stores:

- selected `mean_token_entropy`;
- all-token mean entropy;
- post-`</think>` mean entropy;
- number of entropy-bearing tokens;
- per-token entropy values.

The controller's legacy field name is `confidence`, but for the entropy variant it stores **mean token entropy (an uncertainty score)**, so lower values mean higher confidence.

### Important upstream ordering detail

The released `ma_rag_entropy.py` computes candidate mean entropies and then uses:

```python
np.argsort(previous_answer_entropies)[::-1]
```

Therefore the released code feeds previous answers in **high-entropy -> low-entropy** order. This differs from the semantic statement in its prompt that lower entropy means higher confidence.

To avoid silently changing the baseline, this port makes the behavior explicit:

- `--entropy_order_mode official_code` (default): reproduce released code, high entropy first;
- `--entropy_order_mode low_first`: low entropy / high confidence first;
- `--entropy_order_mode none`: preserve generation order.

The selected order and ranked candidate indices are written into every round record, making the ordering ablation auditable without rerunning analysis code.

## Files in this isolated branch

- `scripts/ma_rag_port.py`: reusable conflict-to-consensus controller.
- `scripts/run_ma_rag_port.py`: standalone vLLM + existing web-search runner with token-entropy extraction.
- `tests/test_ma_rag_port.py`: deterministic controller and entropy-order tests.
- `MA_RAG_PORT.md`: this note.

No pre-existing Search-o1 / TRACE-o1 source file is modified in this branch.

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
  --entropy_top_k 20 \
  --entropy_scope post_think \
  --entropy_order_mode official_code \
  --search_provider ddgs
```

The dataset may use either `Question`/`Answer` or `question`/`answer`; option dictionaries/lists are supported.

## Logging

Each JSONL result stores:

- all candidate texts and normalized predictions for every round;
- candidate token-entropy metadata;
- entropy ordering mode and ranked candidate indices;
- disagreement score `1 - majority_count / N`;
- whether the round reached unanimity;
- conflict-derived queries;
- retrieved documents;
- final answer and stop reason (`consensus` or `round_budget`).

`run_config.json` stores the exact entropy configuration alongside the experiment output.

These fields are sufficient for later analysis of both conflict type and uncertainty dynamics, including whether disagreement and entropy decrease across conflict-resolution rounds.

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
- Main reference implementation inspected: `ma_rag_entropy.py` and `utils.py`.
