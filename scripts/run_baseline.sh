#!/usr/bin/env bash
set -euo pipefail

python scripts/run_search_o1.py \
  --dataset_name "${DATASET_NAME:-aime}" \
  --split "${SPLIT:-test}" \
  --model_path "${MODEL_PATH:-YOUR_MODEL_PATH}" \
  --bing_subscription_key "${BING_KEY:-YOUR_BING_SUBSCRIPTION_KEY}" \
  --jina_api_key "${JINA_KEY:-None}" \
  --max_search_limit "${MAX_SEARCH_LIMIT:-5}" \
  --max_turn "${MAX_TURN:-10}" \
  --top_k "${TOP_K:-10}" \
  --seed "${SEED:-42}"
