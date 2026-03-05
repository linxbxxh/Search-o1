#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --dataset_name "${DATASET_NAME:-aime}"
  --split "${SPLIT:-test}"
  --model_path "${MODEL_PATH:-YOUR_MODEL_PATH}"
  --bing_subscription_key "${BING_KEY:-YOUR_BING_SUBSCRIPTION_KEY}"
  --jina_api_key "${JINA_KEY:-None}"
  --max_search_limit "${MAX_SEARCH_LIMIT:-5}"
  --max_turn "${MAX_TURN:-10}"
  --top_k "${TOP_K:-10}"
  --seed "${SEED:-42}"
)

python scripts/run_search_o1.py "${COMMON_ARGS[@]}"
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_planner
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_critic
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_voting --k_votes "${K_VOTES:-5}"
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_planner --enable_critic
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_planner --enable_voting --k_votes "${K_VOTES:-5}"
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_critic --enable_voting --k_votes "${K_VOTES:-5}"
python scripts/run_search_o1.py "${COMMON_ARGS[@]}" --enable_planner --enable_critic --enable_voting --k_votes "${K_VOTES:-5}" --vote_mode "${VOTE_MODE:-evidence_weighted}"
