# Orchestrator 模块启用说明

默认情况下三个模块都关闭，行为尽量保持原版 Search-o1。

## Baseline（全部关闭）
```bash
python scripts/run_search_o1.py \
  --dataset_name aime --split test \
  --model_path YOUR_MODEL_PATH \
  --search_provider ddgs
```

## + Planner
```bash
python scripts/run_search_o1.py \
  --dataset_name aime --split test \
  --model_path YOUR_MODEL_PATH \
  --search_provider ddgs \
  --enable_planner \
  --planner_pre_retrieve True \
  --max_plan_queries 3 \
  --plan_summary_max_chars 1200
```

## + Critic + Self-Consistency
```bash
python scripts/run_search_o1.py \
  --dataset_name aime --split test \
  --model_path YOUR_MODEL_PATH \
  --search_provider ddgs \
  --enable_critic \
  --critic_mode event \
  --enable_sc \
  --sc_n 3 \
  --sc_vote_mode evidence_constrained
```

## 预算与安全参数
- `--max_injected_chars`: 每次注入外部内容上限
- `--blocklist_patterns`: 可疑指令黑名单（逗号分隔）
- Planner/Critic 的结构化输出均会去除 search tags，防止协议注入。
