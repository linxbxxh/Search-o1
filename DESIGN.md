# Search-o1 研究级三段增强设计（Planner + Critic + Self-Consistency Voting）

## 1) 原始 Search-o1 流程概览与插入点

Search-o1 原始流程在 `scripts/run_search_o1.py` 中实现：
1. 初始化任务提示词和问题，进入 batch generation。  
2. 模型在中间推理中输出 `<|begin_search_query|>...<|end_search_query|>` 时触发检索（Bing）。  
3. 检索结果文档经过 `Reason-in-Documents`（`generate_webpage_to_reasonchain_batch`）提炼成可注入推理链的结构化信息。  
4. 结果通过 `<|begin_search_result|>...<|end_search_result|>` 注入回上下文，继续 interleaved search + reasoning，直到结束。  

**本次增强插入点**：
- **Planner**：插入在主推理循环前（pre-reasoning）。
- **Critic**：插入在主循环完成后的 refine 阶段（可迭代）。
- **Voting**：插入在最终答案输出前（多样本采样 + 聚合）。

## 2) 新增模块接口与 JSON Schema

### 2.1 Planner (`scripts/planner.py`)
输入：`question`、`search_fn`、`context_budget_tokens`。

输出 JSON（严格格式）：
```json
{
  "task_type": "...",
  "subquestions": [
    {"id": "SQ1", "question": "...", "queries": ["...", "..."], "evidence_needed": "..."}
  ],
  "initial_evidence_brief": [
    {"subq_id": "SQ1", "claim": "...", "support": [{"source_id": "...", "snippet": "..."}]}
  ]
}
```

### 2.2 Critic (`scripts/critic.py`)
输入：候选输出文本。

输出 JSON（严格格式）：
```json
{
  "score": {"faithfulness": 0.0, "logic": 0.0, "clarity": 0.0},
  "issues": [
    {"type": "missing_evidence|contradiction|math_error|format", "where": "...", "fix": "..."}
  ],
  "revise_required": true
}
```

### 2.3 Voting (`scripts/voting.py`)
输入：K 个候选答案及证据统计。

输出 JSON（严格格式）：
```json
{
  "candidates": [{"answer_norm": "...", "raw_answer": "...", "evidence_stats": {}}],
  "vote": {"winner": "...", "distribution": {}, "mode": "majority|evidence_weighted"}
}
```

## 3) 触发策略（门控）

- `enable_planner`：默认关闭，显式打开后在主循环前执行。
- `enable_critic`：默认关闭；开启后若检测到不确定性标记或缺失搜索证据则触发。
- `enable_voting`：默认关闭；开启后执行 K 路采样并聚合。

额外阈值门控：`critic_score_threshold`（默认 0.65）。

## 4) 上下文预算与去重策略

- 每阶段注入均限制 `context_budget_tokens`（默认 256）。
- Planner 简报按子问题分配预算，snippet 超长截断。
- Planner 检索结果按 `source_id/url` 去重，避免重复证据。
- Critic/Voting 仅注入紧凑 JSON，不注入长原文。

## 5) 配置项

已添加参数：
- `--enable_planner`
- `--enable_critic`
- `--enable_voting`
- `--k_votes`（范围 [1,15]）
- `--max_refine_iters`（范围 [0,5]）
- `--context_budget_tokens`
- `--vote_mode` (`majority` / `evidence_weighted`)
- `--critic_score_threshold`
- `--seed`

## 6) 可复现实验与消融

提供：
- `scripts/run_baseline.sh`
- `scripts/run_ablation.sh`

并将阶段日志写入 `outputs/.../*.stage_logs.jsonl`，包含 planner/critic/voting 的结构化产物，便于统计 token、检索次数、延迟和错误分析。

## 7) 严格停止条件

- Critic 迭代：`<= max_refine_iters`。
- 原始 Search-o1 检索轮次：沿用 `MAX_SEARCH_LIMIT` 与 `MAX_TURN`。
- 达上限时返回当前最佳答案并保留 critic/voting 结构化记录。
