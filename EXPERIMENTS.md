# 实验与消融运行说明

## 运行
- Baseline: `bash scripts/run_baseline.sh`
- Ablation: `bash scripts/run_ablation.sh`

## 建议日志目录结构

```
outputs/
  runs.baselines/
    <dataset>.<model>.search_o1/
      test.stage_logs.jsonl
      test.<timestamp>.info_extract.json
      test.<timestamp>.json
```

`*.stage_logs.jsonl` 每行一个 JSON，记录 planner/critic/voting 阶段输入输出摘要，可直接用于画图与误差分析。
