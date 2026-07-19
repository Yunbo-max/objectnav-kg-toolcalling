# Results

| Path | Description |
|---|---|
| `runs/` | Per-episode JSONL、mapping 与运行日志（gitignored） |
| `aggregated/` | HM3D `val_60` 六个主实验/消融的汇总、paired CI 与审计 |
| `aggregated_robot_count/` | 1/2/3 robots 消融汇总；2-agent 复用 `M-DS-TEXT` |
| `aggregated_kg_formats_90/` | 按 text / JSON / triples 合并 `val_mini` 30 与 `val_60` 60 的 N=90 汇总 |

KG N=90 汇总说明见 `docs/VALMINI_VAL60_KG_90_SUMMARY.md`；可由 `code/scripts/aggregate_kg_formats_90.py` 从原始 JSONL 重建。
