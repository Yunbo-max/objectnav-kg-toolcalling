# Results

| Path | Description |
|---|---|
| `runs/` | Per-run JSONL and tee logs |
| `dump/logs/` | Habitat logging output per experiment |
| `table1.csv` | Legacy aggregated table |
| `mindnav_deepseek_vs_local_kg.csv` | `val_mini` episode-level comparison |
| `co_navgpt_deepseek_vs_qwen.csv` | `val_mini` episode-level comparison |

## Current Legacy Val Mini Summary

These runs are useful for quick sanity checks, but they are not full HM3D val:

| Method | Episodes | Success | SR | SPL |
|---|---:|---:|---:|---:|
| Co-NavGPT, Qwen | 30 | 20 | 0.667 | 0.332 |
| Co-NavGPT, DeepSeek | 30 | 19 | 0.633 | 0.317 |
| incomplete MindNav | 30 | 18 | 0.600 | 0.334 |
| KG MindNav, local LLM | 30 | 22 | 0.733 | 0.344 |
| KG MindNav, DeepSeek | 30 | 23 | 0.767 | 0.379 |

## Full Val

Full HM3D ObjectNav `val` has 1000 episodes. New paper-ready runs should be
stored with explicit names such as:

```text
results/runs/mindnav_kg_deepseek_val_100.jsonl
results/runs/mindnav_kg_deepseek_val_full.jsonl
```

Compute metrics from any JSONL:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/runs/mindnav_kg_deepseek_val_100.jsonl")
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
sr = sum(r["success"] for r in rows) / len(rows)
spl = sum(r["spl"] for r in rows) / len(rows)
print(f"n={len(rows)} SR={sr:.3f} SPL={spl:.3f}")
PY
```
