# Running the paper experiments

## Table 1 — main comparison, HM3D val_mini

```bash
# MindNav (ours)
bash code/scripts/run_mindnav.sh

# Baselines
bash code/scripts/run_co_navgpt.sh
bash code/scripts/run_mcoconav.sh
# EfficientNav queued — see baselines/efficientnav.md

# Aggregate into Table 1
python code/scripts/aggregate_results.py
```

All logs land in `results/runs/`. The aggregator writes
`results/table1.csv`.

## Ablations

### KG edge types
Comment out `connected_to` in `code/configs/mindnav.yaml`:
```yaml
kg:
  disable_edges: [connected_to]
```
Rerun `run_mindnav.sh`. Expect SR ~0.667 (room-connectivity-blind KG
reasoning).

### Certainty aggregation
Swap noisy-OR for max-confidence:
```yaml
kg:
  aggregation: max    # default: noisy_or
```
Expect small SR drops on categories with multi-detection objects
(`plant`, `tv_monitor`).
