# Baselines

All methods use the same Habitat ObjectNav environment and the same perception
stack where applicable: Detectron2 + RedNet + 2D semantic map + FMM planner.
The main difference is the frontier-assignment brain.

Current local workspace:

```text
/home/huaziheng/project/objectnav-kg-toolcalling
```

| Baseline | Brain type | Local path |
|---|---|---|
| Co-NavGPT | Text LLM, flat prompt | `code/vendor/conavgpt/` |
| MCoCoNav | VLM | `code/vendor/mcoconav/` |
| EfficientNav | VLM | Not vendored in this checkout |

For per-baseline launch recipes see:

- [`co_navgpt.md`](co_navgpt.md)
- [`mcoconav.md`](mcoconav.md)
- [`efficientnav.md`](efficientnav.md)

Legacy `val_mini` runs are stored under `results/runs/`. Full-paper evaluation
should use HM3D ObjectNav `SPLIT=val`.
