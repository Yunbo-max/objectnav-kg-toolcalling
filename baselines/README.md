# Baselines

All methods use the same perception stack (Detectron2 + RedNet + FMM) from
Co-NavGPT. Only the frontier-assignment brain differs.

| Baseline     | Brain type    | Local path |
|--------------|---------------|------------|
| Co-NavGPT    | Text LLM      | `/tf/notebooks/Co-NavGPT/` |
| MCoCoNav     | VLM           | `/tf/notebooks/godie/repos/MCoCoNav/` |
| EfficientNav | VLM           | `/tf/notebooks/godie/repos/EfficientNav/` |

For per-baseline launch recipes see:

- [`co_navgpt.md`](co_navgpt.md)
- [`mcoconav.md`](mcoconav.md)
- [`efficientnav.md`](efficientnav.md)
