# EfficientNav

Upstream: (NeurIPS 2025, code release pending)

## What it is
VLM-head variant designed for tighter action budgets; originally used
LLaVA-34B. We adapted the decision head to Qwen2.5-VL-7B for a
matched-scale comparison.

## Our reproduction
Local path: `/tf/notebooks/godie/repos/EfficientNav/`

## Status
Data set up; runs queued pending GPU availability alongside the other
baselines.

## Numbers (HM3D val_mini, N=2)
- Qwen2.5-VL-7B: SR 0.000 (same collapse as MCoCoNav).
