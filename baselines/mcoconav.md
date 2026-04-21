# MCoCoNav

Upstream: https://github.com/lqn-lab/MCoCoNav

## What it is
VLM extension of Co-NavGPT's perception stack. Same 2D semantic map,
different brain: the VLM reads images (or rendered map snapshots) directly
and emits a frontier assignment.

## Our reproduction
Local path: `/tf/notebooks/godie/repos/MCoCoNav/`

Key substitutions:
  * VLM: Qwen2.5-VL-7B-Instruct (`/tf/notebooks/models/Qwen2.5-VL-7B-Instruct`).
  * OOM patch: `gc.collect() + torch.cuda.empty_cache()` added to the VLM
    server to avoid the ~979-call OOM on Habitat 0.2.1.

## Run

```bash
bash code/scripts/run_mcoconav.sh
```

## Numbers (HM3D val_mini, N=2)
- Qwen2.5-VL-7B: SR 0.000 (VLM spatial reasoning collapse)

This is the central result of the paper's interface-diagnostic table:
with the same perception stack and infrastructure, swapping the brain
from text LLM to VLM of equal parameter count collapses the method.
