# Co-NavGPT

Upstream: https://github.com/yuyang-J/Co-NavGPT

## What it is
Original flat-text LLM frontier-assignment baseline. Perception stack
(Detectron2 + RedNet + 2D semantic map + FMM planner) is shared with
MindNav — only the brain differs.

## Our reproduction
Local path: `/tf/notebooks/Co-NavGPT/`

Key substitutions:
  * Upstream uses GPT-4 via OpenAI API; we swap in Qwen2.5-7B-Instruct for
    matched-scale comparison with MindNav.
  * `exp_main_original_minimax.py` uses MiniMax-M2.5 via SiliconFlow API;
    results are within noise of Qwen2.5-7B-Instruct local.

## Run

```bash
bash code/scripts/run_co_navgpt.sh
```

This invokes `/tf/notebooks/Co-NavGPT/exp_main_original.py` with our
config. Logs land in `results/runs/co_navgpt_val_mini.log`.

## Numbers (HM3D val_mini, N=2)
- Co-NavGPT (random frontier): SR 0.760
- Co-NavGPT (Qwen2.5-7B flat text): SR 0.700, SPL 0.306
- Co-NavGPT (MiniMax-M2.5): SR ≈ 0.700, SPL ≈ 0.306

## Notes
- val_mini has 30 episodes but runs 1000 iterations (2 agents, repeated scenes).
- Reported 24% false-positive rate on Detectron2+RedNet detections is
  inherited by every method on this stack.
