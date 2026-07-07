# Co-NavGPT

Upstream: https://github.com/ybgdgh/Co-NavGPT

## What It Is

Co-NavGPT is the flat-text LLM frontier-assignment baseline. It shares the
perception and planning stack with MindNav:

- Detectron2 / RedNet semantic perception
- 2D semantic map
- frontier extraction
- FMM local planner

The brain receives robot positions, detected objects, walls, previous
frontiers, and unexplored frontiers as text, then emits:

```text
robot_0: frontier_i
robot_1: frontier_j
```

It does not use a knowledge graph or KG tool-calling.

## Local Implementation

Local path:

```text
/home/huaziheng/project/objectnav-kg-toolcalling/code/vendor/conavgpt/
```

Entry point:

```text
code/vendor/conavgpt/exp_main_original.py
```

Run script:

```text
code/scripts/run_co_navgpt.sh
```

This local version supports:

- `BRAIN_BACKEND=local` with `/home/huaziheng/models/Qwen2.5-7B-Instruct`
- `BRAIN_BACKEND=deepseek` with the project-root `.env`

## Run

Qwen local flat-text baseline:

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
SPLIT=val MAX_EPISODES=100 \
MODEL_PATH=/home/huaziheng/models/Qwen2.5-7B-Instruct \
BRAIN_BACKEND=local \
EXP_NAME=co_navgpt_qwen_val_100 \
LOG=results/runs/co_navgpt_qwen_val_100.log \
JSONL=results/runs/co_navgpt_qwen_val_100.jsonl \
bash code/scripts/run_co_navgpt.sh --method_name co_navgpt_qwen
```

DeepSeek flat-text baseline:

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
SPLIT=val MAX_EPISODES=100 \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
EXP_NAME=co_navgpt_deepseek_val_100 \
LOG=results/runs/co_navgpt_deepseek_val_100.log \
JSONL=results/runs/co_navgpt_deepseek_val_100.jsonl \
bash code/scripts/run_co_navgpt.sh --method_name co_navgpt_deepseek
```

## Current Legacy Numbers

These are small `val_mini` sanity results, not full-val paper numbers:

| Method | Episodes | Success | SR | SPL |
|---|---:|---:|---:|---:|
| Co-NavGPT, Qwen | 30 | 20 | 0.667 | 0.332 |
| Co-NavGPT, DeepSeek | 30 | 19 | 0.633 | 0.317 |

The DeepSeek flat-text version did not improve Co-NavGPT on `val_mini`. This
supports the current interpretation that the main gain comes from KG
tool-calling structure, not from simply swapping in a stronger LLM.
