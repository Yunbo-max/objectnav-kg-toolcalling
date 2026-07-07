# ObjectNav KG Tool-Calling

This repository is configured for the local workspace:

```bash
PROJECT=/home/huaziheng/project/objectnav-kg-toolcalling
CONAVGPT=$PROJECT/code/vendor/conavgpt
```

Python and Habitat commands should use the `mindnav38` conda environment.

## What Is In This Repo

- `code/vendor/conavgpt/`: local Co-NavGPT vendor tree, Habitat entry points, MindNav implementation, and run scripts.
- `code/scripts/`: reproducibility checks and experiment launch wrappers.
- `data/README.md`: local HM3D v0.2 layout and dataset verification.
- `docs/INSTALL.md`: environment, models, DeepSeek API, and HM3D setup.
- `docs/RUN_EXPERIMENTS.md`: smoke tests, full-val runs, and baseline commands.
- `results/README.md`: current result files and aggregation notes.

## Local Assets

Current local paths:

```text
/home/huaziheng/models/Qwen2.5-7B-Instruct
/home/huaziheng/models/Qwen2.5-VL-7B-Instruct
code/vendor/conavgpt/RedNet/model/rednet_semmap_mp3d_40.pth
code/vendor/conavgpt/data/datasets/objectnav_hm3d_v2/
code/vendor/conavgpt/data/hm3d-v0.2-full/
code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2
```

`hm3d_v0.2` is a symlink to the downloaded full HM3D v0.2 val assets.

## Quick Check

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
conda run -n mindnav38 python code/scripts/check_repro_env.py \
  --model-path /home/huaziheng/models/Qwen2.5-7B-Instruct
```

Before launching Habitat evaluations, check GPU availability and keep the
default separation: one GPU for Habitat simulation, one for semantic perception,
and one for local LLM/VLM if using local models.

## Smoke Test

Run one full HM3D `val` episode with KG tool-calling MindNav and DeepSeek:

```bash
SPLIT=val MAX_EPISODES=1 MAX_EPISODE_LENGTH=60 \
EXP_NAME=mindnav_kg_deepseek_val_smoke \
LOG=results/runs/mindnav_kg_deepseek_val_smoke.log \
JSONL=results/runs/mindnav_kg_deepseek_val_smoke.jsonl \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
bash code/scripts/run_mindnav.sh --method_name mindnav_kg_deepseek_val_smoke
```

DeepSeek API settings are loaded from the project-root `.env` file. The file is
gitignored and should stay local.

## Full Evaluation

The full HM3D ObjectNav `val` split has 1000 episodes. Use:

```bash
SPLIT=val MAX_EPISODES=0 \
EXP_NAME=mindnav_kg_deepseek_val_full \
LOG=results/runs/mindnav_kg_deepseek_val_full.log \
JSONL=results/runs/mindnav_kg_deepseek_val_full.jsonl \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
bash code/scripts/run_mindnav.sh --method_name mindnav_kg_deepseek
```

For details and baseline commands, see `docs/RUN_EXPERIMENTS.md`.
