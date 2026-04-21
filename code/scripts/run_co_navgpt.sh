#!/usr/bin/env bash
# Co-NavGPT baseline — flat-text LLM frontier assignment.
set -euo pipefail

REPO=${REPO:-/tf/notebooks/Co-NavGPT}
SPLIT=${SPLIT:-val_mini}
LOG=${LOG:-../results/runs/co_navgpt_${SPLIT}.log}
mkdir -p "$(dirname "$LOG")"

cd "$REPO"
python exp_main_original.py \
    --split "$SPLIT" \
    --llm_path /tf/notebooks/models/Qwen2.5-7B-Instruct \
    2>&1 | tee "$LOG"
