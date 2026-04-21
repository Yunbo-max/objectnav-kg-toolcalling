#!/usr/bin/env bash
# Launch MindNav on HM3D val_mini with Qwen2.5-7B brain.
#
# Requires: /tf/notebooks/Co-NavGPT/ as the upstream perception stack.
# We invoke upstream's exp_main_brain.py with our MindNav brain overlay.
set -euo pipefail

REPO=${REPO:-/tf/notebooks/Co-NavGPT}
SPLIT=${SPLIT:-val_mini}
LOG=${LOG:-../results/runs/mindnav_${SPLIT}.log}
mkdir -p "$(dirname "$LOG")"

cd "$REPO"
python exp_main_brain.py \
    --brain helicase \
    --split "$SPLIT" \
    --llm_path /tf/notebooks/models/Qwen2.5-7B-Instruct \
    2>&1 | tee "$LOG"
