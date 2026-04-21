#!/usr/bin/env bash
# MCoCoNav VLM baseline.
set -euo pipefail

REPO=${REPO:-/tf/notebooks/godie/repos/MCoCoNav}
SPLIT=${SPLIT:-val_mini}
LOG=${LOG:-../results/runs/mcoconav_${SPLIT}.log}
mkdir -p "$(dirname "$LOG")"

cd "$REPO"
python -m main.run \
    --split "$SPLIT" \
    --vlm /tf/notebooks/models/Qwen2.5-VL-7B-Instruct \
    2>&1 | tee "$LOG"
