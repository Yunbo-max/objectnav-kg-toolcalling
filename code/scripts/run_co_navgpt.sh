#!/usr/bin/env bash
# Co-NavGPT baseline — flat-text LLM frontier assignment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV=${CONDA_ENV:-mindnav38}
REPO=${REPO:-"$REPO_ROOT/code/vendor/conavgpt"}
SPLIT=${SPLIT:-val}
MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/models/Qwen2.5-7B-Instruct"}
SIM_GPU_ID=${SIM_GPU_ID:-0}
SEM_GPU_ID=${SEM_GPU_ID:-1}
MAX_EPISODES=${MAX_EPISODES:-15}
START_EPISODE_INDEX=${START_EPISODE_INDEX:-0}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTH:-500}
DUMP_LOCATION=${DUMP_LOCATION:-"$REPO_ROOT/results/dump"}
EXP_NAME=${EXP_NAME:-co_navgpt_${SPLIT}_first${MAX_EPISODES}}
LOG=${LOG:-"$REPO_ROOT/results/runs/${EXP_NAME}.log"}
JSONL_LOG=${JSONL_LOG:-"${LOG%.log}.jsonl"}
METHOD_NAME=${METHOD_NAME:-co_navgpt_qwen25_7b}

mkdir -p "$(dirname "$LOG")" "$(dirname "$JSONL_LOG")" "$DUMP_LOCATION"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH="$REPO/multi-robot-setting:${PYTHONPATH:-}"

cd "$REPO"
conda run --no-capture-output -n "$CONDA_ENV" python exp_main_original.py \
    --split "$SPLIT" \
    --llm_path "$MODEL_PATH" \
    --sim_gpu_id "$SIM_GPU_ID" \
    --sem_gpu_id "$SEM_GPU_ID" \
    --max_episodes "$MAX_EPISODES" \
    --start_episode_index "$START_EPISODE_INDEX" \
    --max_episode_length "$MAX_EPISODE_LENGTH" \
    --dump_location "$DUMP_LOCATION" \
    --exp_name "$EXP_NAME" \
    --jsonl_log "$JSONL_LOG" \
    --method_name "$METHOD_NAME" \
    "$@" \
    2>&1 | tee "$LOG"
