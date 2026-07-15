#!/usr/bin/env bash
# Launch MindNav on HM3D val_mini with Qwen2.5-7B brain.
#
# Uses the vendored Co-NavGPT perception stack in this repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

CONDA_ENV=${CONDA_ENV:-mindnav38}
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
    PYTHON_CMD=(python)
fi

REPO=${REPO:-"$REPO_ROOT/code/vendor/conavgpt"}
SPLIT=${SPLIT:-val_mini}
MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/models/Qwen2.5-7B-Instruct"}
SIM_GPU_ID=${SIM_GPU_ID:-0}
SEM_GPU_ID=${SEM_GPU_ID:-1}
LLM_GPU_ID=${LLM_GPU_ID:-2}
MAX_EPISODES=${MAX_EPISODES:-0}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTH:-500}
DUMP_LOCATION=${DUMP_LOCATION:-"$REPO_ROOT/results/dump"}
EXP_NAME=${EXP_NAME:-mindnav_${SPLIT}}
LOG=${LOG:-"$REPO_ROOT/results/runs/mindnav_${SPLIT}.log"}

abspath_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

MODEL_PATH="$(abspath_from_root "$MODEL_PATH")"
DUMP_LOCATION="$(abspath_from_root "$DUMP_LOCATION")"
LOG="$(abspath_from_root "$LOG")"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$DUMP_LOCATION"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH="$REPO/multi-robot-setting:${PYTHONPATH:-}"

cd "$REPO"
"${PYTHON_CMD[@]}" exp_main_brain.py \
    --brain helicase \
    --split "$SPLIT" \
    --llm_path "$MODEL_PATH" \
    --sim_gpu_id "$SIM_GPU_ID" \
    --sem_gpu_id "$SEM_GPU_ID" \
    --llm_gpu_id "$LLM_GPU_ID" \
    --max_episodes "$MAX_EPISODES" \
    --max_episode_length "$MAX_EPISODE_LENGTH" \
    --dump_location "$DUMP_LOCATION" \
    --exp_name "$EXP_NAME" \
    "$@" \
    2>&1 | tee "$LOG"
