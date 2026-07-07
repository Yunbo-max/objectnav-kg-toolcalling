#!/usr/bin/env bash
# MCoCoNav VLM baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON=${PYTHON:-python}
REPO=${REPO:-"$REPO_ROOT/code/vendor/mcoconav"}
SPLIT=${SPLIT:-val_mini}
SIM_GPU_ID=${SIM_GPU_ID:-0}
SEM_GPU_ID=${SEM_GPU_ID:-0}
BASE_URL=${BASE_URL:-http://127.0.0.1:31511}
DUMP_LOCATION=${DUMP_LOCATION:-"$REPO_ROOT/results/dump"}
EXP_NAME=${EXP_NAME:-mcoconav_${SPLIT}}
METHOD_NAME=${METHOD_NAME:-mcoconav}
TASK_CONFIG=${TASK_CONFIG:-tasks/multi_objectnav_hm3d.yaml}
MAX_EPISODES=${MAX_EPISODES:-}
LOG=${LOG:-"$REPO_ROOT/results/runs/mcoconav_${SPLIT}.log"}
JSONL=${JSONL:-"$REPO_ROOT/results/runs/mcoconav_${SPLIT}.jsonl"}
NVIDIA_EGL_DIR=${NVIDIA_EGL_DIR:-/home/huaziheng/nvidia-egl-580.142/driver}
NVIDIA_EGL_VENDOR_JSON=${NVIDIA_EGL_VENDOR_JSON:-/home/huaziheng/nvidia-egl-580.142/nvidia_egl_vendor_580_142.json}

mkdir -p "$(dirname "$LOG")"
mkdir -p "$DUMP_LOCATION"
mkdir -p "$REPO/img"

if [[ -d "$NVIDIA_EGL_DIR" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_EGL_DIR:${LD_LIBRARY_PATH:-}"
fi
if [[ -f "$NVIDIA_EGL_VENDOR_JSON" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
fi
export PYTHONPATH="$REPO/multi-robot-setting:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

cd "$REPO"
ARGS=(
    --split "$SPLIT" \
    --sim_gpu_id "$SIM_GPU_ID" \
    --sem_gpu_id "$SEM_GPU_ID" \
    --base_url "$BASE_URL" \
    --dump_location "$DUMP_LOCATION" \
    --exp_name "$EXP_NAME" \
    --task_config "$TASK_CONFIG" \
    --jsonl_log "$JSONL" \
    --method_name "$METHOD_NAME" \
)
if [[ -n "$MAX_EPISODES" ]]; then
    ARGS+=(--max_episodes "$MAX_EPISODES")
fi

"$PYTHON" main.py "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
