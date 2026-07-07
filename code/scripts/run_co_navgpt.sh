#!/usr/bin/env bash
# Co-NavGPT baseline — flat-text LLM frontier assignment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CALLER_BRAIN_BACKEND_SET="${BRAIN_BACKEND+x}"
CALLER_BRAIN_BACKEND="${BRAIN_BACKEND:-}"
CALLER_BRAIN_MODEL_SET="${BRAIN_MODEL+x}"
CALLER_BRAIN_MODEL="${BRAIN_MODEL:-}"
CALLER_BRAIN_BASE_URL_SET="${BRAIN_BASE_URL+x}"
CALLER_BRAIN_BASE_URL="${BRAIN_BASE_URL:-}"
CALLER_DEEPSEEK_THINKING_SET="${DEEPSEEK_THINKING+x}"
CALLER_DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-}"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi
if [[ -n "$CALLER_BRAIN_BACKEND_SET" ]]; then
    BRAIN_BACKEND="$CALLER_BRAIN_BACKEND"
fi
if [[ -n "$CALLER_BRAIN_MODEL_SET" ]]; then
    BRAIN_MODEL="$CALLER_BRAIN_MODEL"
fi
if [[ -n "$CALLER_BRAIN_BASE_URL_SET" ]]; then
    BRAIN_BASE_URL="$CALLER_BRAIN_BASE_URL"
fi
if [[ -n "$CALLER_DEEPSEEK_THINKING_SET" ]]; then
    DEEPSEEK_THINKING="$CALLER_DEEPSEEK_THINKING"
fi

CONDA_ENV=${CONDA_ENV:-mindnav38}
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
    PYTHON_CMD=(python)
fi
REPO=${REPO:-"$REPO_ROOT/code/vendor/conavgpt"}
SPLIT=${SPLIT:-val_mini}
MODEL_PATH=${MODEL_PATH:-${LLM_PATH:-/home/huaziheng/models/Qwen2.5-7B-Instruct}}
SIM_GPU_ID=${SIM_GPU_ID:-0}
SEM_GPU_ID=${SEM_GPU_ID:-0}
LLM_GPU_ID=${LLM_GPU_ID:-$SEM_GPU_ID}
DUMP_LOCATION=${DUMP_LOCATION:-"$REPO_ROOT/results/dump"}
EXP_NAME=${EXP_NAME:-co_navgpt_${SPLIT}}
LOG=${LOG:-"$REPO_ROOT/results/runs/co_navgpt_${SPLIT}.log"}
JSONL=${JSONL:-"$REPO_ROOT/results/runs/co_navgpt_${SPLIT}.jsonl"}
NVIDIA_EGL_DIR=${NVIDIA_EGL_DIR:-/home/huaziheng/nvidia-egl-580.142/driver}
NVIDIA_EGL_VENDOR_JSON=${NVIDIA_EGL_VENDOR_JSON:-/home/huaziheng/nvidia-egl-580.142/nvidia_egl_vendor_580_142.json}

mkdir -p "$(dirname "$LOG")"
mkdir -p "$DUMP_LOCATION"

if [[ -d "$NVIDIA_EGL_DIR" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_EGL_DIR:${LD_LIBRARY_PATH:-}"
fi
if [[ -f "$NVIDIA_EGL_VENDOR_JSON" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
fi
export PYTHONPATH="$REPO/multi-robot-setting:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${BRAIN_BACKEND:-}" ]]; then
    EXTRA_ARGS+=(--brain_backend "$BRAIN_BACKEND")
fi
if [[ -n "${BRAIN_MODEL:-}" ]]; then
    EXTRA_ARGS+=(--brain_model "$BRAIN_MODEL")
fi
if [[ -n "${BRAIN_BASE_URL:-}" ]]; then
    EXTRA_ARGS+=(--brain_base_url "$BRAIN_BASE_URL")
fi
if [[ -n "${DEEPSEEK_THINKING:-}" ]]; then
    EXTRA_ARGS+=(--deepseek_thinking "$DEEPSEEK_THINKING")
fi
if [[ -n "${MAX_EPISODES:-}" ]]; then
    EXTRA_ARGS+=(--max_episodes "$MAX_EPISODES")
fi
if [[ -n "${MAX_EPISODE_LENGTH:-}" ]]; then
    EXTRA_ARGS+=(--max_episode_length "$MAX_EPISODE_LENGTH")
fi
if [[ -n "${BRAIN_MAX_TOKENS:-}" ]]; then
    EXTRA_ARGS+=(--brain_max_tokens "$BRAIN_MAX_TOKENS")
fi

cd "$REPO"
"${PYTHON_CMD[@]}" exp_main_original.py \
    --split "$SPLIT" \
    --llm_path "$MODEL_PATH" \
    --sim_gpu_id "$SIM_GPU_ID" \
    --sem_gpu_id "$SEM_GPU_ID" \
    --llm_gpu_id "$LLM_GPU_ID" \
    --dump_location "$DUMP_LOCATION" \
    --exp_name "$EXP_NAME" \
    --jsonl_log "$JSONL" \
    --method_name co_navgpt \
    "${EXTRA_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$LOG"
