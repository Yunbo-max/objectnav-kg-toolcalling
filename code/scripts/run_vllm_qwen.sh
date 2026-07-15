#!/usr/bin/env bash
# Serve the local Qwen2.5-3B model through vLLM's OpenAI-compatible API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV=${CONDA_ENV:-vllm}
GPU_ID=${GPU_ID:-2}
MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/models/Qwen2.5-3B-Instruct"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2.5-3b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
API_KEY=${API_KEY:-local-vllm}

export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec conda run --no-capture-output -n "$CONDA_ENV" \
    vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --api-key "$API_KEY" \
    "$@"
