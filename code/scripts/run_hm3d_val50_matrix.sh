#!/usr/bin/env bash
# Sequential HM3D val full-split 50-episode matrix:
# MindNav Qwen, MindNav DeepSeek, Co-NavGPT Qwen, Co-NavGPT DeepSeek.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TS="${1:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$REPO_ROOT/results/runs"
SUMMARY="$RUN_DIR/hm3d_val50_matrix_${TS}_summary.csv"

SIM_GPU_ID="${SIM_GPU_ID:-0}"
SEM_GPU_ID="${SEM_GPU_ID:-2}"
LLM_GPU_ID="${LLM_GPU_ID:-3}"
MODEL_PATH="${MODEL_PATH:-/home/huaziheng/models/Qwen2.5-7B-Instruct}"
MINDNAV_CONFIG="${MINDNAV_CONFIG:-$REPO_ROOT/code/configs/mindnav.yaml}"
DEEPSEEK_MODEL_NAME="${DEEPSEEK_MODEL_NAME:-deepseek-v4-flash}"

mkdir -p "$RUN_DIR"
printf 'method,run_name,jsonl,log,episodes,successes,sr,spl,status,started_at,ended_at\n' > "$SUMMARY"

summarize_jsonl() {
    local method="$1"
    local run_name="$2"
    local jsonl="$3"
    local log="$4"
    local status="$5"
    local started_at="$6"
    local ended_at="$7"
    python - "$method" "$run_name" "$jsonl" "$log" "$status" "$started_at" "$ended_at" "$SUMMARY" <<'PY'
import csv
import json
import pathlib
import sys

method, run_name, jsonl, log, status, started_at, ended_at, summary = sys.argv[1:]
path = pathlib.Path(jsonl)
rows = []
if path.exists():
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
n = len(rows)
successes = sum(float(r.get("success", 0.0)) for r in rows)
sr = successes / n if n else 0.0
spl = sum(float(r.get("spl", 0.0)) for r in rows) / n if n else 0.0
with open(summary, "a", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([method, run_name, jsonl, log, n, successes, f"{sr:.6f}", f"{spl:.6f}", status, started_at, ended_at])
print(f"[summary] {method}: n={n} success={successes:g} SR={sr:.3f} SPL={spl:.3f} status={status}", flush=True)
PY
}

run_eval() {
    local method="$1"
    local run_name="$2"
    shift 2
    local jsonl="$RUN_DIR/${run_name}.jsonl"
    local log="$RUN_DIR/${run_name}.log"
    local started_at
    local ended_at
    local status
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "================================================================"
    echo "[start] $method -> $run_name at $started_at"
    echo "[paths] jsonl=$jsonl log=$log"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

    if "$@"; then
        status="done"
    else
        status="failed"
    fi

    ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    summarize_jsonl "$method" "$run_name" "$jsonl" "$log" "$status" "$started_at" "$ended_at"
    echo "[end] $method -> $run_name at $ended_at status=$status"
}

run_eval "mindnav_kg_qwen" "mindnav_kg_qwen_hm3d_val50_${TS}" \
    env SPLIT=val MAX_EPISODES=50 \
        SIM_GPU_ID="$SIM_GPU_ID" SEM_GPU_ID="$SEM_GPU_ID" LLM_GPU_ID="$LLM_GPU_ID" \
        MODEL_PATH="$MODEL_PATH" BRAIN_BACKEND=local MINDNAV_CONFIG="$MINDNAV_CONFIG" \
        EXP_NAME="mindnav_kg_qwen_hm3d_val50_${TS}" \
        LOG="$RUN_DIR/mindnav_kg_qwen_hm3d_val50_${TS}.log" \
        JSONL="$RUN_DIR/mindnav_kg_qwen_hm3d_val50_${TS}.jsonl" \
        bash "$SCRIPT_DIR/run_mindnav.sh" --method_name mindnav_kg_qwen

run_eval "mindnav_kg_deepseek" "mindnav_kg_deepseek_hm3d_val50_${TS}" \
    env SPLIT=val MAX_EPISODES=50 \
        SIM_GPU_ID="$SIM_GPU_ID" SEM_GPU_ID="$SEM_GPU_ID" \
        BRAIN_BACKEND=deepseek BRAIN_MODEL="$DEEPSEEK_MODEL_NAME" DEEPSEEK_THINKING=disabled \
        MINDNAV_CONFIG="$MINDNAV_CONFIG" \
        EXP_NAME="mindnav_kg_deepseek_hm3d_val50_${TS}" \
        LOG="$RUN_DIR/mindnav_kg_deepseek_hm3d_val50_${TS}.log" \
        JSONL="$RUN_DIR/mindnav_kg_deepseek_hm3d_val50_${TS}.jsonl" \
        bash "$SCRIPT_DIR/run_mindnav.sh" --method_name mindnav_kg_deepseek

run_eval "co_navgpt_qwen" "co_navgpt_qwen_hm3d_val50_${TS}" \
    env SPLIT=val MAX_EPISODES=50 \
        SIM_GPU_ID="$SIM_GPU_ID" SEM_GPU_ID="$SEM_GPU_ID" LLM_GPU_ID="$LLM_GPU_ID" \
        MODEL_PATH="$MODEL_PATH" BRAIN_BACKEND=local \
        EXP_NAME="co_navgpt_qwen_hm3d_val50_${TS}" \
        LOG="$RUN_DIR/co_navgpt_qwen_hm3d_val50_${TS}.log" \
        JSONL="$RUN_DIR/co_navgpt_qwen_hm3d_val50_${TS}.jsonl" \
        bash "$SCRIPT_DIR/run_co_navgpt.sh" --method_name co_navgpt_qwen

run_eval "co_navgpt_deepseek" "co_navgpt_deepseek_hm3d_val50_${TS}" \
    env SPLIT=val MAX_EPISODES=50 \
        SIM_GPU_ID="$SIM_GPU_ID" SEM_GPU_ID="$SEM_GPU_ID" \
        BRAIN_BACKEND=deepseek BRAIN_MODEL="$DEEPSEEK_MODEL_NAME" DEEPSEEK_THINKING=disabled \
        EXP_NAME="co_navgpt_deepseek_hm3d_val50_${TS}" \
        LOG="$RUN_DIR/co_navgpt_deepseek_hm3d_val50_${TS}.log" \
        JSONL="$RUN_DIR/co_navgpt_deepseek_hm3d_val50_${TS}.jsonl" \
        bash "$SCRIPT_DIR/run_co_navgpt.sh" --method_name co_navgpt_deepseek

echo "================================================================"
echo "[complete] Summary: $SUMMARY"
cat "$SUMMARY"
