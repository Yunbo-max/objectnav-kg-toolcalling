#!/usr/bin/env bash
# Launch MindNav on HM3D val_mini with Qwen2.5-7B brain.
#
# By default this uses the vendored Co-NavGPT subset in this repository.
# Override REPO to point at a full upstream checkout if needed.
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
EXP_NAME=${EXP_NAME:-mindnav_${SPLIT}}
LOG=${LOG:-"$REPO_ROOT/results/runs/mindnav_${SPLIT}.log"}
JSONL=${JSONL:-"$REPO_ROOT/results/runs/mindnav_${SPLIT}.jsonl"}
NVIDIA_EGL_DIR=${NVIDIA_EGL_DIR:-/home/huaziheng/nvidia-egl-580.142/driver}
NVIDIA_EGL_VENDOR_JSON=${NVIDIA_EGL_VENDOR_JSON:-/home/huaziheng/nvidia-egl-580.142/nvidia_egl_vendor_580_142.json}

abspath_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

LOG="$(abspath_from_root "$LOG")"
JSONL="$(abspath_from_root "$JSONL")"
DUMP_LOCATION="$(abspath_from_root "$DUMP_LOCATION")"
if [[ -n "${KG_TRACE_DIR:-}" ]]; then
    KG_TRACE_DIR="$(abspath_from_root "$KG_TRACE_DIR")"
fi

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
if [[ -n "${MINDNAV_CONFIG:-}" ]]; then
    EXTRA_ARGS+=(--mindnav_config "$MINDNAV_CONFIG")
fi
if [[ -n "${MINDNAV_MODE:-}" ]]; then
    EXTRA_ARGS+=(--mindnav_mode "$MINDNAV_MODE")
fi
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
if [[ -n "${USE_GTSEM:-}" ]]; then
    EXTRA_ARGS+=(--use_gtsem "$USE_GTSEM")
fi
if [[ -n "${MP3D_CONTEXT_SEMANTICS:-}" ]]; then
    EXTRA_ARGS+=(--mp3d_context_semantics "$MP3D_CONTEXT_SEMANTICS")
fi
if [[ -n "${BRAIN_MAX_TOKENS:-}" ]]; then
    EXTRA_ARGS+=(--brain_max_tokens "$BRAIN_MAX_TOKENS")
fi
if [[ -n "${MINDNAV_TARGET_TAU:-}" ]]; then
    EXTRA_ARGS+=(--mindnav_target_tau "$MINDNAV_TARGET_TAU")
fi
if [[ -n "${TARGET_STOP_MODE:-}" ]]; then
    EXTRA_ARGS+=(--target_stop_mode "$TARGET_STOP_MODE")
fi
if [[ -n "${TARGET_MAP_SCORE_THR:-}" ]]; then
    EXTRA_ARGS+=(--target_map_score_thr "$TARGET_MAP_SCORE_THR")
fi
if [[ -n "${TARGET_MAP_MIN_AREA:-}" ]]; then
    EXTRA_ARGS+=(--target_map_min_area "$TARGET_MAP_MIN_AREA")
fi
if [[ -n "${TARGET_MAP_MIN_MASS:-}" ]]; then
    EXTRA_ARGS+=(--target_map_min_mass "$TARGET_MAP_MIN_MASS")
fi
if [[ -n "${TARGET_CONFIRM_HITS:-}" ]]; then
    EXTRA_ARGS+=(--target_confirm_hits "$TARGET_CONFIRM_HITS")
fi
if [[ -n "${TARGET_CONFIRM_WINDOW:-}" ]]; then
    EXTRA_ARGS+=(--target_confirm_window "$TARGET_CONFIRM_WINDOW")
fi
if [[ -n "${TARGET_CONFIRM_STALE_STEPS:-}" ]]; then
    EXTRA_ARGS+=(--target_confirm_stale_steps "$TARGET_CONFIRM_STALE_STEPS")
fi
if [[ -n "${TARGET_FRESH_MIN_AREA:-}" ]]; then
    EXTRA_ARGS+=(--target_fresh_min_area "$TARGET_FRESH_MIN_AREA")
fi
if [[ -n "${TARGET_FRESH_MIN_MASS:-}" ]]; then
    EXTRA_ARGS+=(--target_fresh_min_mass "$TARGET_FRESH_MIN_MASS")
fi
if [[ -n "${TARGET_FRESH_MIN_OVERLAP:-}" ]]; then
    EXTRA_ARGS+=(--target_fresh_min_overlap "$TARGET_FRESH_MIN_OVERLAP")
fi
if [[ -n "${TARGET_FRESH_MAX_CENTROID_DIST:-}" ]]; then
    EXTRA_ARGS+=(--target_fresh_max_centroid_dist "$TARGET_FRESH_MAX_CENTROID_DIST")
fi
if [[ -n "${KG_TRACE_DIR:-}" ]]; then
    EXTRA_ARGS+=(--kg_trace_dir "$KG_TRACE_DIR")
fi
if [[ -n "${KG_TRACE_PLOTS:-}" ]]; then
    EXTRA_ARGS+=(--kg_trace_plots "$KG_TRACE_PLOTS")
fi
if [[ -n "${DECISION_JSONL:-}" ]]; then
    EXTRA_ARGS+=(--decision_jsonl "$(abspath_from_root "$DECISION_JSONL")")
fi
if [[ -n "${STOP_DIAG_JSONL:-}" ]]; then
    EXTRA_ARGS+=(--stop_diag_jsonl "$(abspath_from_root "$STOP_DIAG_JSONL")")
fi
if [[ -n "${SEMANTIC_BOOST_BACKEND:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_backend "$SEMANTIC_BOOST_BACKEND")
elif [[ -n "${SEMANTIC_BOOST:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_backend "$SEMANTIC_BOOST")
fi
if [[ -n "${SEMANTIC_BOOST_MODEL_ID:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_model_id "$SEMANTIC_BOOST_MODEL_ID")
fi
if [[ -n "${SEMANTIC_BOOST_SAM_TYPE:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_sam_type "$SEMANTIC_BOOST_SAM_TYPE")
fi
if [[ -n "${SEMANTIC_BOOST_SAM_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_sam_checkpoint "$SEMANTIC_BOOST_SAM_CHECKPOINT")
fi
if [[ -n "${SEMANTIC_BOOST_DEVICE:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_device "$SEMANTIC_BOOST_DEVICE")
fi
if [[ -n "${SEMANTIC_BOOST_INTERVAL:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_interval "$SEMANTIC_BOOST_INTERVAL")
fi
if [[ -n "${SEMANTIC_BOOST_BOX_THRESHOLD:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_box_threshold "$SEMANTIC_BOOST_BOX_THRESHOLD")
fi
if [[ -n "${SEMANTIC_BOOST_TEXT_THRESHOLD:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_text_threshold "$SEMANTIC_BOOST_TEXT_THRESHOLD")
fi
if [[ -n "${SEMANTIC_BOOST_SAM_IOU_THRESHOLD:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_sam_iou_threshold "$SEMANTIC_BOOST_SAM_IOU_THRESHOLD")
fi
if [[ -n "${SEMANTIC_BOOST_MIN_MASK_AREA:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_min_mask_area "$SEMANTIC_BOOST_MIN_MASK_AREA")
fi
if [[ -n "${SEMANTIC_BOOST_MAX_MASK_FRAC:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_max_mask_frac "$SEMANTIC_BOOST_MAX_MASK_FRAC")
fi
if [[ -n "${SEMANTIC_BOOST_MAX_DETECTIONS_PER_CATEGORY:-}" ]]; then
    EXTRA_ARGS+=(--semantic_boost_max_detections_per_category "$SEMANTIC_BOOST_MAX_DETECTIONS_PER_CATEGORY")
fi

cd "$REPO"
"${PYTHON_CMD[@]}" exp_main_brain.py \
    --split "$SPLIT" \
    --llm_path "$MODEL_PATH" \
    --sim_gpu_id "$SIM_GPU_ID" \
    --sem_gpu_id "$SEM_GPU_ID" \
    --llm_gpu_id "$LLM_GPU_ID" \
    --dump_location "$DUMP_LOCATION" \
    --exp_name "$EXP_NAME" \
    --jsonl_log "$JSONL" \
    --method_name mindnav \
    "${EXTRA_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$LOG"
