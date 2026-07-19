#!/usr/bin/env bash
# Isolated launcher for the MindNav 1/3-robot count ablation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_AGENTS=${NUM_AGENTS:-}

case "$NUM_AGENTS" in
    1|3) ;;
    2)
        echo "NUM_AGENTS=2 is frozen; reuse M-DS-TEXT instead of rerunning it." >&2
        exit 2
        ;;
    *)
        echo "Robot-count ablation requires NUM_AGENTS=1 or 3." >&2
        exit 2
        ;;
esac

exec bash "$SCRIPT_DIR/run_mindnav.sh" --num_agents "$NUM_AGENTS" "$@"
