#!/bin/bash
# Feature drift between two checkpoints on task-1 images.
#
# Usage:
#   bash scripts/feature_drift.sh --tasks t1.yaml t2.yaml --model1 CKPT --model2 CKPT [SAVE_PATH]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Feature drift between two checkpoints on task-1 images.

  bash scripts/feature_drift.sh --tasks t1.yaml t2.yaml --model1 CKPT --model2 CKPT [SAVE_PATH]
EOF
}

MODEL1=""
MODEL2=""
SAVE_PATH=""
TASK_YAMLS=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --model1)
            MODEL1="${2:?}"
            shift 2
            ;;
        --model2)
            MODEL2="${2:?}"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="${2:?}"
            shift 2
            ;;
        --*)
            experiment_die "Unknown option: $1"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

MODEL1="${MODEL1:-${POSITIONAL[0]:-}}"
MODEL2="${MODEL2:-${POSITIONAL[1]:-}}"
SAVE_PATH="${SAVE_PATH:-${POSITIONAL[2]:-}}"

(( ${#TASK_YAMLS[@]} > 0 )) || {
    usage >&2
    experiment_die "Need --tasks <yaml...>"
}
[[ -n "$MODEL1" && -n "$MODEL2" ]] || {
    usage >&2
    experiment_die "Need model1 and model2"
}

experiment_load_custom_tasks "${TASK_YAMLS[@]}"
SAVE_PATH="${SAVE_PATH:-$(dirname "$MODEL2")/feature_drift_task1_to_task2.json}"

python tools/feature_drift.py \
    --data "${TASK_DATASETS[0]}" \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --save_path "$SAVE_PATH"
