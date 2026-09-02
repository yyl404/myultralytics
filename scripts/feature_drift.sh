#!/bin/bash
# Feature drift between two checkpoints on this split's task-1 images.
#
# Usage:
#   bash scripts/feature_drift.sh --dataset voc-tiny --split 15_5 --model1 CKPT --model2 CKPT [SAVE_PATH]
#   bash scripts/feature_drift.sh voc-tiny 15_5 CKPT1 CKPT2 [SAVE_PATH]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=lib/experiment.sh
source scripts/lib/experiment.sh

usage() {
    cat <<'EOF'
Feature drift between two checkpoints on this split's task-1 images.

  bash scripts/feature_drift.sh --dataset voc-tiny --split 15_5 --model1 CKPT --model2 CKPT [SAVE_PATH]
  bash scripts/feature_drift.sh voc-tiny 15_5 CKPT1 CKPT2 [SAVE_PATH]
EOF
}

DATASET=""
SPLIT=""
MODEL1=""
MODEL2=""
SAVE_PATH=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --dataset)
            DATASET="${2:?}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?}"
            shift 2
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

DATASET="${DATASET:-${POSITIONAL[0]:-}}"
SPLIT="${SPLIT:-${POSITIONAL[1]:-}}"
MODEL1="${MODEL1:-${POSITIONAL[2]:-}}"
MODEL2="${MODEL2:-${POSITIONAL[3]:-}}"
SAVE_PATH="${SAVE_PATH:-${POSITIONAL[4]:-}}"

[[ -n "$DATASET" && -n "$SPLIT" && -n "$MODEL1" && -n "$MODEL2" ]] || {
    usage >&2
    experiment_die "Need dataset, split, model1, and model2"
}

experiment_load_dataset "$DATASET" "$SPLIT"
SAVE_PATH="${SAVE_PATH:-$(dirname "$MODEL2")/feature_drift_task1_to_task2.json}"

python tools/feature_drift.py \
    --data "${TASK_DATASETS[0]}" \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --save_path "$SAVE_PATH"
