#!/bin/bash
# N x N image-domain deep-feature similarity matrix over a dataset yaml
# sequence, using the backbone of a pretrained model (tools/dataset_similarity.py).
#
# Usage:
#   bash scripts/dataset_similarity.sh --tasks a.yaml b.yaml [c.yaml ...]
#   bash scripts/dataset_similarity.sh --tasks a.yaml b.yaml --weights FILE --split test --save-path out.csv
#
# Options:
#   --tasks yaml [yaml ...]       Dataset yaml sequence (required; order = matrix order)
#   --weights FILE                Pretrained weights (default: yoloe-26m-seg.pt)
#   --split auto|test|val|train   Split to sample (default: auto = test, else val)
#   --save-path FILE              Matrix CSV (default: runs/dataset_similarity/<data-tag>.csv)
#
# Env: DEVICE / TOOL_DEVICE (single GPU, default 0), BATCH_SIZE, IMGSZ, WEIGHTS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
N x N image-domain deep-feature similarity over a dataset yaml sequence.

  bash scripts/dataset_similarity.sh --tasks a.yaml b.yaml [--weights FILE] [--save-path FILE]
EOF
}

WEIGHTS="${WEIGHTS:-yoloe-26m-seg.pt}"
SPLIT="auto"
SAVE_PATH=""
TASK_YAMLS=()

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
        --weights)
            WEIGHTS="${2:?--weights needs a value}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?--split needs a value}"
            shift 2
            ;;
        --save-path|--save_path)
            SAVE_PATH="${2:?--save-path needs a value}"
            shift 2
            ;;
        --*)
            experiment_die "Unknown option: $1"
            ;;
        *)
            experiment_die "Unexpected argument: $1"
            ;;
    esac
done

(( ${#TASK_YAMLS[@]} > 0 )) || {
    usage >&2
    experiment_die "Need --tasks <yaml...>"
}
[[ -f "$WEIGHTS" ]] || experiment_die "Weights not found: $WEIGHTS"

experiment_load_custom_tasks "${TASK_YAMLS[@]}"
SAVE_PATH="${SAVE_PATH:-runs/dataset_similarity/${DATA_TAG}.csv}"
mkdir -p "$(dirname "$SAVE_PATH")"

# Single-GPU tool: first GPU of DEVICE unless TOOL_DEVICE overrides it.
DEVICE="${TOOL_DEVICE:-${DEVICE:-0}}"
DEVICE="${DEVICE%%,*}"

echo "=========================================="
echo "Dataset similarity"
echo "  weights : ${WEIGHTS}"
echo "  datasets: ${#TASK_DATASETS[@]}  split=${SPLIT}  device=${DEVICE}"
echo "  output  : ${SAVE_PATH}"
echo "=========================================="

python tools/dataset_similarity.py \
    --data "${TASK_DATASETS[@]}" \
    --weights "$WEIGHTS" \
    --split "$SPLIT" \
    --batch "${BATCH_SIZE:-16}" \
    --imgsz "${IMGSZ:-640}" \
    --device "$DEVICE" \
    --save_path "$SAVE_PATH"
