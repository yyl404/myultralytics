#!/bin/bash
# Incremental inference: run tools/predict.py on each dataset yaml in a sequence.
#
# Usage:
#   bash scripts/predict.sh --model runs/<run>/task-2/best.pt --tasks t1.yaml t2.yaml
#   bash scripts/predict.sh --model runs/<run>/task-2/best.pt --tasks t1.yaml t2.yaml \
#       --cumulative c1.yaml c2.yaml --save-path runs/<run>/predictions
#
# Class IDs are aligned to the model (same conversion as eval) before inference,
# so every yaml is scored against its GT labels (metrics.csv per dataset). For
# inference on images without labels, call tools/predict.py --images DIR directly.
# Extra flags after -- are forwarded to tools/predict.py (e.g. --conf 0.25
# --agnostic_nms True).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Incremental inference over a yaml sequence. Dumps boxes, GT-match folders, and
metrics.csv per dataset.

  bash scripts/predict.sh --model runs/<run>/task-2/best.pt --tasks t1.yaml t2.yaml
  bash scripts/predict.sh --model runs/<run>/task-2/best.pt --tasks t1.yaml t2.yaml \
      --cumulative c1.yaml c2.yaml --save-path runs/<run>/predictions

Options:
  --model FILE        Detector checkpoint (.pt)
  --tasks yaml [yaml ...]       Dataset yaml sequence (per-task datasets)
  --cumulative yaml [yaml ...]  Optional cumulative dataset yaml sequence
  --save-path DIR     Root output dir (default: <model-dir>/predict)
  --split auto|test|val|train   Split to run on (default: auto = test, else val)
  --                  Extra flags forwarded to tools/predict.py

Env: DEVICE (default 0).
EOF
}

MODEL=""
SAVE_PATH=""
SPLIT="auto"
TASK_YAMLS=()
CUMULATIVE_YAMLS=()
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --model)
            MODEL="${2:?}"
            shift 2
            ;;
        --save-path|--save_path)
            SAVE_PATH="${2:?}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?}"
            shift 2
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --cumulative)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            CUMULATIVE_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
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

if [[ -z "$MODEL" && ${#POSITIONAL[@]} -ge 1 ]]; then
    MODEL="${POSITIONAL[0]}"
fi
if [[ -z "$SAVE_PATH" && ${#POSITIONAL[@]} -ge 2 ]]; then
    SAVE_PATH="${POSITIONAL[1]}"
fi
(( ${#POSITIONAL[@]} <= 2 )) || experiment_die "Unexpected extra arguments: ${POSITIONAL[*]:2}"

[[ -n "$MODEL" ]] || {
    usage >&2
    experiment_die "Need --model"
}
[[ -f "$MODEL" ]] || experiment_die "Model checkpoint not found: $MODEL"
if (( ${#TASK_YAMLS[@]} == 0 && ${#CUMULATIVE_YAMLS[@]} == 0 )); then
    usage >&2
    experiment_die "Need --tasks and/or --cumulative <yaml...>"
fi
if (( ${#TASK_YAMLS[@]} > 0 )); then
    experiment_check_yamls "${TASK_YAMLS[@]}"
fi
if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
    experiment_check_yamls "${CUMULATIVE_YAMLS[@]}"
fi

SAVE_PATH="${SAVE_PATH:-$(dirname "$MODEL")/predict}"
mkdir -p "$SAVE_PATH"
# Absolute paths: a relative ultralytics --project would be re-rooted under runs/detect/.
SAVE_PATH="$(cd "$SAVE_PATH" && pwd)"

echo "=========================================="
echo "Incremental predict"
echo "  model   : ${MODEL}"
echo "  tasks   : ${#TASK_YAMLS[@]}  cumulative: ${#CUMULATIVE_YAMLS[@]}"
echo "  split   : ${SPLIT}"
echo "  output  : ${SAVE_PATH}"
echo "=========================================="

# Run tools/predict.py on one dataset yaml with GT labels.
# Args: dataset_yaml output_dir
predict_one() {
    local dataset_yaml="$1" task_out="$2"
    local split
    split="$(experiment_resolve_split "$dataset_yaml" "$SPLIT")"
    local converted_dir="${task_out}/converted"
    mkdir -p "$task_out"
    python tools/convert_dataset_class_ids.py \
        --model "$MODEL" --dataset "$dataset_yaml" \
        --output_dir "$converted_dir" --splits "$split"
    # Clear predictor intermediates from a previous run so reruns do not accumulate predict2/... .
    rm -rf "${task_out}/predict"
    predict_args=(
        python tools/predict.py
        --model "$MODEL"
        --images "${converted_dir}/images/${split}"
        --labels "${converted_dir}/labels/${split}"
        --save_path "$task_out"
        --project "$task_out"
        --device "${DEVICE:-0}"
    )
    if (( ${#PASSTHROUGH[@]} > 0 )); then
        predict_args+=("${PASSTHROUGH[@]}")
    fi
    "${predict_args[@]}"
}

for task_index in "${!TASK_YAMLS[@]}"; do
    predict_one "${TASK_YAMLS[$task_index]}" "${SAVE_PATH}/task-$((task_index + 1))"
done
for task_index in "${!CUMULATIVE_YAMLS[@]}"; do
    predict_one "${CUMULATIVE_YAMLS[$task_index]}" "${SAVE_PATH}/cumulative-$((task_index + 1))"
done
