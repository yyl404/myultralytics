#!/bin/bash
# Unified incremental eval entry. Model-agnostic: pass a finished run directory
# plus an explicit eval yaml sequence.
#
# Usage:
#   bash scripts/eval.sh --run runs/<run> --tasks e1.yaml e2.yaml
#   bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml --cumulative c1.yaml c2.yaml
#
# The result matrix is built strictly from what actually exists: every
# task-k/best.pt found under the run directory x every eval yaml given. The
# train sequence and the eval sequence do not have to match in order, kind, or
# length. Cells whose classes are disjoint from the model's class space produce
# empty per-class CSVs and show up as N/A in the tables.
#
# Per-stage task aggregation reads each checkpoint's own incremental_history
# (tools/stage_task_map.py), never the eval yaml order.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental eval: run directory x explicit eval yaml sequence.

  bash scripts/eval.sh --run runs/<run> --tasks e1.yaml e2.yaml
  bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml --cumulative c1.yaml c2.yaml

Options:
  --run DIR                  Run directory holding task-k/best.pt (also positional)
  --tasks yaml [yaml ...]    Per-task eval yaml sequence (required)
  --cumulative yaml [yaml ...]   Cumulative eval yaml sequence (optional)
  --split auto|test|val|train    Split to evaluate on (default: auto = test, else val)
  --iou-threshold FLOAT      Extra per-class AP IoU threshold column (e.g. 0.75)
  --                         Extra flags forwarded to tools/eval.py (e.g. --agnostic_nms True)

Env: DEVICE (default 0).
EOF
}

OUTPUT_DIR=""
SPLIT="auto"
IOU_THRESHOLD=""
EVAL_YAMLS=()
CUMULATIVE_YAMLS=()
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --run|--output)
            OUTPUT_DIR="${2:?}"
            shift 2
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            EVAL_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --cumulative)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            CUMULATIVE_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --split)
            SPLIT="${2:?}"
            shift 2
            ;;
        --iou-threshold|--iou_threshold)
            IOU_THRESHOLD="${2:?}"
            shift 2
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

if [[ -z "$OUTPUT_DIR" && ${#POSITIONAL[@]} -ge 1 ]]; then
    OUTPUT_DIR="${POSITIONAL[0]}"
fi
[[ -n "$OUTPUT_DIR" ]] || {
    usage >&2
    experiment_die "Need a run directory (--run)"
}
(( ${#EVAL_YAMLS[@]} > 0 )) || {
    usage >&2
    experiment_die "Need --tasks <yaml...>"
}
experiment_check_yamls "${EVAL_YAMLS[@]}"
if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
    experiment_check_yamls "${CUMULATIVE_YAMLS[@]}"
fi
# Absolute paths: a relative ultralytics --project would be re-rooted under runs/detect/.
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

experiment_list_model_tasks "$OUTPUT_DIR"
MODEL_TASKS=("${EXPERIMENT_MODEL_TASKS[@]}")

EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
mkdir -p "$EVAL_OUTPUT_DIR"

eval_extra=(--device "${DEVICE:-0}")
if [[ -n "$IOU_THRESHOLD" ]]; then
    eval_extra+=(--iou_threshold "$IOU_THRESHOLD")
fi
if (( ${#PASSTHROUGH[@]} > 0 )); then
    eval_extra+=("${PASSTHROUGH[@]}")
fi

echo "=========================================="
echo "Incremental evaluation"
echo "  run     : ${OUTPUT_DIR}"
echo "  results : ${EVAL_OUTPUT_DIR}"
echo "  models  : ${#MODEL_TASKS[@]}  eval yamls: ${#EVAL_YAMLS[@]}  cumulative: ${#CUMULATIVE_YAMLS[@]}"
echo "  split   : ${SPLIT}"
echo "=========================================="

# Evaluate one model on one dataset yaml and write the per-class CSVs.
# Args: model_path dataset_yaml tag
eval_one() {
    local model_path="$1" dataset_yaml="$2" tag="$3"
    local split
    split="$(experiment_resolve_split "$dataset_yaml" "$SPLIT")"
    # The ultralytics validator requires train/val keys in the data yaml, so
    # convert every split the yaml defines, not only the evaluated one.
    local available_splits
    available_splits="$(experiment_yaml_splits "$dataset_yaml")"
    local converted_dir="${EVAL_OUTPUT_DIR}/${tag}_converted"
    python tools/convert_dataset_class_ids.py \
        --model "$model_path" --dataset "$dataset_yaml" \
        --output_dir "$converted_dir" --splits $available_splits
    # Clear validator intermediates from a previous eval so reruns do not accumulate val2/val3/... .
    rm -rf "${EVAL_OUTPUT_DIR}/${tag}"
    python tools/eval.py \
        --model "$model_path" --data "${converted_dir}/dataset.yaml" \
        --save_path "${EVAL_OUTPUT_DIR}/${tag}.csv" \
        --confusion_matrix_path "${EVAL_OUTPUT_DIR}/${tag}_confusion_matrix.csv" \
        --project "${EVAL_OUTPUT_DIR}/${tag}" \
        --split "$split" \
        "${eval_extra[@]}"
}

for model_task in "${MODEL_TASKS[@]}"; do
    MODEL_PATH="${OUTPUT_DIR}/task-${model_task}/best.pt"
    echo "=========================================="
    echo "Evaluating model from task ${model_task}"
    echo "=========================================="
    for dataset_index in "${!EVAL_YAMLS[@]}"; do
        dataset_task=$((dataset_index + 1))
        eval_one "$MODEL_PATH" "${EVAL_YAMLS[$dataset_index]}" "model_${model_task}_eval_task_${dataset_task}"
    done
    for dataset_index in "${!CUMULATIVE_YAMLS[@]}"; do
        cumulative_task=$((dataset_index + 1))
        eval_one "$MODEL_PATH" "${CUMULATIVE_YAMLS[$dataset_index]}" "model_${model_task}_eval_cumulative_${cumulative_task}"
    done
    echo ""
done

table_args=(
    --eval_dir "$EVAL_OUTPUT_DIR"
    --model_tasks "${MODEL_TASKS[@]}"
    --num_eval_tasks "${#EVAL_YAMLS[@]}"
    --output_dir "$EVAL_OUTPUT_DIR"
)
if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
    table_args+=(--num_cumulative "${#CUMULATIVE_YAMLS[@]}")
fi
python tools/generate_eval_tables.py "${table_args[@]}"

# Per-stage task aggregation from each checkpoint's own incremental_history.
python tools/stage_task_map.py --run_dir "$OUTPUT_DIR" --eval_dir "$EVAL_OUTPUT_DIR"

echo "Evaluation complete. Tables under ${EVAL_OUTPUT_DIR}"
