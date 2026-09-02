#!/bin/bash
# Unified incremental eval entry. Model-agnostic: pass a finished run directory.
#
# Usage:
#   bash scripts/eval.sh --run runs/<run>
#   bash scripts/eval.sh --dataset voc-tiny --split 15_5 --run runs/<run>
#   bash scripts/eval.sh --tasks t1.yaml t2.yaml --cumulative c1.yaml c2.yaml --run runs/<run>
#   bash scripts/eval.sh runs/<run>
#
# The yaml sequences come from (first match wins): explicit --tasks/--eval-tasks/
# --cumulative, --dataset/--split, the run manifest written by train.sh, or the
# run folder name. Per-task eval defaults to the train sequence; cumulative eval
# runs only when a cumulative sequence is available (registered CIL splits always
# have one). Cumulative results include final_cumulative_task_mAP.csv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental eval. Pass a finished run directory; the yaml sequences are
recovered from the run manifest or inferred from the folder name when not given.

  bash scripts/eval.sh --run runs/<run>
  bash scripts/eval.sh --dataset voc-tiny --split 15_5 --run runs/<run>
  bash scripts/eval.sh --tasks t1.yaml t2.yaml --run runs/<run>
  bash scripts/eval.sh runs/<run>

Options:
  --run DIR                Run directory to evaluate (also accepted positionally)
  --dataset --split        Registered experiment identity
  --tasks yaml [yaml ...]  Explicit train task yaml sequence
  --eval-tasks yaml [yaml ...]   Per-task eval yamls (default: the train sequence)
  --cumulative yaml [yaml ...]   Cumulative eval yamls, one per task (optional)
  --tag NAME               Override the auto-derived DATA_TAG
EOF
}

DATASET=""
SPLIT=""
OUTPUT_DIR=""
DATA_TAG_OVERRIDE=""
TASK_YAMLS=()
EVAL_YAMLS=()
CUMULATIVE_YAMLS=()
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
        --run|--output)
            OUTPUT_DIR="${2:?}"
            shift 2
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#TASK_YAMLS[@]} > 0 )) || experiment_die "--tasks needs at least one yaml"
            ;;
        --eval-tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            EVAL_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#EVAL_YAMLS[@]} > 0 )) || experiment_die "--eval-tasks needs at least one yaml"
            ;;
        --cumulative)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            CUMULATIVE_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#CUMULATIVE_YAMLS[@]} > 0 )) || experiment_die "--cumulative needs at least one yaml"
            ;;
        --tag)
            DATA_TAG_OVERRIDE="${2:?}"
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

if [[ -z "$OUTPUT_DIR" && ${#POSITIONAL[@]} -ge 1 ]]; then
    OUTPUT_DIR="${POSITIONAL[0]}"
fi
[[ -n "$OUTPUT_DIR" ]] || {
    usage >&2
    experiment_die "Need a run directory (--run)"
}

experiment_resolve_eval_dataset "$OUTPUT_DIR"

EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
NUM_TASKS=${#TASK_DATASETS[@]}
mkdir -p "$EVAL_OUTPUT_DIR"

eval_extra=()
if [[ -n "${EVAL_IOU_THRESHOLD:-}" ]]; then
    eval_extra+=(--iou_threshold "$EVAL_IOU_THRESHOLD")
fi

echo "=========================================="
echo "Incremental evaluation (${DATA_TAG}, ${INCREMENTAL_SETTING})"
echo "  run     : ${OUTPUT_DIR}"
echo "  results : ${EVAL_OUTPUT_DIR}"
echo "  tasks   : ${NUM_TASKS}  (eval yamls: ${#EVAL_DATASETS[@]}, cumulative: ${#CUMULATIVE_DATASETS[@]})"
echo "=========================================="

for model_task in $(seq 1 "$NUM_TASKS"); do
    MODEL_PATH="${OUTPUT_DIR}/task-${model_task}/best.pt"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Warning: Model not found: $MODEL_PATH" >&2
        continue
    fi
    echo "=========================================="
    echo "Evaluating model from task ${model_task}"
    echo "=========================================="
    for dataset_task in $(seq 1 "$model_task"); do
        DATASET_PATH="${EVAL_DATASETS[$((dataset_task - 1))]}"
        if [[ ! -f "$DATASET_PATH" ]]; then
            echo "Warning: Dataset not found: $DATASET_PATH" >&2
            continue
        fi
        CONVERTED_DATASET_DIR="${EVAL_OUTPUT_DIR}/task_${model_task}_task_${dataset_task}_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" --dataset "$DATASET_PATH" \
            --output_dir "$CONVERTED_DATASET_DIR" --splits train val test
        python tools/eval.py \
            --model "$MODEL_PATH" --data "${CONVERTED_DATASET_DIR}/dataset.yaml" \
            --device "$DEVICE" --batch 1 \
            --save_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}.csv" \
            --confusion_matrix_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}_confusion_matrix.csv" \
            --project "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}" \
            "${eval_extra[@]}"
    done
    if (( ${#CUMULATIVE_DATASETS[@]} > 0 )); then
        CUMULATIVE_DATASET_PATH="${CUMULATIVE_DATASETS[$((model_task - 1))]}"
        if [[ -f "$CUMULATIVE_DATASET_PATH" ]]; then
            CUMULATIVE_CONVERTED_DIR="${EVAL_OUTPUT_DIR}/task_${model_task}_cumulative_converted"
            python tools/convert_dataset_class_ids.py \
                --model "$MODEL_PATH" --dataset "$CUMULATIVE_DATASET_PATH" \
                --output_dir "$CUMULATIVE_CONVERTED_DIR" --splits train val test
            python tools/eval.py \
                --model "$MODEL_PATH" --data "${CUMULATIVE_CONVERTED_DIR}/dataset.yaml" \
                --device "$DEVICE" \
                --save_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative.csv" \
                --confusion_matrix_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative_confusion_matrix.csv" \
                --project "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative" \
                "${eval_extra[@]}"
        fi
    fi
    echo ""
done

python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" --num_tasks "$NUM_TASKS" --output_dir "$EVAL_OUTPUT_DIR"
echo "Evaluation complete. Tables under ${EVAL_OUTPUT_DIR}"

if (( ${#CUMULATIVE_DATASETS[@]} > 0 )); then
    python tools/summarize_cumulative_task_map.py \
        --evaluation_csv "${EVAL_OUTPUT_DIR}/model_${NUM_TASKS}_eval_cumulative.csv" \
        --task_data "${EVAL_DATASETS[@]}" \
        --output "${EVAL_OUTPUT_DIR}/final_cumulative_task_mAP.csv"
fi
