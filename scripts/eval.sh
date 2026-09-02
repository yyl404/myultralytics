#!/bin/bash
# Unified incremental eval entry. Model-agnostic: pass a finished run directory.
#
# Usage:
#   bash scripts/eval.sh --run runs/<run>
#   bash scripts/eval.sh --dataset voc-tiny --split 15_5 --run runs/<run>
#   bash scripts/eval.sh runs/<run>
#
# Dataset/split are inferred from the run folder name when omitted.
# CIL runs also evaluate cumulative tasks and write final_cumulative_task_mAP.csv.
# TIL (odinw-13) evaluates only individual tasks seen so far.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=lib/experiment.sh
source scripts/lib/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental eval. Pass a finished run directory (dataset/split inferred
from the folder name when omitted).

  bash scripts/eval.sh --run runs/<run>
  bash scripts/eval.sh --dataset voc-tiny --split 15_5 --run runs/<run>
  bash scripts/eval.sh runs/<run>
EOF
}

DATASET=""
SPLIT=""
OUTPUT_DIR=""
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

if [[ -n "$DATASET" || -n "$SPLIT" ]]; then
    [[ -n "$DATASET" && -n "$SPLIT" ]] || experiment_die "Pass both --dataset and --split, or neither (infer from run name)"
    experiment_load_dataset "$DATASET" "$SPLIT"
else
    experiment_infer_dataset_from_run "$OUTPUT_DIR"
fi

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
echo "  tasks   : ${NUM_TASKS}"
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
        DATASET_PATH="${TASK_DATASETS[$((dataset_task - 1))]}"
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
    if [[ "$INCREMENTAL_SETTING" == "cil" ]]; then
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

if [[ "$INCREMENTAL_SETTING" == "cil" ]]; then
    python tools/summarize_cumulative_task_map.py \
        --evaluation_csv "${EVAL_OUTPUT_DIR}/model_${NUM_TASKS}_eval_cumulative.csv" \
        --task_data "${TASK_DATASETS[@]}" \
        --output "${EVAL_OUTPUT_DIR}/final_cumulative_task_mAP.csv"
fi
