#!/bin/bash

set -euo pipefail

OUTPUT_DIR="${1:?Usage: bash scripts/coco/70_10/eval.sh <training-output-dir>}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE="${DEVICE:-0}"

TASK_DATASETS=(
    "data/COCO_70+10/task_1_cls_70/dataset.yaml"
    "data/COCO_70+10/task_2_cls_10/dataset.yaml"
)
CUMULATIVE_DATASETS=(
    "data/COCO_70+10/task_1_cls_70/dataset.yaml"
    "data/COCO_70+10/task_1-2_cls_80/dataset.yaml"
)
NUM_TASKS=${#TASK_DATASETS[@]}
mkdir -p "$EVAL_OUTPUT_DIR"

for model_task in $(seq 1 "$NUM_TASKS"); do
    MODEL_PATH="${OUTPUT_DIR}/task-${model_task}/best.pt"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Warning: Model not found: $MODEL_PATH" >&2
        continue
    fi

    for dataset_task in $(seq 1 "$model_task"); do
        DATASET_PATH="${TASK_DATASETS[$((dataset_task - 1))]}"
        if [[ ! -f "$DATASET_PATH" ]]; then
            echo "Warning: Dataset not found: $DATASET_PATH" >&2
            continue
        fi
        CONVERTED_DATASET_DIR="${EVAL_OUTPUT_DIR}/task_${model_task}_task_${dataset_task}_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" \
            --dataset "$DATASET_PATH" \
            --output_dir "$CONVERTED_DATASET_DIR" \
            --splits train val test
        python tools/eval.py \
            --model "$MODEL_PATH" \
            --data "${CONVERTED_DATASET_DIR}/dataset.yaml" \
            --device "$DEVICE" \
            --batch 1 \
            --iou_threshold 0.75 \
            --save_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}.csv" \
            --confusion_matrix_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}_confusion_matrix.csv" \
            --project "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_task_${dataset_task}"
    done

    CUMULATIVE_DATASET_PATH="${CUMULATIVE_DATASETS[$((model_task - 1))]}"
    if [[ -f "$CUMULATIVE_DATASET_PATH" ]]; then
        CUMULATIVE_CONVERTED_DIR="${EVAL_OUTPUT_DIR}/task_${model_task}_cumulative_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" \
            --dataset "$CUMULATIVE_DATASET_PATH" \
            --output_dir "$CUMULATIVE_CONVERTED_DIR" \
            --splits train val test
        python tools/eval.py \
            --model "$MODEL_PATH" \
            --data "${CUMULATIVE_CONVERTED_DIR}/dataset.yaml" \
            --device "$DEVICE" \
            --iou_threshold 0.75 \
            --save_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative.csv" \
            --confusion_matrix_path "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative_confusion_matrix.csv" \
            --project "${EVAL_OUTPUT_DIR}/model_${model_task}_eval_cumulative"
    fi
done

python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" \
    --num_tasks "$NUM_TASKS" \
    --output_dir "$EVAL_OUTPUT_DIR"
python tools/summarize_cumulative_task_map.py \
    --evaluation_csv "${EVAL_OUTPUT_DIR}/model_${NUM_TASKS}_eval_cumulative.csv" \
    --task_data "${TASK_DATASETS[@]}" \
    --output "${EVAL_OUTPUT_DIR}/final_cumulative_task_mAP.csv"
