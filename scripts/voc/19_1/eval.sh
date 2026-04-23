#!/bin/bash
# Evaluation script for VOC 15_5 incremental setting.
# Independent of initial weight/backbone: pass any training run OUTPUT_DIR for this split.
# Usage: bash eval.sh [OUTPUT_DIR]
# Example: bash eval.sh runs/yolov8l_voc_15_5_fromscratch_pseudo_label

OUTPUT_DIR="${1:-runs/yolov8l_voc_15_5_fromscratch_pseudo_label}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE=0

TASK_DATASETS=(
    "data/VOC_15+5/task_1_cls_15/dataset.yaml"
    "data/VOC_15+5/task_2_cls_5/dataset.yaml"
)

CUMULATIVE_DATASETS=(
    "data/VOC_15+5/task_1_cls_15/dataset.yaml"
    "data/VOC_15+5/task_1-2_cls_20/dataset.yaml"
)

NUM_TASKS=${#TASK_DATASETS[@]}
mkdir -p "$EVAL_OUTPUT_DIR"

echo "=========================================="
echo "Incremental Learning Model Evaluation (VOC 15_5)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation results directory: $EVAL_OUTPUT_DIR"
echo "Number of tasks: $NUM_TASKS"
echo ""

for model_task in $(seq 1 $NUM_TASKS); do
    MODEL_PATH="$OUTPUT_DIR/task-$model_task/best.pt"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Warning: Model not found: $MODEL_PATH"
        continue
    fi
    echo "=========================================="
    echo "Evaluating Model from Task $model_task"
    echo "=========================================="
    for dataset_task in $(seq 1 $model_task); do
        DATASET_PATH="${TASK_DATASETS[$((dataset_task - 1))]}"
        [ ! -f "$DATASET_PATH" ] && echo "Warning: Dataset not found: $DATASET_PATH" && continue
        CONVERTED_DATASET_DIR="$EVAL_OUTPUT_DIR/task_${model_task}_task_${dataset_task}_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" --dataset "$DATASET_PATH" \
            --output_dir "$CONVERTED_DATASET_DIR" --splits train val test
        [ $? -ne 0 ] && continue
        python tools/eval.py \
            --model "$MODEL_PATH" --data "$CONVERTED_DATASET_DIR/dataset.yaml" \
            --device "$DEVICE" --batch 1 \
            --save_path "$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}.csv" \
            --confusion_matrix_path "$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}_confusion_matrix.csv" \
            --project "$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}"
    done
    CUMULATIVE_DATASET_PATH="${CUMULATIVE_DATASETS[$((model_task - 1))]}"
    if [ -f "$CUMULATIVE_DATASET_PATH" ]; then
        CUMULATIVE_CONVERTED_DIR="$EVAL_OUTPUT_DIR/task_${model_task}_cumulative_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" --dataset "$CUMULATIVE_DATASET_PATH" \
            --output_dir "$CUMULATIVE_CONVERTED_DIR" --splits train val test
        [ $? -eq 0 ] && python tools/eval.py \
            --model "$MODEL_PATH" --data "$CUMULATIVE_CONVERTED_DIR/dataset.yaml" \
            --device "$DEVICE" \
            --save_path "$EVAL_OUTPUT_DIR/model_${model_task}_eval_cumulative.csv" \
            --confusion_matrix_path "$EVAL_OUTPUT_DIR/model_${model_task}_eval_cumulative_confusion_matrix.csv" \
            --project "$EVAL_OUTPUT_DIR/model_${model_task}_eval_cumulative"
    fi
    echo ""
done

python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" --num_tasks "$NUM_TASKS" --output_dir "$EVAL_OUTPUT_DIR"
[ $? -eq 0 ] && echo "Evaluation complete. Tables: $EVAL_OUTPUT_DIR/individual_datasets_eval.csv, $EVAL_OUTPUT_DIR/cumulative_datasets_eval.csv" || exit 1
