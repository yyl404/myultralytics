#!/bin/bash
# Evaluation script for OdinW-13 task-incremental setting (lexicographic task order).
# No cumulative datasets: only per-task individual evaluations.
# Usage: bash eval.sh [OUTPUT_DIR]
# Example: bash eval.sh runs/yolov8x_OdinW-13-yolo_pretrained-from-yolov8x-cls_pseudo_label

OUTPUT_DIR="${1:-runs/yolov8x_OdinW-13-yolo_pretrained-from-yolov8x-cls_pseudo_label}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE=0

TASK_DATASETS=(
    "data/OdinW-13-yolo/AerialMaritimeDrone/data.yaml"
    "data/OdinW-13-yolo/Aquarium/data.yaml"
    "data/OdinW-13-yolo/CottontailRabbits/data.yaml"
    "data/OdinW-13-yolo/EgoHands/data.yaml"
    "data/OdinW-13-yolo/NorthAmericaMushrooms/data.yaml"
    "data/OdinW-13-yolo/Packages/data.yaml"
    "data/OdinW-13-yolo/PascalVOC/data.yaml"
    "data/OdinW-13-yolo/Raccoon/data.yaml"
    "data/OdinW-13-yolo/ShellfishOpenImages/data.yaml"
    "data/OdinW-13-yolo/VehiclesOpenImages/data.yaml"
    "data/OdinW-13-yolo/pistols/data.yaml"
    "data/OdinW-13-yolo/pothole/data.yaml"
    "data/OdinW-13-yolo/thermalDogsAndPeople/data.yaml"
)

NUM_TASKS=${#TASK_DATASETS[@]}
mkdir -p "$EVAL_OUTPUT_DIR"

echo "=========================================="
echo "Incremental Learning Model Evaluation (OdinW-13, task-incremental)"
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
    echo ""
done

python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" --num_tasks "$NUM_TASKS" --output_dir "$EVAL_OUTPUT_DIR"
[ $? -eq 0 ] && echo "Evaluation complete. Table: $EVAL_OUTPUT_DIR/individual_datasets_eval.csv" || exit 1
