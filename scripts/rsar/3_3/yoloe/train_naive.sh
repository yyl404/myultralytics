#!/bin/bash
# Configuration
MODEL_CFG="yolov8l-obb.yaml"

YOLOE_MODEL_WEIGHT="yoloe-v8l-seg.pt"

FREEZE_LAYERS=(
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
)
OUTPUT_DIR="runs/yolov8l_rsar_3_3_pretrained-yoloe_naive"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
START_TASK=${START_TASK:-1}

TASK_DATASETS=(
    "data/RSAR_3_3/task_1_cls_3/dataset.yaml"
    "data/RSAR_3_3/task_2_cls_3/dataset.yaml"
)

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    echo "Error: START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

PREV_MODEL=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
        echo "Training task $task_num from pretrained weight..."
        python tools/train.py --model $MODEL_CFG \
            --data $DATASET_PATH \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \
            --weight $YOLOE_MODEL_WEIGHT \
            --freeze ${FREEZE_LAYERS[0]} \
            --patience $PATIENCE
        
        PREV_MODEL="$TASK_DIR/best.pt"

    else
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \
            --model $PREV_MODEL \
            --model_cfg $MODEL_CFG \
            --dataset $DATASET_PATH \
            --save_path $EXPANDED_MODEL
        echo "Converting dataset class IDs for task $task_num..."
        CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_converted"
        python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $CONVERTED_DATASET
        echo "Training task $task_num..."
        python tools/train.py --model $EXPANDED_MODEL \
            --data "$CONVERTED_DATASET/dataset.yaml" \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \
            --patience $PATIENCE
        PREV_MODEL="$TASK_DIR/best.pt"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
