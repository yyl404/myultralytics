#!/bin/bash

# Configuration
MODEL_CFG="yolov8l.yaml"
YOLOE_MODEL_WEIGHT="yoloe-v8l-seg.pt"
OUTPUT_DIR="runs/yolov8l_4-domain_pretrained-yoloe_zero-shot"

# Specify dataset path for each task
# Add or remove entries as needed
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

# Sequentially generate zero-shot fused model
task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "Dataset: $DATASET_PATH"
    echo "=========================================="
    
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    
    if [ $task_num -eq 1 ]; then
        # First task: fuse YOLOE to YOLO and train
        echo "Fusing YOLOE model to YOLO for task $task_num..."
        python tools/fuse_zero-shot_yoloe.py \
            --input "$YOLOE_MODEL_WEIGHT" \
            --output "$TASK_DIR/best.pt" \
            --model_cfg "$MODEL_CFG" \
            --data $DATASET_PATH

        PREV_MODEL="$TASK_DIR/best.pt"
    else
        # Subsequent tasks: extract dataset name from path for output directory naming
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))

        # Expand model head
        echo "Expanding model head for task $task_num..."
        python tools/expand_model_head.py \
            --model $PREV_MODEL \
            --model_cfg $MODEL_CFG \
            --dataset $DATASET_PATH \
            --save_path "$TASK_DIR/best.pt" \
            --class_embedding_init \
            --yoloe_model $YOLOE_MODEL_WEIGHT
        
        PREV_MODEL="$TASK_DIR/best.pt"
    fi
    
    echo "Task $task_num completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="

