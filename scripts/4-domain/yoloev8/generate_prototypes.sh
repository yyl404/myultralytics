#!/bin/bash

# Configuration
OUTPUT_DIR="runs/yolov8l_4-domain_pretrained-yoloe_pseudo_label+proto_rp"
DEVICE=0
IOU_THRESHOLD=0.5

# Start from which task (1-based index, set to 1 to start from beginning)
START_TASK=${START_TASK:-1}

# Specify dataset path for each task
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

# Generate prototypes for each task iteratively
task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    # Skip tasks before START_TASK
    if [ $task_num -lt $START_TASK ]; then
        echo "Skipping task $task_num (before start point)..."
        ((task_num++))
        continue
    fi
    
    echo "=========================================="
    echo "Generating prototypes for Task $task_num"
    echo "Dataset: $DATASET_PATH"
    echo "=========================================="
    
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
    PROTOTYPES_VIS_DIR="$TASK_DIR/prototypes-visualizations"
    
    if [ $task_num -eq 1 ]; then
        # First task: generate prototypes without load_hist
        rm -rf $PROTOTYPES_VIS_DIR
        echo "Generating prototypes for task $task_num..."
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --vis_dir $PROTOTYPES_VIS_DIR \
            --device $DEVICE \
            --iou_threshold $IOU_THRESHOLD
    else
        # Subsequent tasks: convert previous prototypes and generate new ones
        PREV_TASK=$((task_num - 1))
        PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
        PREV_PROTOTYPES="$OUTPUT_DIR/task-$PREV_TASK/prototypes.pt"
        EXPANDED_MODEL="$TASK_DIR/task-$PREV_TASK-best-expanded.pt"
        CONVERTED_PROTOTYPES="$TASK_DIR/task-$PREV_TASK-prototypes-converted.pt"
        
        # Extract dataset name for converted dataset path
        # DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        # CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_converted/dataset.yaml"
        
        # Convert prototype classes from previous task
        echo "Converting prototype classes for task $task_num..."
        python tools/convert_prototype_classes.py \
            --prototypes $PREV_PROTOTYPES \
            --original_model $PREV_MODEL \
            --expanded_model $EXPANDED_MODEL \
            --output $CONVERTED_PROTOTYPES
        
        rm -rf $PROTOTYPES_VIS_DIR
        # Generate prototypes with load_hist using converted dataset
        echo "Generating prototypes for task $task_num..."
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --vis_dir $PROTOTYPES_VIS_DIR \
            --load_hist $CONVERTED_PROTOTYPES \
            --device $DEVICE \
            --iou_threshold $IOU_THRESHOLD
    fi
    
    echo "Task $task_num prototypes generation completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All prototypes generation completed!"
echo "=========================================="
