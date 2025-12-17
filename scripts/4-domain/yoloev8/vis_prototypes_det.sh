#!/bin/bash

# Configuration
OUTPUT_DIR="runs/yolov8l_4-domain_pretrained-yoloe_proto_rp"
DEVICE=0

# Start from which task (1-based index, set to 1 to start from beginning)
START_TASK=${START_TASK:-1}

# Specify number of tasks (should match the number of tasks in training)
NUM_TASKS=4

# Visualize prototypes for each task iteratively
task_num=1
while [ $task_num -le $NUM_TASKS ]; do
    # Skip tasks before START_TASK
    if [ $task_num -lt $START_TASK ]; then
        echo "Skipping task $task_num (before start point)..."
        ((task_num++))
        continue
    fi
    
    echo "=========================================="
    echo "Visualizing prototypes for Task $task_num"
    echo "=========================================="
    
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
    PROTOTYPES_DET_DIR="$TASK_DIR/prototypes-pred-results"
    MODEL_PATH="$TASK_DIR/best.pt"
    
    # Visualize prototypes
    echo "Visualizing prototypes for task $task_num..."
    rm -rf $PROTOTYPES_DET_DIR
    python tools/vis_prototypes_det.py \
        --model $MODEL_PATH \
        --prototypes $PROTOTYPES_PATH \
        --output $PROTOTYPES_DET_DIR \
        --device $DEVICE
    
    echo "Task $task_num visualization completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All prototypes visualization completed!"
echo "=========================================="
