#!/bin/bash

# Configuration
OUTPUT_DIR="${1:-runs/yolov8l_4-domain_pretrained-yoloe_espreg}"  # Default value if not specified
DEVICE=0

# Start from which task (1-based index, set to 1 to start from beginning)
START_TASK=${START_TASK:-1}

# Specify dataset path for each task
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

# Generate PCA cache for each task iteratively
task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    # Skip tasks before START_TASK
    if [ $task_num -lt $START_TASK ]; then
        echo "Skipping task $task_num (before start point)..."
        ((task_num++))
        continue
    fi
    
    echo "=========================================="
    echo "Generating PCA cache for Task $task_num"
    echo "Dataset: $DATASET_PATH"
    echo "=========================================="
    
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
    
    if [ $task_num -eq 1 ]; then
        # First task: generate PCA cache without load_hist
        echo "Performing PCA analysis for task $task_num..."
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --save_path $PCA_CACHE_PATH
    else
        # Subsequent tasks: generate PCA cache with load_hist from previous task
        PREV_TASK=$((task_num - 1))
        PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
        
        echo "Performing PCA analysis for task $task_num..."
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --load_hist $PREV_PCA_CACHE \
            --save_path $PCA_CACHE_PATH
    fi
    
    echo "Task $task_num PCA cache generation completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All PCA cache generation completed!"
echo "=========================================="
