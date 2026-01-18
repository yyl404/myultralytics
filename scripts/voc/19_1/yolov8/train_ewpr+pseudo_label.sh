#!/bin/bash

# Configuration
MODEL_CFG="yolov8l.yaml"
OUTPUT_DIR="runs/yolov8l_voc_19_1_fromscratch_ewpr+pseudo_label"
EPOCHS=300
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10

# Pseudo Label Configuration
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5

# EWPR Configuration
EWPR_LOSS_WEIGHT=1.0

# Start from which task (1-based index, set to 1 to start from beginning)
# Useful for resuming training from a specific task
START_TASK=${START_TASK:-1}

# Specify dataset path for each task
# Add or remove entries as needed
TASK_DATASETS=(
    "data/VOC_19_1/task_1_cls_19/dataset.yaml"
    "data/VOC_19_1/task_2_cls_1/dataset.yaml"
)

# Validate START_TASK
if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    echo "Error: START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

# Initialize PREV_PCA_CACHE for first task
PREV_PCA_CACHE=""

# If starting from a task other than 1, set PREV_MODEL to the previous task's model
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        echo "Cannot resume from task $START_TASK without task $PREV_TASK model."
        exit 1
    fi
    
    if [ ! -f "$PREV_PCA_CACHE" ]; then
        echo "Warning: Previous task PCA cache not found: $PREV_PCA_CACHE"
        echo "Training will proceed without PCA cache."
    fi
    
    echo "=========================================="
    echo "Resuming from Task $START_TASK"
    echo "Using previous model: $PREV_MODEL"
    if [ -f "$PREV_PCA_CACHE" ]; then
        echo "Using previous PCA cache: $PREV_PCA_CACHE"
    fi
    echo "=========================================="
    echo ""
fi

# Train each task iteratively
task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    # Skip tasks before START_TASK
    if [ $task_num -lt $START_TASK ]; then
        echo "Skipping task $task_num (before start point)..."
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "Dataset: $DATASET_PATH"
    echo "=========================================="
    
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    
    if [ $task_num -eq 1 ]; then
        # First task: train from scratch
        echo "Training task $task_num from scratch..."
        python tools/train.py --model $MODEL_CFG \
            --data $DATASET_PATH \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR
        
        # Perform PCA on model's input using original dataset (all layers)
        echo "Performing PCA analysis on all layers using original dataset..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --save_path $PCA_CACHE_PATH
        
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
    else
        # Subsequent tasks: extract dataset name from path for output directory naming
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))

        # Expand model head
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \
            --model $PREV_MODEL \
            --model_cfg $MODEL_CFG \
            --dataset $DATASET_PATH \
            --save_path $EXPANDED_MODEL

        # Convert dataset class IDs
        echo "Converting dataset class IDs for task $task_num..."
        ID_CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_id_converted"
        python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $ID_CONVERTED_DATASET
        
        echo "Training task $task_num..."
        # Build training command with optional PCA cache and prototypes
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \
            --data \"$ID_CONVERTED_DATASET/dataset.yaml\" \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \
            --trainer antiforget"

        # Add pseudo labeling
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        
        # Add EWPR loss if available
        if [ -n "$PREV_PCA_CACHE" ] && [ -f "$PREV_PCA_CACHE" ]; then
            echo "Using PCA cache from previous task: $PREV_PCA_CACHE"
            TRAIN_CMD="$TRAIN_CMD --ewpr True --pca_cache_path $PREV_PCA_CACHE --ewpr_loss_weight $EWPR_LOSS_WEIGHT"
        else
            echo "Warning: No PCA cache available from previous task, training without EWPR loss."
        fi
        
        # Execute training command
        eval $TRAIN_CMD

        # Perform PCA on model's input using original dataset (all layers)
        echo "Performing PCA analysis on all layers using original dataset..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --load_hist $PREV_PCA_CACHE \
            --save_path $PCA_CACHE_PATH
        
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
    fi
    
    echo "Task $task_num completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="

