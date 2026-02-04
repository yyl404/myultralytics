#!/bin/bash

# Configuration
MODEL_CFG="yolov8l.yaml"
OUTPUT_DIR="runs/yolov8l_4-domain_fromscratch_espreg+proto_rp"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=15
NUM_PROTOS=10
PROTO_RP_LOSS_WEIGHT=10000

# PRoRP configuration
# Set to True to use base_model distillation, False to use prototype's built-in supervision
PROTO_RP_USE_BASE_MODEL=${PROTO_RP_USE_BASE_MODEL:-True}

# Start from which task (1-based index, set to 1 to start from beginning)
# Useful for resuming training from a specific task
START_TASK=${START_TASK:-1}

# Specify dataset path for each task
# Add or remove entries as needed
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

# Initialize PREV_PCA_CACHE and PREV_PROTOTYPES for first task
PREV_PCA_CACHE=""
PREV_PROTOTYPES=""

# If starting from a task other than 1, set PREV_MODEL to the previous task's model
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    PREV_PROTOTYPES="$OUTPUT_DIR/task-$PREV_TASK/prototypes.pt"
    
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        echo "Cannot resume from task $START_TASK without task $PREV_TASK model."
        exit 1
    fi
    
    if [ ! -f "$PREV_PCA_CACHE" ]; then
        echo "Error: Previous task PCA cache not found: $PREV_PCA_CACHE"
        echo "You can regenerate previouse task PCA cache using tools/pca.py."
        exit 1
    fi
    
    if [ ! -f "$PREV_PROTOTYPES" ]; then
        echo "Error: Previous task prototypes not found: $PREV_PROTOTYPES"
        echo "You can regenerate previouse task prototypes cache using tools/generate_prototypes.py."
    fi
    
    echo "=========================================="
    echo "Resuming from Task $START_TASK"
    echo "Using previous model: $PREV_MODEL"
    echo "Using previous PCA cache: $PREV_PCA_CACHE"
    echo "Using previous prototypes: $PREV_PROTOTYPES"
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
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \
            --patience $PATIENCE
        
        # Perform PCA on model's input
        echo "Performing PCA analysis on task $task_num..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --save_path $PCA_CACHE_PATH
        
        # Generate prototypes
        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        PROTOTYPES_VIS_DIR="$TASK_DIR/prototypes-visualizations"
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --vis_dir $PROTOTYPES_VIS_DIR \
            --device $DEVICE \
            --num_protos $NUM_PROTOS
        
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
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
            --save_path $EXPANDED_MODEL \
            --yoloe_model $YOLOE_MODEL_WEIGHT

        # Convert prototype classes if prototypes exist from previous task
        echo "Converting prototype classes for task $task_num..."
        CONVERTED_PROTOTYPES="$TASK_DIR/task-$((task_num-1))-prototypes-converted.pt"
        python tools/convert_prototype_classes.py \
            --prototypes $PREV_PROTOTYPES \
            --original_model $PREV_MODEL \
            --expanded_model $EXPANDED_MODEL \
            --output $CONVERTED_PROTOTYPES

        # Convert dataset class IDs
        echo "Converting dataset class IDs for task $task_num..."
        CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_converted"
        python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $CONVERTED_DATASET
        
        echo "Training task $task_num..."
        # Build training command with optional prototypes
        TRAIN_CMD="python tools/train.py \
            --model $EXPANDED_MODEL \
            --data \"$CONVERTED_DATASET/dataset.yaml\" \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \
            --trainer antiforget \
            --patience $PATIENCE \
            --freeze $FREEZE_INC"
        
        # Add PCA cache
        echo "Using PCA cache from previous task: $PREV_PCA_CACHE"
        TRAIN_CMD="$TRAIN_CMD --espreg True --pca_cache_path $PREV_PCA_CACHE"
        
        # Add prototypes
        echo "Using converted prototypes: $CONVERTED_PROTOTYPES"
        TRAIN_CMD="$TRAIN_CMD --proto_rp True"
        TRAIN_CMD="$TRAIN_CMD --prototypes $CONVERTED_PROTOTYPES"
        TRAIN_CMD="$TRAIN_CMD --proto_rp_use_base_model $PROTO_RP_USE_BASE_MODEL"
        TRAIN_CMD="$TRAIN_CMD --proto_rp_loss_weight $PROTO_RP_LOSS_WEIGHT"
        
        # Execute training command
        eval $TRAIN_CMD

        # Perform PCA on model's input using original dataset (all layers)
        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --load_hist $PREV_PCA_CACHE \
            --save_path $PCA_CACHE_PATH
        
        # Generate prototypes for current task
        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        PROTOTYPES_VIS_DIR="$TASK_DIR/prototypes-visualizations"
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --vis_dir $PROTOTYPES_VIS_DIR \
            --load_hist $CONVERTED_PROTOTYPES \
            --device $DEVICE \
            --num_protos $NUM_PROTOS
        
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
    fi
    
    echo "Task $task_num completed!"
    echo ""
    
    ((task_num++))
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="

