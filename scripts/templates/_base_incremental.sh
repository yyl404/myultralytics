#!/bin/bash
# Primary template for incremental learning scripts.
# Usage: Source or copy this structure; override MODEL_CFG, OUTPUT_DIR, TASK_DATASETS,
#        and add method-specific logic (naive / Pseudo Label / ESPReg / EWC / Prototype Replay).

# ---------- Common configuration (override per dataset/backbone) ----------
MODEL_CFG="${MODEL_CFG:-yolov8l.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/incremental}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PATIENCE="${PATIENCE:-10}"
SAVE_PERIOD="${SAVE_PERIOD:-50}"

# ---------- Task loop control ----------
START_TASK=${START_TASK:-1}
# TASK_DATASETS must be set by caller, e.g.:
# TASK_DATASETS=("data/VOC_15_5/task_1_cls_15/dataset.yaml" "data/VOC_15_5/task_2_cls_5/dataset.yaml")

# ---------- Validate START_TASK ----------
if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    echo "Error: START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

# ---------- Resume state ----------
PREV_MODEL=""
PREV_PCA_CACHE=""
PREV_IMPORTANCE_PATH=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    PREV_IMPORTANCE_PATH="$OUTPUT_DIR/task-$PREV_TASK/importance.pth"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK (previous model: $PREV_MODEL)"
fi

# ---------- Task loop (method-specific first-task / subsequent-task logic goes in caller) ----------
task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    TASK_DIR="$OUTPUT_DIR/task-$task_num"
    echo "=========================================="
    echo "Processing Task $task_num: $DATASET_PATH"
    echo "=========================================="
    # ... first task vs subsequent task: train / expand head / convert IDs / train with method ...
    ((task_num++))
done
echo "All tasks completed."
