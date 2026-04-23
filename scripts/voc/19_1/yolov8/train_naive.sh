#!/bin/bash

set -uo pipefail

# Console Color Config
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[DONE]${NC} $1"
}

run_step() {
    local step_name="$1"
    shift
    "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Step failed (${step_name}), exit code: ${exit_code}"
        exit $exit_code
    fi
}

# Model Config
MODEL_NAME="yolov8x"
MODEL_WEIGHT_NAME="yolov8x-cls"

# Dataset Config
DATASET_NAME="VOC"
CLASS_SPLITS=(
    19
    1
)

# Train Config
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
FREEZE_LAYERS=None

# Method Config
ANTIFORGET_METHOD="naive"

# Auto Build Variables
MODEL_CFG="${MODEL_NAME}.yaml"
if [ "$MODEL_WEIGHT_NAME" = "None" ]; then
    MODEL_WEIGHT="None"
else
    MODEL_WEIGHT="${MODEL_WEIGHT_NAME}.pt"
fi

CLASS_SPLIT_TAG=$(IFS=+; echo "${CLASS_SPLITS[*]}")
DATASET_ROOT="data/${DATASET_NAME}_${CLASS_SPLIT_TAG}"
TASK_DATASETS=()
for i in "${!CLASS_SPLITS[@]}"; do
    task_id=$((i + 1))
    cls_count="${CLASS_SPLITS[$i]}"
    TASK_DATASETS+=("${DATASET_ROOT}/task_${task_id}_cls_${cls_count}/dataset.yaml")
done

if [ "$MODEL_WEIGHT_NAME" = "None" ]; then
    INIT_TAG="fromscratch"
else
    INIT_TAG="pretrained-from-${MODEL_WEIGHT_NAME}"
fi
OUTPUT_DIR="runs/${MODEL_NAME}_${DATASET_NAME}_${CLASS_SPLIT_TAG}_${INIT_TAG}_${ANTIFORGET_METHOD}"

START_TASK=${START_TASK:-1}
if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    log_error "START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

PREV_MODEL=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        log_error "Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    log_info "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${TASK_DATASETS[@]}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi

    log_info "Start task $task_num"
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
        log_info "Training for task $task_num"
        TRAIN_CMD=(
            python tools/train.py
            --model "$MODEL_CFG"
            --data "$DATASET_PATH"
            --save_path "$TASK_DIR/best.pt"
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --imgsz "$IMGSZ"
            --workers "$WORKERS"
            --device "$DEVICE"
            --project "$TASK_DIR"
        )

        if [ "$MODEL_WEIGHT_NAME" != "None" ]; then
            TRAIN_CMD+=(--weight "$MODEL_WEIGHT")
        fi
        if [ "$FREEZE_LAYERS" != "None" ]; then
            if [ "${#FREEZE_LAYERS[@]}" -gt 0 ]; then
                TRAIN_CMD+=(--freeze "${FREEZE_LAYERS[0]}")
            fi
        fi

        run_step "train task ${task_num}" "${TRAIN_CMD[@]}"

        PREV_MODEL="$TASK_DIR/best.pt"
    else
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"

        log_info "Expanding model head for task $task_num..."
        run_step "expand head task ${task_num}" python tools/expand_model_head.py \
            --model $PREV_MODEL \
            --model_cfg $MODEL_CFG \
            --dataset $DATASET_PATH \
            --save_path $EXPANDED_MODEL

        log_info "Converting dataset class IDs for task $task_num..."
        ID_CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_id_converted"
        run_step "convert dataset ids task ${task_num}" python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $ID_CONVERTED_DATASET

        log_info "Training task $task_num with ${ANTIFORGET_METHOD}"
        TRAIN_CMD=(
            python tools/train.py
            --model "$EXPANDED_MODEL"
            --data "$ID_CONVERTED_DATASET/dataset.yaml"
            --save_path "$TASK_DIR/best.pt"
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --imgsz "$IMGSZ"
            --workers "$WORKERS"
            --device "$DEVICE"
            --project "$TASK_DIR"
        )

        if [ "$FREEZE_LAYERS" != "None" ]; then
            FREEZE_TASK_IDX=$((task_num - 1))
            if [ "$FREEZE_TASK_IDX" -lt "${#FREEZE_LAYERS[@]}" ]; then
                TRAIN_CMD+=(--freeze "${FREEZE_LAYERS[$FREEZE_TASK_IDX]}")
            fi
        fi
        run_step "train task ${task_num}" "${TRAIN_CMD[@]}"

        PREV_MODEL="$TASK_DIR/best.pt"
    fi

    log_success "Task $task_num completed!"
    ((task_num++))
done

log_success "All tasks completed!"
