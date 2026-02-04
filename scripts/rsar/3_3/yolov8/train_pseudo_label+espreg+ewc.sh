#!/bin/bash
MODEL_CFG="yolov8l-obb.yaml"
OUTPUT_DIR="runs/yolov8l_rsar_3_3_fromscratch_pseudo_label+espreg+ewc"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
EWC_LOSS_WEIGHT=${EWC_LOSS_WEIGHT:-100.0}
START_TASK=${START_TASK:-1}

TASK_DATASETS=(
    "data/RSAR_3_3/task_1_cls_3/dataset.yaml"
    "data/RSAR_3_3/task_2_cls_3/dataset.yaml"
)

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    echo "Error: START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

PREV_PCA_CACHE=""
PREV_IMPORTANCE_PATH=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_IMPORTANCE_PATH="$OUTPUT_DIR/task-$PREV_TASK/importance.pth"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    if [ ! -f "$PREV_IMPORTANCE_PATH" ]; then
        echo "Warning: Previous task importance not found."
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
        
        PREV_MODEL="$TASK_DIR/best.pt"

        echo "Calculating parameter importance..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --save_path $IMPORTANCE_PATH \
            --module_pattern "*bn" \
            --batch_size $BATCH_SIZE \
            --workers $WORKERS \
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
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
        ID_CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_id_converted"
        python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $ID_CONVERTED_DATASET

        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            echo "Expanding importance file..."
            EXPANDED_IMPORTANCE_PATH="$TASK_DIR/task-$((task_num-1))-importance-expanded.pth"
            python tools/expand_importance.py \
                --old_importance $PREV_IMPORTANCE_PATH \
                --old_model $PREV_MODEL \
                --new_model $EXPANDED_MODEL \
                --save_path $EXPANDED_IMPORTANCE_PATH \
                --copy_importance_init
            PREV_IMPORTANCE_PATH="$EXPANDED_IMPORTANCE_PATH"
        fi

        echo "Training task $task_num with pseudo_label + ESPReg + EWC..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \
            --data \"$ID_CONVERTED_DATASET/dataset.yaml\" \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \ \
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            TRAIN_CMD="$TRAIN_CMD --ewc True --importance_path $PREV_IMPORTANCE_PATH --ewc_loss_weight $EWC_LOSS_WEIGHT"
        fi
        eval $TRAIN_CMD

        echo "Calculating parameter importance..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \
            --model $TASK_DIR/best.pt \
            --dataset "$ID_CONVERTED_DATASET/dataset.yaml" \
            --save_path $IMPORTANCE_PATH \
            --module_pattern "*bn" \
            --batch_size $BATCH_SIZE \
            --workers $WORKERS \
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
