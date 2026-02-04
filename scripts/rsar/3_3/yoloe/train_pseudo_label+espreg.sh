#!/bin/bash
MODEL_CFG="yolov8l-obb.yaml"

YOLOE_MODEL_WEIGHT="yoloe-v8l-seg.pt"

FREEZE_LAYERS=(
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
)
OUTPUT_DIR="runs/yolov8l_rsar_3_3_pretrained-yoloe_pseudo_label+espreg"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
ESPREG_LOSS_WEIGHT=${ESPREG_LOSS_WEIGHT:-1000.0}
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
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    if [ ! -f "$PREV_PCA_CACHE" ]; then
        echo "Warning: Previous task PCA cache not found."
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

        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \
            --model $TASK_DIR/best.pt \
            --dataset $DATASET_PATH \
            --save_path $PCA_CACHE_PATH
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
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

        echo "Training task $task_num with pseudo_label + ESPReg..."
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
            --freeze ${FREEZE_LAYERS[$((task_num-1))]} \
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        TRAIN_CMD="$TRAIN_CMD --espreg True --pca_cache_path $PREV_PCA_CACHE --espreg_loss_weight $ESPREG_LOSS_WEIGHT"
        eval $TRAIN_CMD

        echo "Performing PCA analysis..."
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
    ((task_num++))
done
echo "All tasks completed!"
