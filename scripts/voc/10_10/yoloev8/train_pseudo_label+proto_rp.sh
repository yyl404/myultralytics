#!/bin/bash
MODEL_CFG="yolov8l.yaml"

FREEZE_LAYERS=(
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]"
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]"
)
OUTPUT_DIR="runs/yolov8l_voc_10_10_pretrained-yoloe_pseudo_label+proto_rp"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
PROTO_RP_USE_BASE_MODEL=${PROTO_RP_USE_BASE_MODEL:-True}
START_TASK=${START_TASK:-1}

TASK_DATASETS=(
    "data/VOC_inc_10_10/task_1_cls_10/dataset.yaml"
    "data/VOC_inc_10_10/task_2_cls_10/dataset.yaml"
)

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${#TASK_DATASETS[@]} ]; then
    echo "Error: START_TASK must be between 1 and ${#TASK_DATASETS[@]}"
    exit 1
fi

PREV_PROTOTYPES=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PROTOTYPES="$OUTPUT_DIR/task-$PREV_TASK/prototypes.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
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

        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
    else
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \
            --model $PREV_MODEL \
            --model_cfg $MODEL_CFG \
            --dataset $DATASET_PATH \
            --save_path $EXPANDED_MODEL

        CONVERTED_PROTOTYPES=""
        if [ -n "$PREV_PROTOTYPES" ] && [ -f "$PREV_PROTOTYPES" ]; then
            echo "Converting prototype classes..."
            CONVERTED_PROTOTYPES="$TASK_DIR/task-$((task_num-1))-prototypes-converted.pt"
            python tools/convert_prototype_classes.py \
                --prototypes $PREV_PROTOTYPES \
                --original_model $PREV_MODEL \
                --expanded_model $EXPANDED_MODEL \
                --output $CONVERTED_PROTOTYPES
        fi

        echo "Generating pseudo labels..."
        PSEUDO_LABELS_DIR="$TASK_DIR/${DATASET_NAME}_train_pseudo_labels"
        python tools/generate_pseudo_label.py \
            --model $PREV_MODEL \
            --dataset $DATASET_PATH \
            --output_dir $PSEUDO_LABELS_DIR \
            --conf_threshold $CONF_THRESHOLD \
            --splits train
        echo "Merging dataset..."
        MERGED_DATASET_DIR="$TASK_DIR/${DATASET_NAME}_merged"
        python tools/merge_datasets.py \
            --datasets "$PSEUDO_LABELS_DIR/dataset.yaml" "$DATASET_PATH" \
            --output_dir $MERGED_DATASET_DIR
        echo "Converting dataset class IDs..."
        CONVERTED_DATASET="$TASK_DIR/${DATASET_NAME}_converted"
        python tools/convert_dataset_class_ids.py \
            --model $EXPANDED_MODEL \
            --dataset $MERGED_DATASET_DIR/dataset.yaml \
            --output_dir $CONVERTED_DATASET

        echo "Training task $task_num with pseudo_label + proto_rp..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \
            --data \"$CONVERTED_DATASET/dataset.yaml\" \
            --save_path $TASK_DIR/best.pt \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --imgsz $IMGSZ \
            --workers $WORKERS \
            --device $DEVICE \
            --project $TASK_DIR \ \
            --freeze ${FREEZE_LAYERS[$((task_num-1))]} \
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold 0.5"
        if [ -n "$CONVERTED_PROTOTYPES" ] && [ -f "$CONVERTED_PROTOTYPES" ]; then
            TRAIN_CMD="$TRAIN_CMD --prototypes $CONVERTED_PROTOTYPES --proto_rp_use_base_model $PROTO_RP_USE_BASE_MODEL"
        fi
        eval $TRAIN_CMD

        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        python tools/generate_prototypes.py \
            --model $TASK_DIR/best.pt \
            --data $DATASET_PATH \
            --output $PROTOTYPES_PATH \
            --load_hits $CONVERTED_PROTOTYPES \
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
