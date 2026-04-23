#!/bin/bash
set -e

MODEL_CFG="yolov8x.yaml"
PRETRAIN_CLS="yolov8x-cls.pt"
OUTPUT_DIR="runs/yolov8x_voc_19_1_fromcls_osr_pseudo_label"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10

CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5

TASK1_DATASET="data/VOC_19_1/task_1_cls_19/dataset.yaml"
TASK2_DATASET="data/VOC_19_1/task_2_cls_1/dataset.yaml"

TASK1_DIR="$OUTPUT_DIR/task-1"
TASK2_DIR="$OUTPUT_DIR/task-2"

echo "================ Task 1 ================="
python tools/train.py \
  --model $MODEL_CFG \
  --weight $PRETRAIN_CLS \
  --data $TASK1_DATASET \
  --save_path $TASK1_DIR/best.pt \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --imgsz $IMGSZ \
  --workers $WORKERS \
  --device $DEVICE \
  --project $TASK1_DIR \
  --patience $PATIENCE

PREV_MODEL="$TASK1_DIR/best.pt"

echo "================ Expand Head ================="
EXPANDED_MODEL="$TASK2_DIR/task-1-best-expanded.pt"
python tools/expand_model_head.py \
  --model $PREV_MODEL \
  --model_cfg $MODEL_CFG \
  --dataset $TASK2_DATASET \
  --save_path $EXPANDED_MODEL

echo "================ Convert Class IDs ================="
CONVERTED_DATASET="$TASK2_DIR/task_2_cls_1_converted"
python tools/convert_dataset_class_ids.py \
  --model $EXPANDED_MODEL \
  --dataset $TASK2_DATASET \
  --output_dir $CONVERTED_DATASET

echo "================ OSR Memory Bank ================="
MEMORY_BANK="$TASK2_DIR/osr_memory_bank"
python tools/osr.py \
  --generate_memory_bank \
  --base_dataset_cfg $TASK1_DATASET \
  --memory_bank_dir $MEMORY_BANK \
  --model_path $PREV_MODEL \
  --k 1

echo "================ OSR Object Aug ================="
OSR_CP="$TASK2_DIR/task_2_cls_1_osr_cp"
python tools/osr.py \
  --copy_paste_replay \
  --new_dataset_cfg "$CONVERTED_DATASET/dataset.yaml" \
  --memory_bank_dir $MEMORY_BANK \
  --save_dir $OSR_CP \
  --split train

echo "================ OSR Feature Aug ================="
OSR_FA="$TASK2_DIR/task_2_cls_1_osr_fa"
python tools/osr.py \
  --feature_augmentation_replay \
  --new_dataset_cfg "$CONVERTED_DATASET/dataset.yaml" \
  --memory_bank_dir $MEMORY_BANK \
  --save_dir $OSR_FA \
  --split train \
  --num_generations 0 \
  --mixup_alpha 1.0

echo "================ Merge Datasets ================="
MERGED_DATASET="$TASK2_DIR/task_2_cls_1_osr_full"
python tools/merge_datasets.py \
  --datasets "$CONVERTED_DATASET/dataset.yaml" "$OSR_CP/dataset.yaml" "$OSR_FA/dataset.yaml" \
  --output_dir $MERGED_DATASET

echo "================ Task 2 Train ================="
python tools/train.py \
  --model $EXPANDED_MODEL \
  --data "$MERGED_DATASET/dataset.yaml" \
  --save_path $TASK2_DIR/best.pt \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --imgsz $IMGSZ \
  --workers $WORKERS \
  --device $DEVICE \
  --project $TASK2_DIR \
  --patience $PATIENCE \
  --trainer antiforget \
  --pseudo_label True \
  --conf_threshold $CONF_THRESHOLD \
  --filter_iou_threshold $FILTER_IOU_THRESHOLD \
  --mosaic 0.0 \
  --mixup 0.0 \
  --copy_paste 0.0