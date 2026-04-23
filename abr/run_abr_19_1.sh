#!/bin/bash
set -e

MODEL_CFG="yolov8x.yaml"
PRETRAIN_CLS="yolov8x-cls.pt"
OUTPUT_DIR="runs/yolov8x_voc_19_1_fromcls_abr_pseudo_label"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=1
PATIENCE=10

CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5

TASK1_DATASET="data/VOC_19_1/task_1_cls_19/dataset.yaml"
TASK2_DATASET="data/VOC_19_1/task_2_cls_1/dataset.yaml"

TASK1_DIR="$OUTPUT_DIR/task-1"
TASK2_DIR="$OUTPUT_DIR/task-2"

echo "=========================================="
echo "Task 1: train old classes with yolov8x-cls init"
echo "=========================================="

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

echo "=========================================="
echo "Build ABR memory from task 1"
echo "=========================================="

python tools/build_abr_memory.py \
  --data $TASK1_DATASET \
  --output $TASK2_DIR/memory \
  --memory_size 2000 \
  --old_class_num 19 \
  --seed 0

echo "=========================================="
echo "Expand model head for task 2"
echo "=========================================="

EXPANDED_MODEL="$TASK2_DIR/task-1-best-expanded.pt"
python tools/expand_model_head.py \
  --model $PREV_MODEL \
  --model_cfg $MODEL_CFG \
  --dataset $TASK2_DATASET \
  --save_path $EXPANDED_MODEL

echo "=========================================="
echo "Convert task 2 dataset class IDs"
echo "=========================================="

CONVERTED_DATASET="$TASK2_DIR/task_2_cls_1_converted"
python tools/convert_dataset_class_ids.py \
  --model $EXPANDED_MODEL \
  --dataset $TASK2_DATASET \
  --output_dir $CONVERTED_DATASET

echo "=========================================="
echo "Task 2: train with ABR replay + pseudo label"
echo "=========================================="

python tools/train.py \
  --model $EXPANDED_MODEL \
  --data "$CONVERTED_DATASET/dataset.yaml" \
  --save_path $TASK2_DIR/best.pt \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --imgsz $IMGSZ \
  --workers $WORKERS \
  --device $DEVICE \
  --project $TASK2_DIR \
  --patience $PATIENCE \
  --trainer abr \
  --abr \
  --abr_memory $TASK2_DIR/memory/memory.json \
  --abr_ratio '[1,1,2]' \
  --abr_iou_thr 0.05 \
  --abr_max_mix_boxes 2 \
  --abr_mix_beta 32.0 \
  --abr_mosaic_scale '[0.4,0.6]' \
  --pseudo_label True \
  --conf_threshold $CONF_THRESHOLD \
  --filter_iou_threshold $FILTER_IOU_THRESHOLD \
  --mosaic 0.0 \
  --mixup 0.0 \
  --copy_paste 0.0