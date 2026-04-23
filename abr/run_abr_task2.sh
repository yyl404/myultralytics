#!/bin/bash
set -e

OUTPUT_DIR="runs/yolov8l_voc_15_5_fromscratch_abr"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10

TASK2_DIR="$OUTPUT_DIR/task-2"
EXPANDED_MODEL="$TASK2_DIR/task-1-best-expanded.pt"
CONVERTED_DATASET="$TASK2_DIR/task_2_cls_5_converted"

echo "=========================================="
echo "Check required files for task 2"
echo "=========================================="

ls $TASK2_DIR/memory/memory.json
ls $EXPANDED_MODEL
ls $CONVERTED_DATASET/dataset.yaml

echo "=========================================="
echo "Task 2: train with ABR replay"
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
  --mosaic 0.0 \
  --mixup 0.0 \
  --copy_paste 0.0