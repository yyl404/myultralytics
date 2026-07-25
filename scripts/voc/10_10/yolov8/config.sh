#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8x"
MODEL_CONFIG="yolov8x.yaml"
MODEL_WEIGHTS="yolov8x-cls.pt"
DATASET_FAMILY="voc"
TASK_DATASETS=(
    "data/VOC_10+10/task_1_cls_10/dataset.yaml"
    "data/VOC_10+10/task_2_cls_10/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolov8x_VOC_10+10_pretrained-from-yolov8x-cls"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
