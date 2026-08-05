#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8l"
MODEL_CONFIG="yolov8l.yaml"
MODEL_WEIGHTS="yoloe-v8l-seg.pt"
DATASET_FAMILY="voc"
TASK_DATASETS=(
    "data/VOC_19+1/task_1_cls_19/dataset.yaml"
    "data/VOC_19+1/task_2_cls_1/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolov8l_VOC_19+1_pretrained-from-yoloe-v8l-seg"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
