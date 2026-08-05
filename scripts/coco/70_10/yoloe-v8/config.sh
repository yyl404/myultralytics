#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8l"
MODEL_CONFIG="yolov8l.yaml"
MODEL_WEIGHTS="yoloe-v8l-seg.pt"
DATASET_FAMILY="coco"
TASK_DATASETS=(
    "data/COCO_70+10/task_1_cls_70/dataset.yaml"
    "data/COCO_70+10/task_2_cls_10/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolov8l_COCO_70+10"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
