#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8x"
MODEL_CONFIG="yolov8x.yaml"
MODEL_WEIGHTS="yolov8x-cls.pt"
DATASET_FAMILY="coco"
TASK_DATASETS=(
    "data/COCO_40+40/task_1_cls_40/dataset.yaml"
    "data/COCO_40+40/task_2_cls_40/dataset.yaml"
)
OUTPUT_PREFIX="runs/${MODEL_ID}_COCO_40+40"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
