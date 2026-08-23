#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8l"
MODEL_CONFIG="yolov8l.yaml"
MODEL_WEIGHTS="yoloe-v8l-seg.pt"
DATASET_FAMILY="voc"
TASK_DATASETS=(
    "data/VOC_10+10/task_1_cls_10/dataset.yaml"
    "data/VOC_10+10/task_2_cls_10/dataset.yaml"
)
OUTPUT_PREFIX="runs/${MODEL_ID}_VOC_10+10_pretrained-from-${MODEL_WEIGHTS%.pt}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
