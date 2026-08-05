#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8l"
MODEL_CONFIG="yolov8l.yaml"
MODEL_WEIGHTS="yoloe-v8l-seg.pt"
DATASET_FAMILY="voc"
TASK_DATASETS=(
    "data/VOC-TINY_15+5/task_1_cls_15/dataset.yaml"
    "data/VOC-TINY_15+5/task_2_cls_5/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolov8l_VOC-TINY_15+5_pretrained-from-yoloe-v8l-seg"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
