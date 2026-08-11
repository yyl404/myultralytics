#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolo26x"
MODEL_CONFIG="yolo26x.yaml"
MODEL_WEIGHTS="yolo26x.pt"
DATASET_FAMILY="voc"
TASK_DATASETS=(
    "data/VOC_15+5/task_1_cls_15/dataset.yaml"
    "data/VOC_15+5/task_2_cls_5/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolo26x_VOC_15+5_pretrained-from-yolo26x"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
