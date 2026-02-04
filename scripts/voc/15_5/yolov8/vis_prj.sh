#!/bin/bash

# Configuration
# BASE_MODEL="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_pseudo_label/task-2/task-1-best-expanded.pt"
# INCREMENTAL_MODEL="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_pseudo_label/task-2/best.pt"
# PCA_CACHE="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-1/pca_cache.pkl"
# SAVE_DIR="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_pseudo_label/task-2/proj_visualizations"
BASE_MODEL="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/task-1-best-expanded.pt"
INCREMENTAL_MODEL="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/train-w100/weights/best.pt"
PCA_CACHE="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-1/pca_cache.pkl"
SAVE_DIR="/root/hy-tmp/runs_old/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/proj_visualizations"
DEVICE="cuda"
LAYERS=""

# Execute
python tools/vis_kernel_proj_pc.py \
    --base_model "$BASE_MODEL" \
    --incremental_model "$INCREMENTAL_MODEL" \
    --pca_cache "$PCA_CACHE" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    ${LAYERS:+--layers "$LAYERS"}
