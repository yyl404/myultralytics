#!/bin/bash

# Configuration
BASE_MODEL="runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/task-1-best-expanded.pt"
INCREMENTAL_MODEL="runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/train2/weights/last.pt"
PCA_CACHE="runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-1/pca_cache.pkl"
SAVE_DIR="runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/proj_visualizations"
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
