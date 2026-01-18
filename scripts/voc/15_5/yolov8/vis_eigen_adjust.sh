#!/bin/bash

# Configuration
PCA_CACHE="runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-1/pca_cache.pkl"
SAVE_DIR="runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/eigen_adjust_visualizations"

# Execute
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../../../" && \
python tools/vis_eigen_adjust.py \
    --pca_cache "$PCA_CACHE" \
    --save_dir "$SAVE_DIR"

