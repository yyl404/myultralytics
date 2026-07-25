#!/bin/bash
set -euo pipefail
python tools/create_incremental_dataset.py \
    --source_cfg data/coco-yolo/coco.yaml \
    --output_dir data/COCO_40+40 \
    --n_classes 40 40 \
    --workers "${WORKERS:-8}"
