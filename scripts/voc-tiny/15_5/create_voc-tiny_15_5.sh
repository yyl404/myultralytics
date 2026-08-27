#!/bin/bash
set -euo pipefail

# VOC-TINY: a seeded 10% subsample of the full VOC dataset (symlinks, no extra disk cost)
python tools/subsample_dataset.py \
    --source_cfg data/VOC-YOLO/VOC.yaml \
    --output_dir data/VOC-TINY-YOLO \
    --fraction "${TINY_FRACTION:-0.1}" \
    --seed "${SEED:-0}"

python tools/create_incremental_dataset.py \
    --source_cfg data/VOC-TINY-YOLO/VOC.yaml \
    --output_dir data/VOC-TINY_15+5 \
    --n_classes 15 5 \
    --overwrite \
    --workers "${WORKERS:-8}"
