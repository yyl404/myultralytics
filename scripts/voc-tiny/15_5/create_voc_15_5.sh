python tools/create_incremental_dataset.py \
    --source_cfg data/VOC-TINY-YOLO/VOC.yaml \
    --output_dir data/VOC-TINY_15+5 \
    --n_classes 15 5 \
    --workers "${WORKERS:-8}"