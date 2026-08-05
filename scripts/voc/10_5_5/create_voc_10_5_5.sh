python tools/create_incremental_dataset.py \
    --source_cfg data/VOC-YOLO/VOC.yaml \
    --output_dir data/VOC_10+5+5 \
    --n_classes 10 5 5 \
    --workers "${WORKERS:-8}"
