python tools/create_incremental_dataset.py \
    --source_cfg data/VOC-YOLO/VOC.yaml \
    --output_dir data/VOC_10+2+2+2+2+2 \
    --n_classes 10 2 2 2 2 2 \
    --workers "${WORKERS:-8}"