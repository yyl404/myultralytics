TASK_DIR="runs/yolov8l_voc_15_5_fromscratch_pseudo_label+espreg/task-1"
# DATASET_PATH="data/VOC_15_5/task_1_cls_15/dataset.yaml"
SAMPLE_DIR="data/VOC_15_5/task_1_cls_15/images/train"
PCA_CACHE_PATH="runs/yolov8l_voc_15_5_fromscratch_pseudo_label+espreg/task-1/pca_cache.pkl"

python tools/pca.py \
    --model $TASK_DIR/best.pt \
    --save_path $PCA_CACHE_PATH \
    --sample_dir $SAMPLE_DIR \
    # --dataset $DATASET_PATH \