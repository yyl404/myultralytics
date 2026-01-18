TASK_DIR="runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-1"
DATASET_PATH="data/VOC_15_5/task_1_cls_15/dataset.yaml"
PCA_CACHE_PATH="runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-1/pca_cache.pkl"

python tools/pca.py \
    --model $TASK_DIR/best.pt \
    --dataset $DATASET_PATH \
    --save_path $PCA_CACHE_PATH