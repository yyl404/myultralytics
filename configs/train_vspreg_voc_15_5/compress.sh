python tools/model_compress.py \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
    --base_model_cfg yolov8l.yaml \
    --device cuda \
    --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed_fuse.pt \
    --ratio 0.9 \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --sample_images data/VOC_inc_15_5/task_1_cls_15/images/val \
    --sample_num 100 \
    --fuse

python tools/model_compress.py \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
    --base_model_cfg yolov8l.yaml \
    --device cuda \
    --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt \
    --ratio 0.9 \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --sample_images data/VOC_inc_15_5/task_1_cls_15/images/val \
    --sample_num 100

python tools/model_compress.py \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
    --base_model_cfg yolov8l.yaml \
    --device cuda \
    --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best_compressed_fuse.pt \
    --ratio 0.9 \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --sample_images data/VOC_inc_15_5/task_1_cls_15/images/val \
    --sample_num 100 \
    --fuse

python tools/model_compress.py \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
    --base_model_cfg yolov8l.yaml \
    --device cuda \
    --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best_compressed.pt \
    --ratio 0.9 \
    --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --sample_images data/VOC_inc_15_5/task_1_cls_15/images/val \
    --sample_num 100