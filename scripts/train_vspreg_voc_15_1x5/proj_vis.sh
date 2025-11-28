python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt --incremental_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/proj_vis_model_2-1

python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt --incremental_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/proj_vis_model_3-2

python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt --incremental_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/proj_vis_model_4-3

python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt --incremental_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/proj_vis_model_5-4

python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --incremental_model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/proj_vis_model_6-5