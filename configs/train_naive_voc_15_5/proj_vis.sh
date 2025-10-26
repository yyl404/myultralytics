python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
    --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1