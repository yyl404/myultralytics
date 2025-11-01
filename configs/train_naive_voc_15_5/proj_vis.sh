# python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
#     --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1_pseudolabels_novsp \
#     --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/train-pseudolabels-novsp/weights/last.pt \
#     --sample_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/images/val \
#     --label_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/labels/val \
#     --unfold --stage 3 --sample_num 1 --xlim_main "0,1.4" --xlim_kernel_update "0,0.01"

# python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
#     --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1_pseudolabels_vsp_verybig \
#     --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/train-pseudolabels-vsp-verybig/weights/last.pt \
#     --sample_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/images/val \
#     --label_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/labels/val \
#     --unfold --stage 3 --sample_num 1 --xlim_main "0,1.4" --xlim_kernel_update "0,0.01"

python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache_all-layers.joblib \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1_pseudolabels_vsp_verybig_pca_alllayers \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/task-1-best_expanded.pt \
    --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/train-pseudolabels-vsp-verybig-pca-alllayers/weights/last.pt \
    --sample_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/dataset_train/labels/val \
    --unfold --stage 3 --sample_num 1 --xlim_main "0,1.4" --xlim_kernel_update "0,0.01"

# python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
#     --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_1-1 \
#     --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     --unfold