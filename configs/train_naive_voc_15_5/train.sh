# python tools/train_incremental.py --cfg configs/train_naive_voc_15_5/train_1.yaml --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-1
# python tools/pca.py --model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     --sample_dir data/VOC_inc_15_5/task_1_cls_15/images/train \
#     --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
#     --label_dir data/VOC_inc_15_5/task_1_cls_15/labels/train \
#     --sample_num 100 \
#     --layers 12 15 18 21 --unfold
python tools/proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-1/proj_vis
# python tools/model_compress.py --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     --model_cfg yolov8l.yaml \
#     --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
#     --save_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt \
#     --ratio 0.90 \
#     --layers 12 15 18 21 \
#     --sample_images data/VOC_inc_15_5/task_1_cls_15/images/val
# python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
#     --model_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt \
#     --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval-task-1-compress \
#     --target_dataset_name eval_dataset
# yolo detect val data=runs/yolov8l_voc_inc_15_5_fromscratch/eval-task-1-compress/eval_dataset/dataconfig.yaml \
#     model=runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt \
#     imgsz=640 device=0 \
#     conf=0.25 iou=0.6 \
#     project=runs/yolov8l_voc_inc_15_5_fromscratch/eval-task-1-compress
# yolo detect train data=runs/yolov8l_voc_inc_15_5_fromscratch/eval-task-1-compress/eval_dataset/dataconfig.yaml \
#     model=runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
#     epochs=10 imgsz=640 device=0 \
#     lr0=0.000001 lrf=0.000001 optimizer="SGD" \
#     freeze="[0,1,2,3,4,5,6,7,8,9]" \
#     project=runs/yolov8l_voc_inc_15_5_fromscratch/finetune-task-1-compress
# yolo detect train data=runs/yolov8l_voc_inc_15_5_fromscratch/eval-task-1-compress/eval_dataset/dataconfig.yaml \
#     model=runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt \
#     epochs=1 imgsz=640 device=0 \
#     lr0=1e-10 lrf=1e-10 optimizer="SGD" \
#     freeze="[0,1,2,3,4,5,6,7,8,9]" \
#     project=runs/yolov8l_voc_inc_15_5_fromscratch/finetune-task-1-compress

# python tools/train_incremental.py --cfg configs/train_naive_voc_15_5/train_2.yaml --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2
