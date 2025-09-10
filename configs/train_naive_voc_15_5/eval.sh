python tools/eval_incremental.py --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_5_fromscratch_naive/task-1/best.pt --save_dir runs/yolov8l_voc_inc_15_5_fromscratch_naive/eval-task-1-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_5_fromscratch_naive/task-2/best.pt --save_dir runs/yolov8l_voc_inc_15_5_fromscratch_naive/eval_task-2-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_5/task_2_cls_5/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_5_fromscratch_naive/task-2/best.pt --save_dir runs/yolov8l_voc_inc_15_5_fromscratch_naive/eval_task-2-2