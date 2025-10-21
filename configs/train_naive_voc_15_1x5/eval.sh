# Task 1 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
   --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval-task-1-1

# Task 2 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-2-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-2-2

# Task 3 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-3-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-3-2
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-3-3

# Task 4 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-4-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-4-2
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-4-3
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-4-4

# Task 5 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-5-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-5-2
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-5-3
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-5-4
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-5-5

# Task 6 evaluations
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-1
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-2
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-3
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-4
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-5
python tools/eval_incremental.py --data_cfg data/VOC_inc_15_1x5/task_6_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_task-6-6
