# python tools/train_incremental.py --cfg configs/train_naive_voc_15_5/train_1.yaml --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-1
python tools/generate_osr_memory_bank.py --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataconfig.yaml --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2/osr_memory_bank --model_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt
python tools/train_incremental.py --cfg configs/train_naive_voc_15_5/train_2.yaml --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/task-2
