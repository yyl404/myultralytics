TASK_DIR="runs/yolov8l_voc_15_5_fromscratch_ewc+pseudo_label/task-1"
DATASET_PATH="data/VOC_15_5/task_1_cls_15/dataset.yaml"
IMPORTANCE_PATH="runs/yolov8l_voc_15_5_fromscratch_ewc+pseudo_label/task-1/importance.pth"
BATCH_SIZE=4
WORKERS=8
DEVICE=0

python tools/cal_importance.py \
    --model $TASK_DIR/best.pt \
    --dataset $DATASET_PATH \
    --save_path $IMPORTANCE_PATH \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --device $DEVICE

