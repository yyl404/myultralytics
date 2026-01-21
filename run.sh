# bash scripts/voc/15_5/yolov8/generate_importance.sh
EWC_LOSS_WEIGHT=1 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh
EWC_LOSS_WEIGHT=10 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh
EWC_LOSS_WEIGHT=100 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh
EWC_LOSS_WEIGHT=1000 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh
EWC_LOSS_WEIGHT=10000 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh