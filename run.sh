# bash scripts/voc/15_5/yolov8/generate_importance.sh
# EWPR_LOSS_WEIGHT=100 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewc+pseudo_label.sh
# EWPR_LOSS_WEIGHT=1 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
# EWPR_LOSS_WEIGHT=10 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
# EWPR_LOSS_WEIGHT=100 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
# EWPR_LOSS_WEIGHT=10000 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
# EWPR_LOSS_WEIGHT=100000 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
# EWPR_LOSS_WEIGHT=1000000 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
EWPR_LOSS_WEIGHT=100 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
EWPR_LOSS_WEIGHT=1e3 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
EWPR_LOSS_WEIGHT=1e4 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
EWPR_LOSS_WEIGHT=1e5 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh
EWPR_LOSS_WEIGHT=1e6 START_TASK=2 bash scripts/voc/15_5/yolov8/train_ewpr+pseudo_label.sh