# bash scripts/coco/40_40/create_coco_40_40.sh
# START_TASK=2 bash scripts/coco/40_40/yolov8/train_nsgp+pseudo_label.sh
# bash scripts/coco/40_40/yolov8/eval.sh runs/yolov8l_coco_40_40_fromscratch_nsgp+pseudo_label

# bash scripts/coco/70_10/create_coco_70_10.sh
# START_TASK=2 bash scripts/coco/70_10/yolov8/train_nsgp+pseudo_label.sh
# bash scripts/coco/70_10/yolov8/eval.sh runs/yolov8l_coco_70_10_fromscratch_nsgp+pseudo_label

# bash scripts/coco/40_40/create_coco_40_40.sh
# bash scripts/coco/40_40/yolov8/generate_pca.sh
# START_TASK=2 bash scripts/coco/40_40/yolov8/train_espreg+pseudo_label.sh
# bash scripts/coco/40_40/yolov8/eval.sh runs/yolov8l_coco_40_40_fromscratch_ewpr+pseudo_label

# bash scripts/voc/15_5/create_voc_15_5.sh
START_TASK=2 bash scripts/voc/15_5/yolov8/train_nsgp+pseudo_label.sh
# bash scripts/voc/15_5/yolov8/eval.sh runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label
# START_TASK=2 bash scripts/voc/15_5/yolov8/train_espreg+pseudo_label.sh
# # bash scripts/voc/15_5/yolov8/eval.sh runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label