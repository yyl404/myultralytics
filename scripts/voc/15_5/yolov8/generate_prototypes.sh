python tools/generate_prototypes.py \
    --model runs/yolov8l_voc_inc_15_5_fromscratch_vspreg+pseudo_label+proto_rp/task-1/best.pt \
    --data data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
    --output runs/yolov8l_voc_inc_15_5_fromscratch_vspreg+pseudo_label+proto_rp/task-1/prototypes.pt \
    --vis_dir runs/yolov8l_voc_inc_15_5_fromscratch_vspreg+pseudo_label+proto_rp/task-1/prototypes-visualizations \
    --device 0