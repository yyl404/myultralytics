python tools/generate_prototypes.py \
    --model runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label+espreg+proto_rp/task-1/best.pt \
    --data data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
    --output runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label+espreg+proto_rp/task-1/prototypes.pt \
    --vis_dir runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label+espreg+proto_rp/task-1/prototypes-visualizations \
    --device 0