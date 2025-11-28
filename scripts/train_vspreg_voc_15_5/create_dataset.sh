# python tools/create_incremental_dataset.py \
#     --source_cfg data/VOC/VOC.yaml \
#     --output_dir data/VOC_inc_15_5 \
#     --n_classes 15 5

# python tools/match_and_pick_dataset.py \
#     --dataset_a data/VOC/VOC.yaml \
#     --dataset_b data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
#     --save_dir data/VOC_inc_15_5/task_1_cls_15_full-labels

# python tools/match_and_pick_dataset.py \
#     --dataset_a data/VOC/VOC.yaml \
#     --dataset_b data/VOC_inc_15_5/task_2_cls_5/dataset.yaml \
#     --save_dir data/VOC_inc_15_5/task_2_cls_5_full-labels