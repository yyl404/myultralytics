export model_1=runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt
export model_2=runs/yolov8l_voc_inc_15_5_fromscratch/task-2/train/weights/last.pt

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
    --model_path $model_1 --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/eval_dataset/dataconfig.yaml \
    --model $model_1 --project runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1 \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/results.csv \
    --confusion_matrix_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/confusion_matrix.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_5/task_1_cls_15/dataset.yaml \
    --model_path $model_2 --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1/eval_dataset/dataconfig.yaml \
    --model $model_2 --project runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1 \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_5/task_2_cls_5/dataset.yaml \
    --model_path $model_2 --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-2/eval_dataset/dataconfig.yaml \
    --model $model_2 --project runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-2 \
    --save_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC/VOC.yaml \
    --model_path $model_2 --save_dir "runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1-2/eval_dataset"
python tools/eval.py --data "runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1-2/eval_dataset/dataconfig.yaml" \
    --model $model_2 --project "runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1-2" \
    --save_path "runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-2_task-1-2/results.csv"