export model_1=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt
export model_2=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt
export model_3=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt
export model_4=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt
export model_5=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt
export model_6=runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt

# Model 1 evaluations (15 classes)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_1 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-1_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-1_task-1/eval_dataset/dataconfig.yaml \
    --model $model_1 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-1_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-1_task-1/results.csv \
    --confusion_matrix_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-1_task-1/confusion_matrix.csv

# Model 2 evaluations (16 classes: 15+1)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_2 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1/eval_dataset/dataconfig.yaml \
    --model $model_2 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path $model_2 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-2/eval_dataset/dataconfig.yaml \
    --model $model_2 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1-2_cls_16/dataset.yaml \
    --model_path $model_2 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1-2/eval_dataset/dataconfig.yaml \
    --model $model_2 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-2_task-1-2/results.csv

# Model 3 evaluations (17 classes: 15+1+1)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_3 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1/eval_dataset/dataconfig.yaml \
    --model $model_3 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path $model_3 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-2/eval_dataset/dataconfig.yaml \
    --model $model_3 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path $model_3 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-3/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-3/eval_dataset/dataconfig.yaml \
    --model $model_3 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-3 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-3/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1-3_cls_17/dataset.yaml \
    --model_path $model_3 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1-3/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1-3/eval_dataset/dataconfig.yaml \
    --model $model_3 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1-3 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-3_task-1-3/results.csv

# Model 4 evaluations (18 classes: 15+1+1+1)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_4 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1/eval_dataset/dataconfig.yaml \
    --model $model_4 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path $model_4 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-2/eval_dataset/dataconfig.yaml \
    --model $model_4 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path $model_4 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-3/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-3/eval_dataset/dataconfig.yaml \
    --model $model_4 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-3 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-3/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path $model_4 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-4/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-4/eval_dataset/dataconfig.yaml \
    --model $model_4 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-4 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-4/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1-4_cls_18/dataset.yaml \
    --model_path $model_4 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1-4/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1-4/eval_dataset/dataconfig.yaml \
    --model $model_4 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1-4 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-4_task-1-4/results.csv

# Model 5 evaluations (19 classes: 15+1+1+1+1)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-2/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-3/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-3/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-3 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-3/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-4/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-4/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-4 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-4/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-5/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-5/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-5 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-5/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1-5_cls_19/dataset.yaml \
    --model_path $model_5 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1-5/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1-5/eval_dataset/dataconfig.yaml \
    --model $model_5 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1-5 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-5_task-1-5/results.csv

# Model 6 evaluations (20 classes: 15+1+1+1+1+1)
python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-2/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-2/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-2 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-2/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-3/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-3/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-3 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-3/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-4/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-4/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-4 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-4/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-5/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-5/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-5 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-5/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_6_cls_1/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-6/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-6/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-6 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-6/results.csv

python tools/incremental_utils.py --convert_dataset_class_id \
    --data_cfg data/VOC_inc_15_1x5/task_1-6_cls_20/dataset.yaml \
    --model_path $model_6 --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1-6/eval_dataset
python tools/eval.py --data runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1-6/eval_dataset/dataconfig.yaml \
    --model $model_6 --project runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1-6 \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/eval_model-6_task-1-6/results.csv
