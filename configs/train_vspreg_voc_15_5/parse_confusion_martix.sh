python tools/parse_confusion_matrix.py \
    --confusion_matrix_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/confusion_matrix.csv \
    --old_classes aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person \
    --new_classes pottedplant sheep sofa train tvmonitor \
    --background_name background \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-1/confusion_matrix_analysis