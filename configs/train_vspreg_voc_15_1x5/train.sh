# 训练基础模型
python tools/train.py --model yolov8l.yaml \
    --data data/VOC_inc_15_1x5/task_1_cls_15/dataset.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1

# 用前序任务模型生成增量数据集上的伪标签（一定要用前序任务模型而不是前序任务模型检测头扩展后的模型，否则新增的检测通道参数是随机初始化的，会生成错误的标签）
python tools/incremental_utils.py --create_pseudo_labels_dataset --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_pseudo_labels

# 扩展前序任务模型的检测通道
python tools/incremental_utils.py --expand_detection_head --data_cfg data/VOC_inc_15_1x5/task_2_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/best.pt \
    --model_cfg yolov8l.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/task-1-best_expanded.pt

# 转换伪标签数据集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_pseudo_labels/dataconfig.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/task-1-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train

# 转换完整验证集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC/VOC.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/task-1-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_val_fullset

# 用完整验证集替换伪标签数据集的验证集
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val

mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_val_fullset/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val
mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_val_fullset/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val

rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_val_fullset

# 用PCA分析前序任务模型各层的输入分布
python tools/pca.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/task-1-best_expanded.pt \
    --sample_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib --unfold

# 使用Antiforget训练器训练当前任务模型
python tools/train.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/task-1-best_expanded.pt \
    --data runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/dataconfig.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2 \
    --freeze [0,1,2,3,4,5,6,7,8,9] \
    --trainer antiforget \
    --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib

# 用前序任务模型生成增量数据集上的伪标签（一定要用前序任务模型而不是前序任务模型检测头扩展后的模型，否则新增的检测通道参数是随机初始化的，会生成错误的标签）
python tools/incremental_utils.py --create_pseudo_labels_dataset --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_pseudo_labels

# 扩展前序任务模型的检测通道
python tools/incremental_utils.py --expand_detection_head --data_cfg data/VOC_inc_15_1x5/task_3_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/best.pt \
    --model_cfg yolov8l.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/task-2-best_expanded.pt

# 转换伪标签数据集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_pseudo_labels/dataconfig.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/task-2-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train

# 转换完整验证集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC/VOC.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/task-2-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_val_fullset

# 用完整验证集替换伪标签数据集的验证集
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/images/val
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/labels/val

mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_val_fullset/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/images/val
mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_val_fullset/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/labels/val

rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_val_fullset

# 用PCA分析前序任务模型各层的输入分布
python tools/pca.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/task-2-best_expanded.pt \
    --sample_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib --unfold

# 使用Antiforget训练器训练当前任务模型
python tools/train.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/task-2-best_expanded.pt \
    --data runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/dataconfig.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3 \
    --freeze [0,1,2,3,4,5,6,7,8,9] \
    --trainer antiforget \
    --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib

# 用前序任务模型生成增量数据集上的伪标签（一定要用前序任务模型而不是前序任务模型检测头扩展后的模型，否则新增的检测通道参数是随机初始化的，会生成错误的标签）
python tools/incremental_utils.py --create_pseudo_labels_dataset --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_pseudo_labels

# 扩展前序任务模型的检测通道
python tools/incremental_utils.py --expand_detection_head --data_cfg data/VOC_inc_15_1x5/task_4_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/best.pt \
    --model_cfg yolov8l.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/task-3-best_expanded.pt

# 转换伪标签数据集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_pseudo_labels/dataconfig.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/task-3-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train

# 转换完整验证集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC/VOC.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/task-3-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_val_fullset

# 用完整验证集替换伪标签数据集的验证集
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/images/val
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/labels/val

mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_val_fullset/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/images/val
mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_val_fullset/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/labels/val

rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_val_fullset

# 用PCA分析前序任务模型各层的输入分布
python tools/pca.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/task-3-best_expanded.pt \
    --sample_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/labels/val \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib --unfold

# 使用Antiforget训练器训练当前任务模型
python tools/train.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/task-3-best_expanded.pt \
    --data runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/dataconfig.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4 \
    --freeze [0,1,2,3,4,5,6,7,8,9] \
    --trainer antiforget \
    --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib

# 用前序任务模型生成增量数据集上的伪标签（一定要用前序任务模型而不是前序任务模型检测头扩展后的模型，否则新增的检测通道参数是随机初始化的，会生成错误的标签）
python tools/incremental_utils.py --create_pseudo_labels_dataset --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_pseudo_labels

# 扩展前序任务模型的检测通道
python tools/incremental_utils.py --expand_detection_head --data_cfg data/VOC_inc_15_1x5/task_5_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/best.pt \
    --model_cfg yolov8l.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/task-4-best_expanded.pt

# 转换伪标签数据集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_pseudo_labels/dataconfig.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/task-4-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train

# 转换完整验证集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC/VOC.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/task-4-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_val_fullset

# 用完整验证集替换伪标签数据集的验证集
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/images/val
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/labels/val

mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_val_fullset/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/images/val
mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_val_fullset/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/labels/val

rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_val_fullset

# 用PCA分析前序任务模型各层的输入分布
python tools/pca.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/task-4-best_expanded.pt \
    --sample_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/labels/val \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib --unfold

# 使用Antiforget训练器训练当前任务模型
python tools/train.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/task-4-best_expanded.pt \
    --data runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/dataconfig.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5 \
    --freeze [0,1,2,3,4,5,6,7,8,9] \
    --trainer antiforget \
    --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib

# 用前序任务模型生成增量数据集上的伪标签（一定要用前序任务模型而不是前序任务模型检测头扩展后的模型，否则新增的检测通道参数是随机初始化的，会生成错误的标签）
python tools/incremental_utils.py --create_pseudo_labels_dataset --data_cfg data/VOC_inc_15_1x5/task_6_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_pseudo_labels

# 扩展前序任务模型的检测通道
python tools/incremental_utils.py --expand_detection_head --data_cfg data/VOC_inc_15_1x5/task_6_cls_1/dataset.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/best.pt \
    --model_cfg yolov8l.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/task-5-best_expanded.pt

# 转换伪标签数据集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_pseudo_labels/dataconfig.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/task-5-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train

# 转换完整验证集的类别ID
python tools/incremental_utils.py --convert_dataset_class_id --data_cfg data/VOC/VOC.yaml \
    --model_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/task-5-best_expanded.pt \
    --save_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_val_fullset

# 用完整验证集替换伪标签数据集的验证集
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train/images/val
rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train/labels/val

mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_val_fullset/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train/images/val
mv runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_val_fullset/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train/labels/val

rm -r runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_val_fullset

# 用PCA分析前序任务模型各层的输入分布
python tools/pca.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/task-5-best_expanded.pt \
    --sample_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/images/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/images/val \
    --label_dir runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-2/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-3/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-4/dataset_train/labels/val runs/yolov8l_voc_inc_15_1x5_fromscratch/task-5/dataset_train/labels/val \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib --unfold

# 使用Antiforget训练器训练当前任务模型
python tools/train.py --model runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/task-5-best_expanded.pt \
    --data runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/dataset_train/dataconfig.yaml \
    --save_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6/best.pt \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --workers 8 \
    --device 0 \
    --project runs/yolov8l_voc_inc_15_1x5_fromscratch/task-6 \
    --freeze [0,1,2,3,4,5,6,7,8,9] \
    --trainer antiforget \
    --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib