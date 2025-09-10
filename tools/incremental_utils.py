import os
import shutil
from tqdm import tqdm

from torch.nn import Sequential

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.nn.tasks import yaml_model_load, DetectionModel, Detect


def convert_class_id(label_lines, class_id_map):
    """转换标注文件中的类别ID"""
    converted_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_cat_id = int(parts[0])
            if old_cat_id in class_id_map:
                new_cat_id = class_id_map[old_cat_id]
                parts[0] = str(new_cat_id)
                converted_lines.append(' '.join(parts) + '\n')
    return converted_lines


def read_labels_and_convert_class_id(labels_dir, class_id_map):
    """读取一个目录下所有的标注文件并转换类别ID
    
    Args:
        labels_dir: 标注文件目录路径
        class_id_map: 原始数据集类别ID到目标数据集类别ID的映射 {old_id: new_id}
    
    Returns:
        转换后的标注文件字典 {label_file: converted_label_lines}
    """
    labels = {}
    if os.path.exists(labels_dir):
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                labels[label_file] = convert_class_id(lines, class_id_map)
    return labels


def save_labels(labels, output_dir):
    """保存标注文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for label_file, lines in labels.items():
        output_path = os.path.join(output_dir, label_file)
        with open(output_path, 'w') as f:
            f.writelines(lines)


def create_classes_expanded_dataset(cfg_source, class_id_map, save_dir, target_dataset_name, target_dataset_classes):
    """
    创建类别列表扩展后的数据集，用于在增量学习训练和评估流程中将原始数据集的类别列表扩展至与模型输出类别列表一致
    
    Args:
        cfg_source: 原始数据集配置文件路径
        class_id_map: 原始数据集类别ID到目标数据集类别ID的映射 {old_id: new_id}
        save_dir: 目标数据集和中间文件的保存目录
        target_dataset_name: 目标数据集目录名称
        target_dataset_classes: 目标数据集类别列表
    
    Returns:
        生成的目标数据集配置文件路径
    """
    # 读取原始数据集配置
    source_dataset_dir = os.path.dirname(cfg_source)
    cfg_source = YAML.load(cfg_source)
    
    # 创建目标数据集目录
    target_dataset_dir = os.path.abspath(os.path.join(save_dir, target_dataset_name))
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.makedirs(target_dataset_dir, exist_ok=True)
    
    # 复制图像文件并转换标签
    for split in ['train', 'val', 'test']:
        if split in cfg_source:
            # 原始数据集图像目录
            source_images = os.path.join(cfg_source['path'], cfg_source[split]) if 'path' in cfg_source.keys() else \
                os.path.join(source_dataset_dir, cfg_source[split])
            
            # 目标图像目录
            target_images = os.path.join(target_dataset_dir, f"images/{split}")
            os.makedirs(target_images, exist_ok=True)
            shutil.copytree(source_images, target_images, dirs_exist_ok=True)
            
            # 原始数据集标签目录
            source_labels = os.path.join(cfg_source['path'], cfg_source[split].replace('images', 'labels')) if 'path' in cfg_source.keys() else \
                os.path.join(source_dataset_dir, cfg_source[split].replace('images', 'labels'))
            
            # 目标标签目录
            target_labels = os.path.join(target_dataset_dir, f"labels/{split}")
            os.makedirs(target_labels, exist_ok=True)
            
            # 转换标签的类别ID并保存
            if os.path.exists(source_labels):
                converted_labels = read_labels_and_convert_class_id(source_labels, class_id_map)
                save_labels(converted_labels, target_labels)
    
    # 创建目标数据集的配置文件
    config = {
        'names': {i: cls for i, cls in enumerate(target_dataset_classes)}
    }
    for split in ['train', 'val', 'test']:
        if split in cfg_source:
            config[split] = f"images/{split}"
    config_path = os.path.join(target_dataset_dir, 'dataconfig.yaml')
    YAML.save(data=config, file=config_path)
    return config_path


def expand_detection_head(ckpt_path, model_cfg, channel_map, classes_names, save_dir, output_name):
    """扩充模型检测头的输出通道数，为新增类别分配新通道，同时将原本检测头的权重迁移到对应通道的权重
    
    Args:
        ckpt_path: 模型权重路径
        model_cfg: 模型结构配置
        channel_map: 通道映射(旧模型通道到新模型对应通道的映射表)
        classes_names: 类别名称列表
        save_dir: 保存目录
        output_name: 输出名称
    """
    model = YOLO(ckpt_path)
    assert isinstance(model.model, DetectionModel) and isinstance(model.model.model, Sequential)\
        and isinstance(model.model.model[-1], Detect), "Only support DetectionModel with Detect in the last layer"
    weight = model.model.state_dict()
    
    model_name = model_cfg.split(".")[0]
    model_cfg = yaml_model_load(model_cfg)
    model_cfg["nc"] = len(classes_names)
    YAML.save(data=model_cfg, file=os.path.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    new_model = YOLO(os.path.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    new_weight = new_model.model.state_dict()

    # 权重迁移：分类层按映射迁移，其他层直接复制
    for key in new_weight.keys():
        if key in weight:
            layer_id = int(key.split('.')[1])
            # 处理cv3中最后的分类层权重（Conv2d层，即.2.weight）
            if layer_id == len(model.model.model) - 1 and 'cv3' in key and key.endswith('.2.weight'):
                print(f"Migrating weight for {key}")
                print(weight[key].shape)
                print(new_weight[key].shape)
                # 根据transfer_map迁移权重
                for old_idx, new_idx in channel_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # 处理cv3中最后的分类层偏置（Conv2d层，即.2.bias）
            elif layer_id == len(model.model.model) - 1 and 'cv3' in key and key.endswith('.2.bias'):
                # 根据transfer_map迁移偏置
                for old_idx, new_idx in channel_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # 其他层直接复制（形状相同）
            else:
                new_weight[key] = weight[key].clone()

    new_model.model.load_state_dict(new_weight)
    new_model.model.names = {k: v for k, v in enumerate(classes_names)}
    new_model.save(os.path.join(save_dir, output_name))


def merge_labels(original_labels, pseudo_labels):
    """合并标注文件，用于在基于蒸馏的增量学习方法训练和评估流程中，合并原始标注和伪标注
    
    Args:
        original_labels: 原始标注文件
        pseudo_labels: 伪标注文件
    
    Returns:
        合并后的标注文件 {label_file: merged_label_lines}
    """
    merged_labels = {}
    all_files = set(original_labels.keys()) | set(pseudo_labels.keys())
    
    for label_file in all_files:
        merged_lines = []
        
        # 添加原始标注
        if label_file in original_labels:
            merged_lines.extend(original_labels[label_file])
        
        # 添加伪标注
        if label_file in pseudo_labels:
            merged_lines.extend(pseudo_labels[label_file])
        
        merged_labels[label_file] = merged_lines
    
    return merged_labels


def create_pseudo_labels_dataset(teacher_model, base_class_id_map, new_class_id_map, cfg_source, save_dir, target_dataset_classes, conf_threshold=0.25):
    """生成带有伪标签的数据集
    """
    source_dataset_dir = os.path.dirname(cfg_source)
    cfg_source = YAML.load(cfg_source)

    splits = ['train', 'val', 'test']
    for split in splits:
        if split in cfg_source:
            source_images = os.path.join(cfg_source['path'], cfg_source[split]) if 'path' in cfg_source.keys() else \
                    os.path.join(source_dataset_dir, cfg_source[split])
            source_labels = source_images.replace('images', 'labels')

            # 1. 生成伪标注
            if split == 'train': # 只对训练集生成伪标注
                results = teacher_model.predict(source_images, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
                                                project=save_dir, name=f"pseudo_labels/{split}", verbose=False)
                for result in tqdm(results, desc=f"Generating pseudo labels", total=len(os.listdir(source_images)),
                                   position=0, leave=True, ncols=80):
                    pass # 遍历结果生成器的同时会自动保存结果文件

                # 2. 将伪标签的类别ID转换为目标数据集的类别ID
                pesudo_labels = read_labels_and_convert_class_id(os.path.join(save_dir, f"pseudo_labels/{split}/labels"), base_class_id_map)
            else:
                pesudo_labels = {} # 非训练集不需要伪标注，直接为空
            
            # 3. 读取数据集本身自带的真实标注并转换类别ID
            ground_truth_labels = read_labels_and_convert_class_id(source_labels, new_class_id_map)

            # 4. 合并真实标注和伪标注并保存
            merged_labels = merge_labels(ground_truth_labels, pesudo_labels)
            save_labels(merged_labels, os.path.join(save_dir, f"labels/{split}"))

            # 5. 复制图像文件
            images_output_dir = os.path.join(save_dir, f"images/{split}")
            shutil.copytree(source_images, images_output_dir)

            # 6. 创建数据集配置文件
            config = {
                'names': {i: cls for i, cls in enumerate(target_dataset_classes)}
            }
            for split in ['train', 'val', 'test']:
                if split in cfg_source:
                    config[split] = f"images/{split}"
            config_path = os.path.join(save_dir, f"dataconfig.yaml")
            YAML.save(data=config, file=config_path)
            
            # 删除临时生成的伪标注文件
            if os.path.exists(os.path.join(save_dir, f"pseudo_labels")):
                shutil.rmtree(os.path.join(save_dir, f"pseudo_labels"))
