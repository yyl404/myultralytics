import argparse
import os
import random
import shutil
import subprocess
import types

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import YAML


def _predict_once_custom(self, x, profile=False, visualize=False, embed=None):
    """
    A customized version of _predict_once in DetectionModel.
    Change list:
    - When use embed in _predict_once, the embeddings are no longer pooled and flattened, 
      instead, the embeddings remain the feature maps generated from embedding layers.
    """
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if embed and m.i in embed:
            # embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            # if m.i == max(embed):
                # return torch.unbind(torch.cat(embeddings, 1), dim=0)
            embeddings.append(x)
            if m.i == max(embed):
                return embeddings
    return x


def generate_memory_bank(data_cfg, save_dir, model_path, k=5):
    classes = [data_cfg["names"][k] for k in sorted(data_cfg["names"].keys())]
    train_images_dir = os.path.join(os.path.dirname(data_cfg), data_cfg["train"])
    train_labels_dir = os.path.join(os.path.dirname(data_cfg), data_cfg["train"].replace("images", "labels"))
    model = YOLO(model_path)
    model.model._predict_once = types.MethodType(_predict_once_custom, model.model)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, "all_samples"))

    instance_features = {name: 0. for name in classes}
    instance_features_count = {name: 0 for name in classes}
    samples = {name: [] for name in classes}
    images = os.listdir(train_images_dir)
    labels = os.listdir(train_labels_dir)
    for i in tqdm(range(len(images)), desc="Inferring through all samples"):
        image_path = os.path.join(train_images_dir, images[i])
        image_array = cv2.imread(image_path)
        image_size = image_array.shape[:2]
        label_path = os.path.join(train_labels_dir, labels[i])
        with open(label_path, "r") as f:
            label = [line.strip() for line in f.readlines()]
        for box in label:
            cls, cx, cy, w, h = box.split(" ")
            cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)
            # crop the image
            x1, y1, x2, y2 = int((cx - w/2) * image_size[1]), int((cy - h/2) * image_size[0]),\
                int((cx + w/2) * image_size[1]), int((cy + h/2) * image_size[0])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_size[1]-1, x2), min(image_size[0]-1, y2)
            cropped_image = image_array[y1:y2+1, x1:x2+1]
            n = instance_features_count[classes[cls]]
            cv2.imwrite(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"), cropped_image)
            samples[classes[cls]].append(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"))
            embedding = model.embed(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"), verbose=False)[-1]
            feature = torch.mean(embedding, dim=(2, 3)).squeeze(0)
            instance_features[classes[cls]] = instance_features[classes[cls]] * (n/(n+1)) + feature/(n+1)
            instance_features_count[classes[cls]] += 1
    
    for name, samples_list in samples.items():
        similarity = []
        for sample in tqdm(samples_list, desc=f"Picking {name}'s best samples"):
            embedding = model.embed(sample, verbose=False)[-1]
            feature = torch.mean(embedding, dim=(2, 3)).squeeze(0)
            sim = torch.cosine_similarity(feature, instance_features[name], dim=0)
            similarity.append({"sample_path": sample, "sim": sim})
        top_k_samples = [sample["sample_path"] for sample in sorted(similarity, key=lambda x: x["sim"], reverse=True)[:args.k]]
        for k, sample_path in enumerate(top_k_samples):
            shutil.copy(sample_path, os.path.join(args.save_dir, f"{name}_best_sample_{k}.jpg"))
    shutil.rmtree(os.path.join(args.save_dir, "all_samples"))

    system_size = get_directory_size_system(args.save_dir)
    if system_size >= 0:
        print(f"\033[94mINFO:\033[0m Memory bank occupies {system_size/1024:.2f} KB")
    else:
        print(f"\033[94mINFO:\033[0m Memory bank occupies ? KB")


def get_directory_size_system(directory_path):
    """
    使用系统命令计算目录占用的磁盘大小（更高效）
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        int: 目录大小（字节），如果失败返回0
    """
    try:
        # 使用 du 命令获取目录大小（以字节为单位）
        result = subprocess.run(
            ['du', '-sb', directory_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        # du 命令输出格式: "大小\t路径"
        size_bytes = int(result.stdout.split('\t')[0])
        return size_bytes
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Warning: 无法使用系统命令获取目录大小: {e}")
        return -1


def calculate_iou(box1, box2):
    """计算两个边界框的IoU
    
    Args:
        box1: [x_center, y_center, width, height] (归一化坐标)
        box2: [x_center, y_center, width, height] (归一化坐标)
    
    Returns:
        float: IoU值
    """
    # 转换为左上角和右下角坐标
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def parse_paste_sample_filename(filename):
    """解析粘贴样本文件名，提取类别名和索引
    
    Args:
        filename: 文件名，格式为 {class_name}_best_sample_{index}.jpg
    
    Returns:
        tuple: (class_name, index)
    """
    if not filename.endswith('.jpg'):
        return None, None
    
    parts = filename.replace('.jpg', '').split('_best_sample_')
    if len(parts) != 2:
        return None, None
    
    class_name = parts[0]
    try:
        index = int(parts[1])
        return class_name, index
    except ValueError:
        return None, None


def get_paste_samples_by_class(paste_sample_dir, class_names):
    """获取按类别分组的粘贴样本
    
    Args:
        paste_sample_dir: 粘贴样本目录
        class_names: 类别名称列表
    
    Returns:
        dict: {class_name: [sample_files]}
    """
    paste_samples = {}
    for class_name in class_names:
        paste_samples[class_name] = []
    
    if not os.path.exists(paste_sample_dir):
        return paste_samples
    
    for filename in os.listdir(paste_sample_dir):
        class_name, index = parse_paste_sample_filename(filename)
        if class_name in class_names:
            paste_samples[class_name].append(filename)
    
    return paste_samples


def paste_sample_on_image(base_image, paste_image, paste_x, paste_y, paste_width, paste_height):
    """将粘贴样本粘贴到基础图像上
    
    Args:
        base_image: 基础图像 (numpy array)
        paste_image: 粘贴样本图像 (numpy array)
        paste_x, paste_y: 粘贴位置 (像素坐标)
        paste_width, paste_height: 粘贴尺寸 (像素)
    
    Returns:
        numpy array: 粘贴后的图像
    """
    # 调整粘贴样本尺寸
    paste_image_resized = cv2.resize(paste_image, (paste_width, paste_height))
    
    # 获取基础图像尺寸
    h, w = base_image.shape[:2]
    
    # 确保粘贴位置在图像范围内
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(w, paste_x + paste_width)
    y2 = min(h, paste_y + paste_height)
    
    if x2 <= x1 or y2 <= y1:
        return base_image
    
    # 计算实际粘贴区域
    paste_x_offset = max(0, -paste_x)
    paste_y_offset = max(0, -paste_y)
    paste_w_actual = x2 - x1
    paste_h_actual = y2 - y1
    
    # 粘贴样本
    base_image[y1:y2, x1:x2] = paste_image_resized[paste_y_offset:paste_y_offset+paste_h_actual, 
                                                   paste_x_offset:paste_x_offset+paste_w_actual]
    
    return base_image


def copy_paste_replay(cfg_source, memory_bank_dir, save_dir, split):
    """复制粘贴增强
    
    Args:
        cfg_source: 源数据集（yaml配置文件）
        memory_bank_dir: 记忆库样本目录
        save_dir: 保存目录
        split: 数据集分割（train/val）
    """
    # 读取源数据集配置
    dataset_dir = os.path.dirname(cfg_source)
    cfg_source = YAML.load(cfg_source)
    class_names = list(cfg_source['names'].values())
    class_name_to_id = {name: idx for idx, name in cfg_source['names'].items()}
    
    # 获取粘贴样本
    memory_samples = get_paste_samples_by_class(memory_bank_dir, class_names)
    
    # 创建保存目录
    images_save_dir = os.path.join(save_dir, 'images', split)
    labels_save_dir = os.path.join(save_dir, 'labels', split)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)
    
    # 获取源数据集图像和标签路径
    source_images_dir = os.path.join(dataset_dir, cfg_source[split])
    source_labels_dir = os.path.join(dataset_dir, 'labels', split)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(image_files)} images for {split} split...")
    
    for image_file in tqdm(image_files, desc=f"Copy-paste augmentation for {split}"):
        # 读取基础图像
        image_path = os.path.join(source_images_dir, image_file)
        base_image = cv2.imread(image_path)
        if base_image is None:
            continue
        
        h, w = base_image.shape[:2]
        
        # 读取基础标签
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(source_labels_dir, label_file)
        base_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                base_labels = [line.strip().split() for line in f.readlines()]
        
        # 随机决定粘贴样本数量 (0-3)
        num_paste_samples = random.randint(0, 3)
        
        # 复制基础图像和标签
        augmented_image = base_image.copy()
        augmented_labels = base_labels.copy()
        
        # 记录需要删除的原始标注索引
        labels_to_remove = set()
        
        # 记录已粘贴样本的边界框，用于检查粘贴样本之间的重叠
        pasted_boxes = []
        
        for _ in range(num_paste_samples):
            # 随机选择一个类别
            available_classes = [cls for cls in class_names if memory_samples[cls]]
            if not available_classes:
                break
            
            selected_class = random.choice(available_classes)
            selected_sample = random.choice(memory_samples[selected_class])
            
            # 读取粘贴样本图像
            paste_image_path = os.path.join(memory_bank_dir, selected_sample)
            paste_image = cv2.imread(paste_image_path)
            if paste_image is None:
                continue
            
            # 随机生成粘贴位置和尺寸
            paste_h, paste_w = paste_image.shape[:2]
            max_paste_w = min(w // 3, paste_w)
            max_paste_h = min(h // 3, paste_h)
            
            paste_width = random.randint(max_paste_w // 2, max_paste_w)
            paste_height = random.randint(max_paste_h // 2, max_paste_h)
            
            # 尝试找到不重叠的位置
            max_attempts = 50  # 最大尝试次数
            valid_position_found = False
            
            for attempt in range(max_attempts):
                paste_x = random.randint(0, w - paste_width)
                paste_y = random.randint(0, h - paste_height)
                
                # 计算粘贴样本的归一化边界框
                paste_center_x = (paste_x + paste_width / 2) / w
                paste_center_y = (paste_y + paste_height / 2) / h
                paste_norm_width = paste_width / w
                paste_norm_height = paste_height / h
                
                paste_box = [paste_center_x, paste_center_y, paste_norm_width, paste_norm_height]
                
                # 检查与现有标注的重叠
                overlap_with_existing = False
                for i, label in enumerate(augmented_labels):
                    if len(label) >= 5:
                        existing_box = [float(label[1]), float(label[2]), float(label[3]), float(label[4])]
                        iou = calculate_iou(paste_box, existing_box)
                        if iou > 0.5:
                            labels_to_remove.add(i)
                            overlap_with_existing = True
                
                # 检查与已粘贴样本的重叠
                overlap_with_pasted = False
                for pasted_box in pasted_boxes:
                    iou = calculate_iou(paste_box, pasted_box)
                    if iou > 1e-3:
                        overlap_with_pasted = True
                        break
                
                # 如果位置合适，跳出循环
                if not overlap_with_pasted:
                    valid_position_found = True
                    break
            
            # 如果没有找到合适的位置，跳过这个样本
            if not valid_position_found:
                continue
            
            # 粘贴样本到图像
            augmented_image = paste_sample_on_image(augmented_image, paste_image,
                                                    paste_x, paste_y, paste_width, paste_height)
            
            # 添加粘贴样本的标注
            class_id = class_name_to_id[selected_class]
            augmented_labels.append([str(class_id), str(paste_center_x), str(paste_center_y), 
                                   str(paste_norm_width), str(paste_norm_height)])
            
            # 记录已粘贴的边界框
            pasted_boxes.append(paste_box)
        
        # 删除重叠的原始标注
        final_labels = []
        for i, label in enumerate(augmented_labels):
            if i not in labels_to_remove:
                final_labels.append(label)
        
        # 保存增强后的图像
        save_image_path = os.path.join(images_save_dir, image_file)
        cv2.imwrite(save_image_path, augmented_image)
        
        # 保存增强后的标签
        save_label_path = os.path.join(labels_save_dir, label_file)
        with open(save_label_path, 'w') as f:
            for label in final_labels:
                f.write(' '.join(label) + '\n')
    
    # 复制并修改yaml配置文件
    new_yaml_path = os.path.join(save_dir, 'dataconfig.yaml')
    YAML.save(new_yaml_path, cfg_source)

    # 复制其他split的图像和标签
    for _split in ['val', "test"]:
        source_images_dir = os.path.join(dataset_dir, cfg_source[_split])
        source_labels_dir = os.path.join(dataset_dir, 'labels', _split)
        shutil.copytree(source_images_dir, os.path.join(save_dir, 'images', _split), dirs_exist_ok=True)
        shutil.copytree(source_labels_dir, os.path.join(save_dir, 'labels', _split), dirs_exist_ok=True)
    
    print(f"Copy-paste augmentation completed. Results saved to {save_dir}")
    return new_yaml_path


def crop_instance_from_image(image, bbox, class_name, index):
    """从图像中裁剪出指定边界框的实例
    
    Args:
        image: 输入图像
        bbox: 边界框 [x_center, y_center, width, height] (归一化坐标)
        class_name: 类别名称
        index: 索引
    
    Returns:
        tuple: (cropped_image, filename)
    """
    h, w = image.shape[:2]
    
    # 转换为像素坐标
    x_center = int(bbox[0] * w)
    y_center = int(bbox[1] * h)
    bbox_width = int(bbox[2] * w)
    bbox_height = int(bbox[3] * h)
    
    # 计算边界框的左上角坐标
    x1 = max(0, x_center - bbox_width // 2)
    y1 = max(0, y_center - bbox_height // 2)
    x2 = min(w, x_center + bbox_width // 2)
    y2 = min(h, y_center + bbox_height // 2)
    
    # 裁剪图像
    cropped = image[y1:y2, x1:x2]
    
    # 生成文件名
    filename = f"{class_name}_sample_{index}.jpg"
    
    return cropped, filename


def mix_up_augmentation(cfg_source, memory_bank_dir, save_dir, split, num_generations):
    """mix-up增强
    
    Args:
        cfg_source: 源数据集（yaml配置文件）
        memory_bank_dir: 记忆库样本目录
        save_dir: 保存目录
        split: 数据集分割（train/val）
        num_generations: 生成的新样本数量
    """
    # 读取源数据集配置
    yaml_data = YAML.load(cfg_source)
    class_names = list(yaml_data['names'].values())
    
    # 创建保存目录
    cropped_source_dir = os.path.join(save_dir, 'cropped_samples_source')
    mixed_up_dir = os.path.join(save_dir, 'mixed_up_samples')
    
    os.makedirs(cropped_source_dir, exist_ok=True)
    os.makedirs(mixed_up_dir, exist_ok=True)
    
    # 获取源数据集图像和标签路径
    source_images_dir = os.path.join(os.path.dirname(cfg_source), yaml_data[split])
    source_labels_dir = os.path.join(os.path.dirname(cfg_source), 'labels', split)
    
    # 步骤1: 裁剪源数据集中的实例
    cropped_count = 0
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in tqdm(image_files, desc="Cropping instances from source dataset"):
        # 读取图像
        image_path = os.path.join(source_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # 读取标签
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(source_labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
            
            # 裁剪每个实例
            for i, label in enumerate(labels):
                if len(label) >= 5:
                    class_id = int(label[0])
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        bbox = [float(label[1]), float(label[2]), float(label[3]), float(label[4])]
                        
                        cropped_image, filename = crop_instance_from_image(image, bbox, class_name, cropped_count)
                        
                        if cropped_image.size > 0:  # 确保裁剪的图像不为空
                            save_path = os.path.join(cropped_source_dir, filename)
                            cv2.imwrite(save_path, cropped_image)
                            cropped_count += 1
    print(f"Cropped {cropped_count} instances from source dataset")
    
    # 步骤2: 创建mix-up样本
    mixed_up_count = 0
    cropped_source_files = [f for f in os.listdir(cropped_source_dir) if f.endswith('.jpg')]
    
    # 获取记忆库样本
    memory_samples = {}
    if os.path.exists(memory_bank_dir):
        for filename in os.listdir(memory_bank_dir):
            if filename.endswith('.jpg'):
                class_name, index = parse_paste_sample_filename(filename)
                if class_name not in memory_samples:
                    memory_samples[class_name] = []
                memory_samples[class_name].append(filename)
    
    for source_file in tqdm(cropped_source_files, desc="Creating mixed-up samples"):
        # 解析源文件名
        parts = source_file.replace('.jpg', '').split('_sample_')
        if len(parts) != 2:
            continue
        
        source_class = parts[0]
        
        # 随机选择记忆库中的样本
        available_classes = list(memory_samples.keys())
        if not available_classes:
            continue
        
        memory_class = random.choice(available_classes)
        memory_file = random.choice(memory_samples[memory_class])
        
        # 读取源样本
        source_path = os.path.join(cropped_source_dir, source_file)
        source_image = cv2.imread(source_path)
        
        # 读取记忆库样本
        memory_path = os.path.join(memory_bank_dir, memory_file)
        memory_image = cv2.imread(memory_path)
        
        if source_image is None or memory_image is None:
            continue
        
        # 调整到相同尺寸 (使用较小的尺寸)
        target_size = (min(source_image.shape[1], memory_image.shape[1]), 
                      min(source_image.shape[0], memory_image.shape[0]))
        
        source_resized = cv2.resize(source_image, target_size)
        memory_resized = cv2.resize(memory_image, target_size)
        
        # 生成lambda值 (beta分布，参数为(1,1))
        lambda_val = np.random.beta(1, 1)
        
        # 进行mix-up
        mixed_image = cv2.addWeighted(source_resized, lambda_val, memory_resized, 1 - lambda_val, 0)
        
        # 保存mix-up样本
        mixed_filename = f"{source_class}_{memory_class}_mixed_up_sample_{mixed_up_count}.jpg"
        mixed_path = os.path.join(mixed_up_dir, mixed_filename)
        cv2.imwrite(mixed_path, mixed_image)
        mixed_up_count += 1
    
    print(f"Created {mixed_up_count} mixed-up samples")
    
    # 步骤3: 生成最终样本
    mixed_up_files = [f for f in os.listdir(mixed_up_dir) if f.endswith('.jpg')]
    generation_log = {}
    
    for i in tqdm(range(num_generations), desc="Generating final samples"):
        # 创建噪声基础图像 (640x640)
        base_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 随机选择k个mix-up样本 (1-4个)
        k = random.randint(1, 4)
        selected_samples = random.sample(mixed_up_files, min(k, len(mixed_up_files)))
        
        # 记录使用的样本
        generation_log[f"sample_{i}.jpg"] = selected_samples
        
        # 粘贴选中的样本到基础图像上
        pasted_boxes = []
        
        for sample_file in selected_samples:
            # 读取样本图像
            sample_path = os.path.join(mixed_up_dir, sample_file)
            sample_image = cv2.imread(sample_path)
            
            if sample_image is None:
                continue
            
            # 随机生成粘贴位置和尺寸
            sample_h, sample_w = sample_image.shape[:2]
            
            paste_width = random.randint(int(sample_w * 0.75), sample_w)
            paste_height = random.randint(int(sample_h * 0.75), sample_h)
            
            # 尝试找到不重叠的位置
            max_attempts = 50
            valid_position_found = False
            
            for attempt in range(max_attempts):
                paste_x = random.randint(0, 640 - paste_width)
                paste_y = random.randint(0, 640 - paste_height)
                
                # 计算粘贴样本的归一化边界框
                paste_center_x = (paste_x + paste_width / 2) / 640
                paste_center_y = (paste_y + paste_height / 2) / 640
                paste_norm_width = paste_width / 640
                paste_norm_height = paste_height / 640
                
                paste_box = [paste_center_x, paste_center_y, paste_norm_width, paste_norm_height]
                
                # 检查与已粘贴样本的重叠
                overlap = False
                for pasted_box in pasted_boxes:
                    iou = calculate_iou(paste_box, pasted_box)
                    if iou > 1e-3:
                        overlap = True
                        break
                
                if not overlap:
                    valid_position_found = True
                    break
            
            if valid_position_found:
                # 粘贴样本到图像
                sample_resized = cv2.resize(sample_image, (paste_width, paste_height))
                base_image[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = sample_resized
                pasted_boxes.append(paste_box)
        
        # 保存最终样本
        cv2.imwrite(os.path.join(save_dir, f"sample_{i}.jpg"), base_image)
    
    # 步骤4: 删除中间文件
    shutil.rmtree(cropped_source_dir)
    shutil.rmtree(mixed_up_dir)

    system_size = get_directory_size_system(save_dir)
    if system_size >= 0:
        print(f"\033[94mINFO:\033[0m Mixed-up samples occupies {system_size/1024:.2f} KB")
    else:
        print(f"\033[94mINFO:\033[0m Mixed-up samples occupies ? KB")
    
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dataset_cfg", type=str, required=True, help="Base dataset configuration file")
    parser.add_argument("--new_dataset_cfg", type=str, required=True, help="New dataset configuration file")
    parser.add_argument("--memory_bank_dir", type=str, required=True, help="Memory bank directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--split", type=str, default="train", help="Split")
    parser.add_argument("--num_generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--k", type=int, default=5, help="Number of samples to select for mix-up")

    parser.add_argument("--generate_memory_bank", action="store_true", help="Generate memory bank")
    parser.add_argument("--copy_paste_replay", action="store_true", help="Copy paste replay")
    parser.add_argument("--mix_up_augmentation", action="store_true", help="Mix up augmentation")
    parser.add_argument("--all", action="store_true", help="Process all steps(generate_memory_bank->copy_paste_replay->mix_up_augmentation)")
    args = parser.parse_args()
    
    base_dataset_cfg = args.base_dataset_cfg
    new_dataset_cfg = args.new_dataset_cfg
    memory_bank_dir = args.memory_bank_dir
    save_dir = args.save_dir
    model_path = args.model_path
    if args.generate_memory_bank:
        generate_memory_bank(base_dataset_cfg, memory_bank_dir, model_path, args.k)
    if args.copy_paste_replay:
        copy_paste_replay(new_dataset_cfg, memory_bank_dir, save_dir, args.split)
    if args.mix_up_augmentation:
        mix_up_augmentation(new_dataset_cfg, memory_bank_dir, save_dir, args.split, args.num_generations)
    if args.all:
        generate_memory_bank(base_dataset_cfg, memory_bank_dir, model_path, args.k)
        copy_paste_replay(new_dataset_cfg, memory_bank_dir, save_dir, args.split)
        mix_up_augmentation(new_dataset_cfg, memory_bank_dir, save_dir, args.split, args.num_generations)