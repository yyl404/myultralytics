import os
from os import path as OSP
import shutil
from tqdm import tqdm
import argparse

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional
from torch.nn import Sequential
from torch.nn import functional as F

from ultralytics import YOLO
from ultralytics.utils import YAML, LOGGER
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
            else:
                LOGGER.warning(f"Class ID {old_cat_id} not found in class_id_map")
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
    if OSP.exists(labels_dir):
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = OSP.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                labels[label_file] = convert_class_id(lines, class_id_map)
    return labels


def save_labels(labels, output_dir):
    """保存标注文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for label_file, lines in labels.items():
        output_path = OSP.join(output_dir, label_file)
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
    source_dataset_dir = OSP.dirname(cfg_source)
    cfg_source = YAML.load(cfg_source)
    
    # 创建目标数据集目录
    target_dataset_dir = OSP.abspath(OSP.join(save_dir, target_dataset_name))
    if OSP.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.makedirs(target_dataset_dir, exist_ok=True)
    
    # 复制图像文件并转换标签
    for split in ['train', 'val', 'test']:
        if split in cfg_source:
            # 原始数据集图像目录
            source_images = OSP.join(cfg_source['path'], cfg_source[split]) if 'path' in cfg_source.keys() else \
                OSP.join(source_dataset_dir, cfg_source[split])
            
            # 目标图像目录
            target_images = OSP.join(target_dataset_dir, f"images/{split}")
            os.makedirs(target_images, exist_ok=True)
            shutil.copytree(source_images, target_images, dirs_exist_ok=True)
            
            # 原始数据集标签目录
            source_labels = OSP.join(cfg_source['path'], cfg_source[split].replace('images', 'labels')) if 'path' in cfg_source.keys() else \
                OSP.join(source_dataset_dir, cfg_source[split].replace('images', 'labels'))
            
            # 目标标签目录
            target_labels = OSP.join(target_dataset_dir, f"labels/{split}")
            os.makedirs(target_labels, exist_ok=True)
            
            # 转换标签的类别ID并保存
            if OSP.exists(source_labels):
                converted_labels = read_labels_and_convert_class_id(source_labels, class_id_map)
                save_labels(converted_labels, target_labels)
    
    # 创建目标数据集的配置文件
    config = {
        'names': {i: cls for i, cls in enumerate(target_dataset_classes)}
    }
    for split in ['train', 'val', 'test']:
        if split in cfg_source:
            config[split] = f"images/{split}"
    config_path = OSP.join(target_dataset_dir, 'dataconfig.yaml')
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
    YAML.save(data=model_cfg, file=OSP.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    new_model = YOLO(OSP.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    new_weight = new_model.model.state_dict()

    # 权重迁移：分类层按映射迁移，其他层直接复制
    for key in new_weight.keys():
        if key in weight:
            layer_id = int(key.split('.')[1])
            # 处理cv3中最后的分类层权重（Conv2d层，即.2.weight）
            if layer_id == len(model.model.model) - 1 and 'cv3' in key and key.endswith('.2.weight'):
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
    new_model.save(OSP.join(save_dir, output_name))


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
    source_dataset_dir = OSP.dirname(cfg_source)
    cfg_source = YAML.load(cfg_source)

    splits = ['train', 'val', 'test']
    for split in splits:
        if split in cfg_source:
            source_images = OSP.join(cfg_source['path'], cfg_source[split]) if 'path' in cfg_source.keys() else \
                    OSP.join(source_dataset_dir, cfg_source[split])
            source_labels = source_images.replace('images', 'labels')

            # 1. 生成伪标注
            if split == 'train': # 只对训练集生成伪标注
                results = teacher_model.predict(source_images, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
                                                project=save_dir, name=f"pseudo_labels/{split}", verbose=False)
                for result in tqdm(results, desc=f"Generating pseudo labels", total=len(os.listdir(source_images)),
                                   position=0, leave=True, ncols=80):
                    pass # 遍历结果生成器的同时会自动保存结果文件

                # 2. 将伪标签的类别ID转换为目标数据集的类别ID
                pesudo_labels = read_labels_and_convert_class_id(OSP.join(save_dir, f"pseudo_labels/{split}/labels"), base_class_id_map)
            else:
                pesudo_labels = {} # 非训练集不需要伪标注，直接为空
            
            # 3. 读取数据集本身自带的真实标注并转换类别ID
            ground_truth_labels = read_labels_and_convert_class_id(source_labels, new_class_id_map)

            # 4. 合并真实标注和伪标注并保存
            merged_labels = merge_labels(ground_truth_labels, pesudo_labels)
            save_labels(merged_labels, OSP.join(save_dir, f"labels/{split}"))

            # 5. 复制图像文件
            images_output_dir = OSP.join(save_dir, f"images/{split}")
            shutil.copytree(source_images, images_output_dir)

            # 6. 创建数据集配置文件
            config = {
                'names': {i: cls for i, cls in enumerate(target_dataset_classes)}
            }
            for split in ['train', 'val', 'test']:
                if split in cfg_source:
                    config[split] = f"images/{split}"
            config_path = OSP.join(save_dir, f"dataconfig.yaml")
            YAML.save(data=config, file=config_path)
            
            # 删除临时生成的伪标注文件
            if OSP.exists(OSP.join(save_dir, f"pseudo_labels")):
                shutil.rmtree(OSP.join(save_dir, f"pseudo_labels"))


# ============================ VAE 回放相关 ============================
def _vae_patch_tensor(images: torch.Tensor, patch_size: int = 64, patch_stride: int = 64, patch_padding: int = 0) -> torch.Tensor:
    """将 [N,C,H,W] 图像分块为 [N*L,C,K,K]，与 VAE 训练/推理保持一致。"""
    unfolded = F.unfold(images, kernel_size=patch_size, stride=patch_stride, padding=patch_padding)
    n, ckk, l = unfolded.shape
    c = images.shape[1]
    k = patch_size
    patches = unfolded.permute(0, 2, 1).contiguous().view(n * l, c, k, k)
    return patches


def _vae_concat_patches(patches: torch.Tensor, image_hw=(640, 640), patch_size: int = 64, patch_stride: int = 64, patch_padding: int = 0) -> torch.Tensor:
    """将 [L,C,K,K] 或 [N*L,C,K,K] 拼回 [N,C,H,W]；此处仅用于 N=1 情况。"""
    h, w = image_hw
    c = patches.shape[1]
    k = patch_size
    s = patch_stride
    p = patch_padding
    num_h = (h - k + 2 * p) // s + 1
    num_w = (w - k + 2 * p) // s + 1
    recon = torch.zeros(1, c, h, w, device=patches.device, dtype=patches.dtype)
    count = torch.zeros(h, w, device=patches.device, dtype=patches.dtype)
    idx = 0
    for i in range(num_h):
        for j in range(num_w):
            sh, eh = i * s, i * s + k
            sw, ew = j * s, j * s + k
            recon[0, :, sh:eh, sw:ew] += patches[idx]
            count[sh:eh, sw:ew] += 1
            idx += 1
    count = count.clamp(min=1)
    recon = recon / count.unsqueeze(0).unsqueeze(0)
    return recon


def _load_vae(model_ckpt: str, device: str = "cuda", arch: str = "vq", latent_dim: int = 512):
    """从 /root/vae-search/vae.py 加载 VAE 模型并恢复权重。"""
    import sys
    vae_repo = "/root/vae-search"
    if vae_repo not in sys.path:
        sys.path.append(vae_repo)
    from vae import VQVAE, VanillaVAE  # type: ignore

    if arch == "vq":
        model = VQVAE(3, latent_dim, num_embeddings=512)
    else:
        model = VanillaVAE(3, latent_dim)
    state = torch.load(model_ckpt, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _prepare_transform(image_size=(640, 640)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_vae_replay_dataset(
    base_model: YOLO,
    base_class_id_map: dict,
    new_class_id_map: dict,
    cfg_source: str,
    save_dir: str,
    target_dataset_classes: list,
    vae_ckpt: str,
    replay_source_images: str,
    arch: str = "vq",
    image_size=(640, 640),
    patch_size: int = 64,
    patch_stride: int = 64,
    patch_padding: int = 0,
    conf_threshold: float = 0.25,
    device: str = "cuda",
) -> str:
    """
    使用预训练 VAE 对给定目录的图像进行重构→用旧模型打伪标签→与原数据集合并，输出新的 dataconfig.yaml。

    Args:
        base_model: 旧任务教师模型（用于伪标签）
        base_class_id_map/new_class_id_map: 类别映射
        cfg_source: 原数据集 yaml
        save_dir: 输出根目录（将在其下创建 training_dataset_with_pseudo_labels_replay）
        target_dataset_classes: 完整类别列表
        vae_ckpt: VAE 权重路径（如 /root/vae-search/logs/best.pt）
        replay_source_images: 用于回放生成的源图像目录（建议用历史任务样本目录或 memory bank 图像目录）
        arch: "vq" 或 "vanilla"
    Returns:
        新 dataconfig.yaml 路径
    """
    os.makedirs(save_dir, exist_ok=True)
    out_root = OSP.join(save_dir, "training_dataset_with_pseudo_labels_replay")
    if OSP.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    # 1) 准备 VAE 与图像增广
    vae = _load_vae(vae_ckpt, device=device, arch=arch)
    tfm = _prepare_transform(image_size)

    # 2) 遍历源目录，重构图像并保存到 out_root/images/train
    images_out = OSP.join(out_root, "images", "train")
    labels_out = OSP.join(out_root, "labels", "train")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    def _iter_images(root):
        for name in os.listdir(root):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                yield OSP.join(root, name)

    for img_path in tqdm(list(_iter_images(replay_source_images)), desc="VAE replay reconstruct"):
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_t = tfm(img).unsqueeze(0).to(device)
            patches = _vae_patch_tensor(img_t, patch_size, patch_stride, patch_padding)
            with torch.no_grad():
                recon, *_ = vae(patches)
            recon_full = _vae_concat_patches(recon, image_hw=image_size, patch_size=patch_size, patch_stride=patch_stride, patch_padding=patch_padding)
            # 反标准化保存
            mean = torch.tensor([0.485, 0.456, 0.406], device=recon_full.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=recon_full.device).view(1, 3, 1, 1)
            recon_denorm = torch.clamp(recon_full * std + mean, 0, 1)
            recon_np = (recon_denorm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)
            out_name = OSP.basename(img_path)
            out_img_path = OSP.join(images_out, out_name)
            cv2.imwrite(out_img_path, recon_bgr)
        except Exception:
            continue

    # 3) 复制原数据集其余 split，建立基础 yaml（沿用 create_pseudo_labels_dataset 的风格）
    source_dataset_dir = OSP.dirname(cfg_source)
    cfg = YAML.load(cfg_source)
    for split in ["val", "test"]:
        if split in cfg:
            src_images = OSP.join(cfg.get("path", source_dataset_dir), cfg[split]) if "path" in cfg.keys() else OSP.join(source_dataset_dir, cfg[split])
            src_labels = src_images.replace("images", "labels")
            os.makedirs(OSP.join(out_root, "images", split), exist_ok=True)
            os.makedirs(OSP.join(out_root, "labels", split), exist_ok=True)
            if OSP.exists(src_images):
                shutil.copytree(src_images, OSP.join(out_root, "images", split), dirs_exist_ok=True)
            if OSP.exists(src_labels):
                shutil.copytree(src_labels, OSP.join(out_root, "labels", split), dirs_exist_ok=True)

    # 4) 用教师模型对重构的 train 图像打伪标签（保存到 labels/train）并做类别映射
    results = base_model.predict(images_out, conf=conf_threshold, save_txt=True, save_conf=False, stream=True,
                                 project=out_root, name=f"pseudo_labels/train", verbose=False)
    for _ in tqdm(results, desc="Generate pseudo labels for replay", ncols=80):
        pass
    pseudo_labels_src = OSP.join(out_root, "pseudo_labels", "train", "labels")
    if OSP.exists(pseudo_labels_src):
        converted = read_labels_and_convert_class_id(pseudo_labels_src, base_class_id_map)
        save_labels(converted, labels_out)
        shutil.rmtree(OSP.join(out_root, "pseudo_labels"))

    # 5) 写 dataconfig.yaml
    config = {
        "names": {i: cls for i, cls in enumerate(target_dataset_classes)}
    }
    for split in ["train", "val", "test"]:
        if split in cfg:
            config[split] = f"images/{split}"
    data_yaml = OSP.join(out_root, "dataconfig.yaml")
    YAML.save(data=config, file=data_yaml)
    return data_yaml


def main(args):
    if args.convert_dataset_class_id:
        model = YOLO(args.model_path)
        model_classes = [model.names[i] for i in sorted(model.names.keys())]
        
        data_cfg = YAML.load(args.data_cfg)
        source_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]
        
        class_id_map = {}
        for i, cls in enumerate(source_classes):
            class_id_map[i] = model_classes.index(cls)

        root_dir, dataset_name = OSP.split(args.save_dir)
        create_classes_expanded_dataset(args.data_cfg, class_id_map, root_dir, dataset_name, model_classes)
        return 0

    if args.expand_detection_head:
        base_model = YOLO(args.model_path)
        base_classes = [base_model.names[i] for i in sorted(base_model.names.keys())]
        
        data_cfg = YAML.load(args.data_cfg)
        new_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]

        all_classes = list(set(base_classes).union(new_classes))
        
        base_class_id_map = {}
        for i, cls in enumerate(base_classes):
            base_class_id_map[i] = all_classes.index(cls)

        new_class_id_map = {}
        for i, cls in enumerate(new_classes):
            new_class_id_map[i] = all_classes.index(cls)
        
        root_dir, model_name = OSP.split(args.save_path)
        expand_detection_head(args.model_path, args.model_cfg, base_class_id_map, all_classes,
                              root_dir, model_name)
        return 0
    
    if args.create_pseudo_labels_dataset:
        teacher_model = YOLO(args.model_path)
        base_classes = [teacher_model.names[i] for i in sorted(teacher_model.names.keys())]

        data_cfg = YAML.load(args.data_cfg)
        new_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]

        all_classes = list(set(base_classes).union(new_classes))

        base_class_id_map = {}
        for i, cls in enumerate(base_classes):
            base_class_id_map[i] = all_classes.index(cls)

        new_class_id_map = {}
        for i, cls in enumerate(new_classes):
            new_class_id_map[i] = all_classes.index(cls)
        
        if OSP.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        
        root_dir, model_name = OSP.split(args.save_dir)
        create_pseudo_labels_dataset(teacher_model, base_class_id_map, new_class_id_map, 
                                     args.data_cfg,
                                     args.save_dir, all_classes, args.conf_threshold)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, help="Data config path")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    # 转换数据集类别ID
    parser.add_argument("--convert_dataset_class_id", action="store_true", help="Convert dataset class id")
    # 合并增量类别并扩展检测头
    parser.add_argument("--expand_detection_head", action="store_true", help="Expand detection head")
    parser.add_argument("--model_cfg", type=str, help="Model config path")
    parser.add_argument("--save_path", type=str, help="Save path")
    # 创建伪标签数据集
    parser.add_argument("--create_pseudo_labels_dataset", action="store_true", help="Create pseudo labels dataset")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    main(args)