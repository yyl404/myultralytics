import torch
import torch.nn as nn
import cv2
import os
from tqdm import tqdm
import psutil
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

from ultralytics import YOLO
from ultralytics.engine.projection import ConvInputPCAHooker, SubspaceProjectionLoss


class RealTimeMemoryMonitor:
    """实时内存监控器"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # 存储进度条引用
        
    def get_gpu_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        """设置进度条引用"""
        self.pbar = pbar
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # 实时更新进度条描述
            if self.pbar is not None:
                self.pbar.set_description(f"Processing images - GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        """获取当前状态"""
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


def plot_eigenvalue_distribution(lambdas, title="Eigenvalue Distribution", save_path=None):
    """
    绘制特征值分布图（保留原有功能作为对比）
    
    Args:
        lambdas: 特征值数组 [n_components]
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1：特征值条形图
    ax1.bar(range(len(lambdas)), lambdas, color='skyblue', alpha=0.7)
    ax1.set_title('Eigenvalues Bar Chart', fontweight='bold')
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：特征值对数图
    ax2.semilogy(range(len(lambdas)), lambdas, 'o-', color='red', alpha=0.7)
    ax2.set_title('Eigenvalues (Log Scale)', fontweight='bold')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Eigenvalue (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征值分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":

    test_images = "data/VOC_inc_15_5/task_1_cls_15/images/train"
    test_labels = "data/VOC_inc_15_5/task_1_cls_15/labels/train"

    model_s = YOLO("yolov8m.pt").to("cuda")
    model_t = YOLO("yolov8m.pt").to("cuda")

    layers_hooked = [i for i in range(9, len(model_t.model.model))]
    hooker = ConvInputPCAHooker(model_t.model, layers_hooked)

    image_files = os.listdir(test_images)
    random.shuffle(image_files)
    image_files = image_files[:100]
    
    # 创建实时内存监控器
    memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)
    
    # 创建进度条
    pbar = tqdm(image_files, desc="Processing images")
    
    # 将进度条传递给监控器
    memory_monitor.set_progress_bar(pbar)
    
    # 开始监控（监控器会自动更新进度条）
    memory_monitor.start_monitoring()
    
    with torch.no_grad():
        for idx, image_file in enumerate(pbar):
            image = cv2.imread(os.path.join(test_images, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).float()
            image = image.to("cuda")

            label_file = image_file.replace(".jpg", ".txt")
            bboxes = []
            with open(os.path.join(test_labels, label_file), "r") as f:
                labels = f.readlines()
                for _label in labels:
                    _label = _label.strip().split()
                    x, y, w, h = float(_label[1]), float(_label[2]), float(_label[3]), float(_label[4])
                    x_min, y_min, x_max, y_max = x - w/2, y - h/2, x + w/2, y + h/2
                    bboxes.append([x_min, y_min, x_max, y_max])

            hooker.set_bboxes([bboxes])
            hooker.register_hook()
            model_t.model.forward(image.unsqueeze(0))
            hooker.remove_handle_()

    # 停止内存监控
    memory_monitor.stop_monitoring()
    
    hooker.clear_feature_cache()
    
    # 创建输出目录
    output_dir = "runs/pca_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始分析PCA结果，共 {len(hooker)} 个模块...")
    
    subspace_basis_matrices = []
    for i in range(len(hooker)):
        components, lambdas, mean = hooker.get_pca_results(i)
        # components: [n_components, n_features]
        # lambdas: [n_components]
        # mean: [n_features]
        
        print(f"\n模块 {i}:")
        print(f"  - 特征值数量: {len(lambdas)}")
        print(f"  - 特征值范围: {lambdas.min():.2e} ~ {lambdas.max():.2e}")
        print(f"  - 前5个最大特征值: {lambdas[:5]}")
        
        # 绘制传统特征值分布图
        dist_title = f"Eigenvalue Distribution - Module {i}"
        dist_path = os.path.join(output_dir, f"eigenvalue_distribution_module_{i}.png")
        plot_eigenvalue_distribution(lambdas, title=dist_title, save_path=dist_path)

        subspace_basis_matrices.append((torch.from_numpy(components).to("cuda").T @ torch.diag(torch.sqrt(torch.from_numpy(lambdas).to("cuda")))).requires_grad_(False))
    
    print(f"\n所有可视化结果已保存到目录: {output_dir}")
    print("分析完成！")

    loss_fn = SubspaceProjectionLoss(model_s.model, model_t.model, layers_hooked, subspace_basis_matrices)

    loss_fn.register_hook()
    losses = []
    with torch.no_grad():
        for idx, image_file in enumerate(pbar):
            image = cv2.imread(os.path.join(test_images, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).float()
            image = image.to("cuda")

            model_t.model.forward(image.unsqueeze(0))
            losses.append(loss_fn.get_loss())
            loss_fn.remove_handle_()

    print("损失值：", sum(losses) / len(losses))