"""
对yolo模型的中间层输入特征激活值分布进行PCA分析

用法:
    python pca.py --model <model_path> --sample_dir <sample_dir> --save_path <save_path> \
    [--sample_num <sample_num> --label_dir <label_dir> --load_path <load_path> --layers "<layer1,layer2,...>" --check]

参数:
    --model: 模型权重路径（格式为YOLO模型）
    --sample_dir: 用于计算PCA的样本路径
    --sample_num: 采样的最大数量
    --save_path: 保存PCA结果的文件路径
    --label_dir: （可选）标签路径
    --layers: （可选）要计算PCA的层，用逗号分隔，如果不指定则默认计算所有中间层
    --check: （可选）是否检查卷积核展开操作正确性
"""

import threading
import psutil
import time
import joblib
import os
import random
import cv2
from tqdm import tqdm
import glob
import argparse
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils import (
    LOGGER
)

from pca_on_gpu import IncrementalPCAonGPU as IncrementalPCA


class RealTimeMemoryMonitor:
    """Real-time memory monitor"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # store progress bar reference
        
    def get_gpu_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        self.pbar = pbar
    
    def _monitor_loop(self):
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # Real-time update progress bar description
            if self.pbar is not None:
                self.pbar.set_description(f"PCA computing - GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


class PCAHooker:
    def __init__(self, model, layers, modules=None, device="cuda", check=False, unfold=False):
        self.model = model
        self.modules = {}
        self.pca_operators = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        self.check = check
        self.unfold = unfold
        
        def _match(n, m, lid):
            "dfl layer is always frozen, so we don't need to calculate PCA for it"
            return f"model.{lid}." in n and isinstance(m, nn.Conv2d) and "dfl" not in n

        self.feature_caches, self._handles = {}, []

        if modules is not None:
            # If modules are provided, only calculate PCA for the specified modules
            for n, m in model.named_modules():
                if n in modules:
                    k, c_in, g = m.kernel_size, m.in_channels, m.groups
                    self.modules[n] = m
                    self.pca_operators[n] = []
                    for i in range(g):
                        if self.unfold:
                            n_components = c_in//g*k[0]*k[1]
                        else:
                            n_components = c_in//g
                        self.pca_operators[n].append(IncrementalPCA(n_components=n_components))
                    self.feature_caches[n] = []
        elif layers is not None:
            for lid in layers:
                # If layers are provided, calculate PCA for all conv modules within layers
                for n, m in model.named_modules():
                    if _match(n, m, lid):
                        k, c_in, g = m.kernel_size, m.in_channels, m.groups
                        self.modules[n] = m
                        self.pca_operators[n] = []
                        for i in range(g):
                            if self.unfold:
                                n_components = c_in//g*k[0]*k[1]
                            else:
                                n_components = c_in//g
                            self.pca_operators[n].append(IncrementalPCA(n_components=n_components))
                        self.feature_caches[n] = []
        else:
            raise ValueError("Either modules or layers must be provided")

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        # randomly sample feature indices
        sample_feature_indices = torch.randperm(bs*h_out*w_out, device=self.device)[:100]
        return sample_feature_indices

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(n, self.unfold)))

    def _hook(self, module_name, unfold=False):
        """
        If unfold is True, the input feature will be unfolded into a sequence of vectors with shape [c_in*k[0]*k[1], bs*h_out*w_out],
        otherwise, the input feature will be reshaped into a matrix with shape [c_in, bs*h_out*w_out].
        """
        def fn(module, feat_in, feat_out):
            if isinstance(module, nn.Conv2d):
                k, s, p, g, d, c_in, c_out = module.kernel_size, module.stride, module.padding, \
                    module.groups, module.dilation, module.in_channels, module.out_channels

                feat_in = feat_in[0] # module may accept multiple input feats, and we only extract the first
                if p[0] > 0 or p[1] > 0:
                    feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
                else:
                    feat_in_padded = feat_in
                bs, _, h_in, w_in = feat_in_padded.shape
                h_out, w_out = feat_out.shape[2], feat_out.shape[3]

                # group the input features
                feat_in_padded_grouped = feat_in_padded.reshape(bs, g, c_in//g, h_in, w_in)
                c_in_grouped = c_in//g

                if unfold:
                    # Use the sliding window with the same settings as convolution kernels
                    # to unfold input features into a sequence of vectors, considering dilation
                    # For dilated convolution, we need to adjust the unfold parameters
                    if d[0] > 1 or d[1] > 1:
                        # For dilation > 1, we need to unfold with larger window size
                        # Effective kernel size becomes: k[0] + (k[0]-1)*(d[0]-1), k[1] + (k[1]-1)*(d[1]-1)
                        effective_k_h = k[0] + (k[0] - 1) * (d[0] - 1)
                        effective_k_w = k[1] + (k[1] - 1) * (d[1] - 1)
                        
                        # [bs, g, c_in//g, h, w] --> [bs, g, c_in//g, h_out, w, effective_k_h]
                        feat_unfold_h = feat_in_padded_grouped.unfold(3, effective_k_h, s[0])
                        # [bs, g, c_in//g, h_out, w, effective_k_h] --> [bs, g, c_in//g, h_out, w_out, effective_k_h, effective_k_w]
                        feat_unfold = feat_unfold_h.unfold(4, effective_k_w, s[1])
                        
                        # Now subsample to get the actual dilated kernel positions
                        # feat_unfold shape: [bs, g, c_in//g, h_out, w_out, effective_k_h, effective_k_w]
                        # We keep only every d[0]-th and d[1]-th element in kernel dimensions
                        feat_unfold = feat_unfold[:, :, :, :, :, ::d[0], ::d[1]]
                    else:
                        # Standard convolution (dilation = 1)
                        # [bs, g, c_in//g, h, w] --> [bs, g, c_in//g, h_out, w, k[0]]
                        feat_unfold_h = feat_in_padded_grouped.unfold(3, k[0], s[0])
                        # [bs, g, c_in//g, h_out, w, k[0]] --> [bs, g, c_in//g, h_out, w_out, k[0], k[1]]
                        feat_unfold = feat_unfold_h.unfold(4, k[1], s[1])
                    # Get actual kernel dimensions after dilation processing
                    actual_k_h, actual_k_w = feat_unfold.shape[5], feat_unfold.shape[6]
                    
                    # Permute the dims: [bs, g, c_in//g, h_out, w_out, actual_k_h, actual_k_w] -> [g, c_in//g, actual_k_h, actual_k_w, bs, h_out, w_out]
                    feat_unfold = feat_unfold.permute(1, 2, 5, 6, 0, 3, 4).contiguous()
                    # Squeeze: [g, c_in//g, actual_k_h, actual_k_w, bs, h_out, w_out] --> [g, c_in//g*actual_k_h*actual_k_w, bs*h_out*w_out]
                    feat_reshaped = feat_unfold.view(g, c_in_grouped*actual_k_h*actual_k_w, bs*h_out*w_out)

                    # The following code is used to check whether the unfolding representation of convolution operation
                    # is equivalent with the original convolution operation
                    if self.check:
                        weight = module.weight.data.reshape(g, c_out//g, -1) # [g, c_out//g, c_in//g*actual_k_h*actual_k_w]
                        feat_out_reshaped = weight @ feat_reshaped # [g, c_out//g, c_in//g*actual_k_h*actual_k_w] @ [c_in//g*actual_k_h*actual_k_w, bs*h_out*w_out] --> [g, c_out//g, bs*h_out*w_out]
                        feat_out_reshaped = feat_out_reshaped.reshape(c_out, -1) # [c_out, bs*h_out*w_out]

                        # [c_out, bs*h_out*w_out] --> [c_out, bs, h_out, w_out] --> [bs, c_out, h_out, w_out]
                        feat_out_reshaped_reversed = feat_out_reshaped.view(-1, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()
                        if module.bias is not None:
                            feat_out_reshaped_reversed = feat_out_reshaped_reversed + module.bias.data.reshape(1, c_out, 1, 1)
                        
                        LOGGER.info(f"Module {module_name}'s unfolding error: {F.mse_loss(feat_out, feat_out_reshaped_reversed)}")
                else:
                    feat_reshaped = feat_in_padded_grouped.permute(1, 2, 0, 3, 4).view(g, c_in_grouped, -1) # [bs, g, c_in//g, h_out, w_out] --> [g, c_in//g, bs*h_in*w_in]

                    if self.check:
                        weight = module.weight.data.reshape(g, c_out//g, c_in//g, k[0], k[1]) # [g, c_out//g, c_in//g, k[0], k[1]]
                        feat_out_reshaped = []
                        for i in range(k[0]):
                            for j in range(k[1]):
                                # Group-wise convolution: weight[kernel_i,kernel_j] @ input_features[kernel_i,kernel_j] 
                                # Shape: [g, c_out//g, c_in//g] @ [bs, h_out, w_out, g, c_in//g, 1] --> [bs, h_out, w_out, g, c_out//g, 1]
                                # Permute the dims: [bs, h_out, w_out, g, c_out//g, 1] -> [bs, g, c_out//g, h_out, w_out, 1]
                                # Flatten the dims: [bs, g, c_out//g, h_out, w_out, 1] -> [bs, c_out, h_out, w_out, 1]
                                # Squeeze the last dim: [bs, c_out, h_out, w_out, 1] -> [bs, c_out, h_out, w_out]
                                feat_out_reshaped_tmp = feat_in_padded_grouped[:, :, :, i*d[0]::s[0], j*d[1]::s[1], None].permute(0, 3, 4, 1, 2, 5)
                                feat_out_reshaped_tmp = weight[:, :, :, i, j] @ feat_out_reshaped_tmp
                                feat_out_reshaped_tmp = feat_out_reshaped_tmp.permute(0, 3, 4, 1, 2, 5).flatten(1, 2).squeeze(-1)
                                feat_out_reshaped.append(feat_out_reshaped_tmp[:, :, :h_out, :w_out])
                        feat_out_reshaped = sum(feat_out_reshaped) # [bs, c_out, h_out, w_out]

                        feat_out_reshaped_reversed = feat_out_reshaped
                        if module.bias is not None:
                            feat_out_reshaped_reversed = feat_out_reshaped_reversed + module.bias.data.reshape(1, c_out, 1, 1)
                        
                        LOGGER.info(f"Module {module_name}'s unfolding error: {F.mse_loss(feat_out, feat_out_reshaped_reversed)}")

                if self.unfold:
                    sample_feature_indices = self._get_sample_feature_indices(bs, h_out, w_out)
                else:
                    sample_feature_indices = self._get_sample_feature_indices(bs, h_in, w_in)
                if sample_feature_indices.max() >= feat_reshaped.shape[2]:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.max()} >= {feat_reshaped.shape[2]}")
                if sample_feature_indices.min() < 0:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.min()} < 0")
                if sample_feature_indices.shape[0] == 0:
                    # Some batches may have no bounding boxes, so we need to return here
                    return
                feat_sampled = feat_reshaped[:, :, sample_feature_indices]
                # unfold true: [g, c_in//g*k*k, len(sample_feature_indices)] | unfold false: [g, c_in//g, len(sample_feature_indices)]
                
                feature_cache = self.feature_caches[module_name]
                feature_cache.append(feat_sampled)

                pca_operators = self.pca_operators[module_name]
                if sum([x.shape[2] for x in feature_cache]) >= pca_operators[0].n_components: # Incremental PCA requires the first batch's size is larger than n_components
                    feat_sampled = torch.cat(feature_cache, dim=2)
                    feature_cache.clear()
                    for ig in range(g):
                        pca_operators[ig].partial_fit(feat_sampled[ig].T)
            else:
                raise RuntimeError(f"Module type {type(module)} is not supported")

        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_feature_cache(self):
        for n, cache in self.feature_caches.items():
            if len(cache) > 0:
                for ig in range(len(self.pca_operators[n])):
                    # before clearing the cache, we need to fit the PCA operator with the final batch of features
                    if not hasattr(self.pca_operators[n][ig], 'components_'):
                        # If components_ is not calculated, it means the PCA operator has not been called yet, and that
                        # means the number of features in this cache has never reached n_components, so we need to
                        # reduce n_components to the number of features in this cache.
                        LOGGER.warning(f"Too few samples to fit PCA in module {n}. Could result in instability.")
                        self.pca_operators[n][ig].n_components = torch.cat(cache, dim=2)[ig].shape[1]   
                    # self.pca_operators[n].partial_fit(torch.cat(cache, dim=1).cpu().T.numpy())
                    self.pca_operators[n][ig].partial_fit(torch.cat(cache, dim=2)[ig].T)
            cache.clear()
    
    def get_pca_results(self, name, ig=None):
        if ig is not None:
            return (self.pca_operators[name][ig].components_, 
                    self.pca_operators[name][ig].explained_variance_, 
                    self.pca_operators[name][ig].mean_)
        else:
            componets_ = []
            variances_ = []
            means_ = []
            for ig in range(len(self.pca_operators[name])):
                componets_.append(self.pca_operators[name][ig].components_)
                variances_.append(self.pca_operators[name][ig].explained_variance_)
                means_.append(self.pca_operators[name][ig].mean_)
            return (torch.stack(componets_), torch.stack(variances_), torch.stack(means_))

    def get_pca_operators(self, name):
        return self.pca_operators[name]
    
    def set_pca_operator(self, name, ig, pca_operator):
        self.pca_operators[name][ig] = pca_operator

    def save_pca_cache(self, save_path):
        pca_cache = {}
        for n in self.names:
            pca_cache[n] = self.get_pca_operators(n)

        LOGGER.info(f"Saving PCA cache to {save_path}")
        with open(save_path, "wb") as f:
            joblib.dump(pca_cache, f)
    
    def load_pca_cache(self, load_path):
        with open(load_path, "rb") as f:
            pca_cache = joblib.load(f)
        for n in self.names:
            for ig in range(len(pca_cache[n])):
                self.set_pca_operator(n, ig, pca_cache[n][ig])
        LOGGER.info(f"Loaded PCA cache from {load_path}")


class PCAHookerWithBboxes(PCAHooker):
    def __init__(self, model, layers, modules=None, bboxes=None, device="cuda", check=False, unfold=False):
        super().__init__(model, layers, modules, device, check, unfold)
        self.bboxes = bboxes
        
    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        """Use tensor operation to accelerate the extraction of feature indices within bounding boxes
        """
        # Collect all bounding box information
        all_bboxes = []
        batch_ids = []
        for _batch_id, _bboxes in enumerate(self.bboxes):
            for _bbox in _bboxes:
                all_bboxes.append(_bbox)
                batch_ids.append(_batch_id)
        
        # If boxes in this paticular batch is empty, return an empty tensor
        if len(all_bboxes) == 0:
            return torch.tensor([], device=self.device)

        # Convert to tensor for vectorized calculation
        bbox_tensor = torch.tensor(all_bboxes, device=self.device)  # [N, 4]
        
        # Scale bbox coordinates to feature map size
        feat_coords = bbox_tensor * torch.tensor([w_out, h_out, w_out, h_out], device=self.device)
        feat_x_min = torch.clamp(feat_coords[:, 0].int(), 0, w_out-1)
        feat_y_min = torch.clamp(feat_coords[:, 1].int(), 0, h_out-1)
        feat_x_max = torch.clamp(feat_coords[:, 2].int(), 0, w_out-1)
        feat_y_max = torch.clamp(feat_coords[:, 3].int(), 0, h_out-1)

        # Generate x and y ranges
        sample_feature_indices = []
        for i in range(len(feat_y_min)):
            y_ranges = torch.arange(feat_y_min[i].item(), feat_y_max[i].item()+1, device=self.device)
            x_ranges = torch.arange(feat_x_min[i].item(), feat_x_max[i].item()+1, device=self.device)

            # Generate grid indices
            grid = torch.meshgrid(y_ranges, x_ranges)
            grid = grid[0].flatten().tolist(), grid[1].flatten().tolist()

            # Calculate feature indices
            for grid_y, grid_x in zip(grid[0], grid[1]):
                _batch_id = batch_ids[i]
                sample_feature_indices.append(_batch_id * h_out * w_out + grid_y * w_out + grid_x)
        sample_feature_indices = torch.tensor(sample_feature_indices, device=self.device)

        # If too many features, randomly sample to accelerate the PCA computation
        if len(sample_feature_indices) > 100:
            sampled_indices = torch.randperm(len(sample_feature_indices), device=self.device)[:100]
            sample_feature_indices = sample_feature_indices[sampled_indices]
        
        return sample_feature_indices


def do_pca(model, layers, modules, sample_dir=None, label_dir=None,
           device="cuda", check=False, pca_cache_save_path=None, sample_num=100,
           unfold=False):
    # Create PCA Hooker
    if label_dir is not None:
        pca_hooker = PCAHookerWithBboxes(model, layers, modules, None, device, check, unfold)
    else:
        pca_hooker = PCAHooker(model, layers, modules, device, check, unfold)

    memory_monitor = RealTimeMemoryMonitor(update_interval=0.2) # Monitor memory and CUDA memory usage
    pbar = tqdm(range(sample_num), desc="PCA computing", total=sample_num)
    memory_monitor.set_progress_bar(pbar)
    memory_monitor.start_monitoring()

    if sample_dir is not None:
        image_extensions = ['jpg', 'png', 'jpeg', 'bmp']
        sample_files = []
        if isinstance(sample_dir, list) or isinstance(sample_dir, tuple):
            for _dir in sample_dir:
                for ext in image_extensions:
                    sample_files.extend(glob.glob(os.path.join(_dir, f'*.{ext.lower()}')))
                    sample_files.extend(glob.glob(os.path.join(_dir, f'*.{ext.upper()}')))
        else:
            for ext in image_extensions:
                sample_files.extend(glob.glob(os.path.join(sample_dir, f'*.{ext.lower()}')))
                sample_files.extend(glob.glob(os.path.join(sample_dir, f'*.{ext.upper()}')))
        random.shuffle(sample_files)
        sample_files = sample_files[:sample_num]
        
        if label_dir is not None:
            label_files = []
            for _sample_file in sample_files:
                _label_name = os.path.basename(_sample_file).split('.')[0] + '.txt'
                if isinstance(label_dir, list) or isinstance(label_dir, tuple):
                    exist_label_file = False
                    for _dir_label in label_dir:
                        if os.path.exists(os.path.join(_dir_label, _label_name)):
                            label_files.append(os.path.join(_dir_label, _label_name))
                            exist_label_file = True
                            break
                    if not exist_label_file:
                        label_files.append(None)
                        LOGGER.warning(f"Label file {_label_name} not found in {label_dir}")
                else:
                    if os.path.exists(os.path.join(label_dir, _label_name)):
                        label_files.append(os.path.join(label_dir, _label_name))
                    else:
                        label_files.append(None)
                        LOGGER.warning(f"Label file {_label_name} not found in {label_dir}")
        
        for i in pbar:
            image = cv2.imread(sample_files[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).float()
            image = image.to(device)

            if label_dir is not None:
                bboxes = []
                if label_files[i] is not None:
                    with open(label_files[i], "r") as f:
                        labels = f.readlines()
                        for _label in labels:
                            _label = _label.strip().split()
                            x, y, w, h = float(_label[1]), float(_label[2]), float(_label[3]), float(_label[4])
                            x_min, y_min, x_max, y_max = x - w/2, y - h/2, x + w/2, y + h/2
                            bboxes.append([x_min, y_min, x_max, y_max])
                pca_hooker.set_bboxes([bboxes])
            
            pca_hooker.register_hook()
            with torch.no_grad():
                _ = model(image.unsqueeze(0))
            pca_hooker.remove_handle_()
    else:
        LOGGER.warning("No sample images provided, using random images for PCA")
        for i in pbar:
            image = torch.randn(3, 640, 640).to(device)
            pca_hooker.register_hook()
            with torch.no_grad():
                _ = model(image.unsqueeze(0))
            pca_hooker.remove_handle_()
    
    memory_monitor.stop_monitoring()
    pca_hooker.clear_feature_cache()
    
    if pca_cache_save_path:
        pca_hooker.save_pca_cache(pca_cache_save_path)


def main(args):
    # test cuda available
    if "cuda" in args.device and not torch.cuda.is_available():
        LOGGER.warning(f"{args.device} is not available, using cpu instead")
        args.device = "cpu"

    # Load model
    model = YOLO(args.model).model.to(args.device).eval()

    # Get layers
    if args.layers is None and args.modules is None:
        layers = list(range(len(model.model)))
    else:
        layers = args.layers

    # Get module names directly
    if args.modules is not None:
        modules = args.modules
    else:
        modules = None

    # Do PCA
    do_pca(model, layers, modules, args.sample_dir, args.label_dir,
           args.device, args.check, args.save_path, args.sample_num,
           args.unfold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
        help="Layers to calculate PCA for, use comma to separate, conv modules within layers are analyzed.")
    parser.add_argument("--modules", nargs="+", type=str, default=None,
        help="Modules to calculate PCA for, use comma to separate, providing more detailed control over the modules to calculate PCA.")
    parser.add_argument("--unfold", action="store_true",
        help="Unfold the input feature before calculating PCA.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    main(args)