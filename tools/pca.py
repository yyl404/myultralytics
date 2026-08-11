"""Perform PCA analysis on intermediate layer input feature activations of YOLO models.

Usage:
    $ python tools/pca.py \
        --model <path/to/model.pt> \
        --sample_dir <path/to/sample_dir> \
        --save_path <path/to/save_path> \
        [--sample_num <sample_num> \
         --label_dir <label_dir> \
         --layers <layer1> <layer2> ... \
         --modules <module1> <module2> ... \
         --mode <mode> \
         --check]

Arguments:
    --model: Path to the model checkpoint (.pt file)
    --sample_dir: Path to the sample images directory for PCA computation
    --save_path: Path to save the PCA results file
    --sample_num: Maximum number of samples to use (default: 100)
    --label_dir: (optional) Path to the label directory for bounding box-based feature sampling
    --layers: (optional) Layers to calculate PCA for, space-separated. If not specified, all intermediate layers are analyzed
    --modules: (optional) Specific module names to calculate PCA for, space-separated. Provides more detailed control than --layers
    --mode: (optional) Mode to calculate PCA, choices: unfold (default), fold
    --check: (optional) Check the correctness of convolution kernel unfolding operations
    --device: Device to use (default: "cuda")

Examples:
    $ python tools/pca.py \
        --model yolov8n.pt \
        --sample_dir data/images/train \
        --save_path pca_results.pkl \
        --sample_num 200 \
        --layers 10 11 12
    
    $ python tools/pca.py \
        --model yolov8n.pt \
        --sample_dir data/images/train \
        --save_path pca_results.pkl \
        --modules model.10.conv model.11.conv \
        --label_dir data/labels/train \
        --mode unfold \
        --check
"""

import joblib
import os.path as OSP
import random
import cv2
from tqdm import tqdm
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils import (
    LOGGER, YAML
)

from pca_on_gpu import IncrementalPCAonGPU as IncrementalPCA
from pca_on_gpu import UncenteredPCAonGPU as UncenteredPCA
from utils import RealTimeMemoryMonitor


def conv_meta(module):
    if isinstance(module, nn.Conv2d):
        return {
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "groups": module.groups,
            "dilation": module.dilation,
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
        }
    # CosineConv2d is a 1x1 dense classifier (no groups/dilation/padding by design).
    return {
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "groups": 1,
        "dilation": (1, 1),
        "in_channels": module.weight.shape[1],
        "out_channels": module.weight.shape[0],
    }


def is_supported_conv_module(module):
    return isinstance(module, nn.Conv2d)


class PCAHooker:
    def __init__(
        self, model, layers=None, modules=None, device="cuda", check=False, unfold=True, uncentered=False
    ):
        self.model = model
        self.modules = {}
        self.pca_operators = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        self.check = check
        self.unfold = unfold
        self.uncentered = uncentered

        def _build_operator(module, input_dim):
            if self.uncentered:
                return UncenteredPCA(n_components=input_dim, device=self.device)
            return IncrementalPCA(n_components=input_dim)
        
        def _match(n, m, lid):
            "dfl layer is always frozen, so we don't need to calculate PCA for it"
            return f"model.{lid}." in n and is_supported_conv_module(m) and "dfl" not in n

        self.feature_caches, self._handles = {}, []

        if modules is not None:
            # If modules are provided, only calculate PCA for the specified modules
            for n, m in model.named_modules():
                if n in modules:
                    meta = conv_meta(m)
                    k, c_in, g = meta["kernel_size"], meta["in_channels"], meta["groups"]
                    self.modules[n] = m
                    self.pca_operators[n] = []
                    for i in range(g):
                        if self.unfold:
                            n_components = c_in//g*k[0]*k[1]
                        else:
                            n_components = c_in//g
                        self.pca_operators[n].append(_build_operator(m, n_components))
                    self.feature_caches[n] = []
        elif layers is not None:
            for lid in layers:
                # If layers are provided, calculate PCA for all conv modules within layers
                for n, m in model.named_modules():
                    if _match(n, m, lid):
                        meta = conv_meta(m)
                        k, c_in, g = meta["kernel_size"], meta["in_channels"], meta["groups"]
                        self.modules[n] = m
                        self.pca_operators[n] = []
                        for i in range(g):
                            if self.unfold:
                                n_components = c_in//g*k[0]*k[1]
                            else:
                                n_components = c_in//g
                            self.pca_operators[n].append(_build_operator(m, n_components))
                        self.feature_caches[n] = []
        else:
            raise ValueError("Either modules or layers must be provided")

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        if self.uncentered:
            return torch.arange(bs * h_out * w_out, device=self.device)
        # Randomly sample feature indices for the legacy PCA/ESPReg path.
        sample_feature_indices = torch.randperm(bs*h_out*w_out, device=self.device)[:100]
        return sample_feature_indices

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(n, self.unfold)))

    def _hook(self, module_name, unfold=True):
        """
        If unfold is True, the input feature will be unfolded into a sequence of vectors with shape [c_in*k[0]*k[1], bs*h_out*w_out],
        otherwise, the input feature will be reshaped into a matrix with shape [c_in, bs*h_out*w_out].
        """
        def fn(module, feat_in, feat_out):
            if is_supported_conv_module(module):
                meta = conv_meta(module)
                k, s, p, g, d, c_in, c_out = (
                    meta["kernel_size"],
                    meta["stride"],
                    meta["padding"],
                    meta["groups"],
                    meta["dilation"],
                    meta["in_channels"],
                    meta["out_channels"],
                )

                feat_in = feat_in[0]  # Module may accept multiple input features, and we only extract the first
                if self.uncentered:
                    feat_in = feat_in.mean(dim=0, keepdim=True)
                bs, _, h_in_raw, w_in_raw = feat_in.shape
                if p[0] > 0 or p[1] > 0:
                    feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
                else:
                    feat_in_padded = feat_in
                _, _, h_in, w_in = feat_in_padded.shape
                h_out, w_out = feat_out.shape[2], feat_out.shape[3]

                # Group the input features
                c_in_grouped = c_in//g

                if unfold:
                    feat_in_padded_grouped = feat_in_padded.reshape(bs, g, c_in_grouped, h_in, w_in)

                    # Use the sliding window with the same settings as convolution kernels
                    # to unfold input features into a sequence of vectors, considering dilation
                    # For dilated convolution, we need to unfold with larger window size
                    # Effective kernel size becomes: k[0] + (k[0]-1)*(d[0]-1), k[1] + (k[1]-1)*(d[1]-1)
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
                    if module.bias is not None and not self.uncentered:
                        feat_reshaped = torch.concat((feat_reshaped, torch.ones(g, 1, bs*h_out*w_out).to(feat_reshaped.device)), dim=1)
                    pos_h, pos_w = h_out, w_out

                    # The following code is used to check whether the unfolding representation of convolution operation
                    # is equivalent with the original convolution operation
                    if self.check:
                        weight = module.weight.data.reshape(g, c_out//g, -1)  # [g, c_out//g, c_in//g*actual_k_h*actual_k_w]
                        if module.bias is not None:
                            weight = torch.concat((weight, module.bias.data.reshape(g, c_out//g, 1)), dim=2)
                        feat_out_reshaped = weight @ feat_reshaped  # [g, c_out//g, c_in//g*actual_k_h*actual_k_w] @ [c_in//g*actual_k_h*actual_k_w, bs*h_out*w_out] --> [g, c_out//g, bs*h_out*w_out]
                        feat_out_reshaped = feat_out_reshaped.reshape(c_out, -1)  # [c_out, bs*h_out*w_out]

                        # [c_out, bs*h_out*w_out] --> [c_out, bs, h_out, w_out] --> [bs, c_out, h_out, w_out]
                        feat_out_reshaped_reversed = feat_out_reshaped.view(-1, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()

                        LOGGER.info(f"Module {module_name}'s unfolding error: {F.mse_loss(feat_out, feat_out_reshaped_reversed)}")
                else:
                    # Fold mode: channel vectors at every input position, [g, c_in//g, bs*h_in*w_in]
                    feat_fold = feat_in.reshape(bs, g, c_in_grouped, h_in_raw * w_in_raw)
                    feat_reshaped = feat_fold.permute(1, 2, 0, 3).reshape(g, c_in_grouped, bs * h_in_raw * w_in_raw)
                    pos_h, pos_w = h_in_raw, w_in_raw

                sample_feature_indices = self._get_sample_feature_indices(bs, pos_h, pos_w)
                if sample_feature_indices.shape[0] == 0:
                    # Some batches may have no bounding boxes, so we need to return here
                    return
                if sample_feature_indices.max() >= feat_reshaped.shape[2]:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.max()} >= {feat_reshaped.shape[2]}")
                if sample_feature_indices.min() < 0:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.min()} < 0")
                feat_sampled = feat_reshaped[:, :, sample_feature_indices]
                # unfold true: [g, c_in//g*k*k, len(sample_feature_indices)] | unfold false: [g, c_in//g, len(sample_feature_indices)]
                
                feature_cache = self.feature_caches[module_name]
                pca_operators = self.pca_operators[module_name]
                if self.uncentered:
                    for group_idx in range(g):
                        pca_operators[group_idx].partial_fit(feat_sampled[group_idx].T)
                    torch.cuda.empty_cache()
                    return

                feature_cache.append(feat_sampled)
                # Incremental PCA requires the first batch's size is larger than n_components
                if sum([x.shape[2] for x in feature_cache]) >= pca_operators[0].n_components:
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
                    # Before clearing the cache, we need to fit the PCA operator with the final batch of features
                    if hasattr(self.pca_operators[n][ig], 'components_'):
                        self.pca_operators[n][ig].partial_fit(torch.cat(cache, dim=2)[ig].T)
                    else:
                        # No components_ means PCA hasn't been fitted yet (samples < n_components), 
                        # use normal PCA instead of Incremental PCA
                        LOGGER.info(f"Too few samples to fit PCA in module {n}. Use normal PCA instead.")
                        self.pca_operators[n][ig].fit(torch.cat(cache, dim=2)[ig].T)
            cache.clear()
            for operator in self.pca_operators[n]:
                if hasattr(operator, "finalize"):
                    operator.finalize()
    
    def get_pca_results(self, name, ig=None):
        if ig is not None:
            return (self.pca_operators[name][ig].components_, 
                    self.pca_operators[name][ig].explained_variance_)
        else:
            componets_ = []
            variances_ = []
            for ig in range(len(self.pca_operators[name])):
                componets_.append(self.pca_operators[name][ig].components_)
                variances_.append(self.pca_operators[name][ig].explained_variance_)
            return (torch.stack(componets_), torch.stack(variances_))

    def get_pca_operators(self, name):
        return self.pca_operators[name]
    
    def set_pca_operator(self, name, ig, pca_operator):
        self.pca_operators[name][ig] = pca_operator

    def save_pca_cache(self, save_path):
        """Save PCA operators in a device-agnostic (CPU-serialized) form.

        Tensor attributes are moved to CPU before dumping so that the artifact
        can be loaded on any device, regardless of the device used to compute it.
        """
        pca_cache = {}
        for n in self.names:
            operators = self.get_pca_operators(n)
            for ig, operator in enumerate(operators):
                if not hasattr(operator, "components_"):
                    raise RuntimeError(
                        f"PCA operator for module '{n}' group {ig} was never fitted "
                        f"(missing components_). Check that enough boxed samples" 
                        f"were collected."
                    )
            pca_cache[n] = [operator.to("cpu") for operator in operators]

        LOGGER.info(f"Saving PCA cache to {save_path}")
        with open(save_path, "wb") as f:
            joblib.dump(pca_cache, f)
    
    def load_pca_cache(self, load_path):
        """Load PCA cache and use it as initial state.

        Operators are moved to this hooker's device after loading, so caches
        serialized on a different device (including CPU-serialized caches)
        can be used seamlessly.

        Args:
            load_path: Path to PCA cache file
        """
        with open(load_path, "rb") as f:
            pca_cache = joblib.load(f)
        
        for n in self.names:
            if n in pca_cache:
                if len(pca_cache[n]) != len(self.pca_operators[n]):
                    LOGGER.warning(
                        f"Module {n}: PCA cache has {len(pca_cache[n])} groups, "
                        f"but expected {len(self.pca_operators[n])} groups. "
                        f"This may indicate errors."
                    )
                    continue
                for ig in range(min(len(self.pca_operators[n]), len(pca_cache[n]))):
                    self.set_pca_operator(n, ig, pca_cache[n][ig].to(self.device))
        
        LOGGER.info(f"Loaded PCA cache from {load_path}")


class PCAHookerWithBboxes(PCAHooker):
    def __init__(
        self,
        model,
        layers,
        modules=None,
        bboxes=None,
        device="cuda",
        check=False,
        unfold=True,
        uncentered=False,
    ):
        super().__init__(model, layers, modules, device, check, unfold, uncentered)
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
        
        # If boxes in this particular batch is empty, return an empty tensor
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
            grid = torch.meshgrid(y_ranges, x_ranges, indexing='ij')
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


def do_pca(model, layers, modules, sample_dir=None, label_dir=None, device="cuda",
           check=False, pca_cache_save_path=None, sample_num=100, unfold=True, load_hist=None,
           uncentered=False, batch_size=1):
    if batch_size < 1:
        raise ValueError(f"PCA batch_size must be positive, got {batch_size}")
    # Create PCA Hooker
    if label_dir is not None and not uncentered:
        pca_hooker = PCAHookerWithBboxes(
            model, layers, modules, None, device, check, unfold, uncentered
        )
    else:
        pca_hooker = PCAHooker(model, layers, modules, device, check, unfold, uncentered)
    
    # Load historical PCA cache if specified
    if load_hist is not None:
        if OSP.exists(load_hist):
            pca_hooker.load_pca_cache(load_hist)
        else:
            LOGGER.warning(f"Historical PCA cache file not found: {load_hist}. Starting from scratch.")

    memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)  # Monitor memory and CUDA memory usage
    if sample_dir is not None:
        image_extensions = ['jpg', 'png', 'jpeg', 'bmp']
        sample_files = []
        if isinstance(sample_dir, list) or isinstance(sample_dir, tuple):
            for _dir in sample_dir:
                for ext in image_extensions:
                    sample_files.extend(glob.glob(OSP.join(_dir, f'*.{ext.lower()}')))
                    sample_files.extend(glob.glob(OSP.join(_dir, f'*.{ext.upper()}')))
        else:
            for ext in image_extensions:
                sample_files.extend(glob.glob(OSP.join(sample_dir, f'*.{ext.lower()}')))
                sample_files.extend(glob.glob(OSP.join(sample_dir, f'*.{ext.upper()}')))
        random.shuffle(sample_files)
        
        if label_dir is not None:
            label_files = []
            for _sample_file in sample_files:
                _label_name = OSP.splitext(OSP.basename(_sample_file))[0] + '.txt'
                if isinstance(label_dir, list) or isinstance(label_dir, tuple):
                    exist_label_file = False
                    for _dir_label in label_dir:
                        if OSP.exists(OSP.join(_dir_label, _label_name)):
                            label_files.append(OSP.join(_dir_label, _label_name))
                            exist_label_file = True
                            break
                    if not exist_label_file:
                        label_files.append(None)
                        LOGGER.warning(f"Label file {_label_name} not found in {label_dir}")
                else:
                    if OSP.exists(OSP.join(label_dir, _label_name)):
                        label_files.append(OSP.join(label_dir, _label_name))
                    else:
                        label_files.append(None)
                        LOGGER.warning(f"Label file {_label_name} not found in {label_dir}")
        
        
        sample_count = len(sample_files) if sample_num <= 0 else min(sample_num, len(sample_files))
        step_size = batch_size if uncentered else 1
        sample_starts = range(0, sample_count, step_size)
        pbar = tqdm(sample_starts, desc="PCA computing", total=len(sample_starts))
        memory_monitor.set_progress_bar(pbar)
        memory_monitor.start_monitoring()
        for sample_start in pbar:
            sample_indices = range(sample_start, min(sample_start + step_size, sample_count))
            images = []
            for sample_idx in sample_indices:
                image = cv2.imread(sample_files[sample_idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (640, 640))
                image = image.transpose(2, 0, 1) / 255.0
                images.append(torch.from_numpy(image).float())
            image_batch = torch.stack(images).to(device)

            if label_dir is not None and not uncentered:
                bboxes = []
                if label_files[sample_start] is not None:
                    with open(label_files[sample_start], "r") as f:
                        labels = f.readlines()
                        for _label in labels:
                            _label = _label.strip().split()
                            x, y, w, h = float(_label[1]), float(_label[2]), float(_label[3]), float(_label[4])
                            x_min, y_min, x_max, y_max = x - w/2, y - h/2, x + w/2, y + h/2
                            bboxes.append([x_min, y_min, x_max, y_max])
                pca_hooker.set_bboxes([bboxes])
            
            pca_hooker.register_hook()
            with torch.no_grad():
                _ = model(image_batch)
            pca_hooker.remove_handle_()
    else:
        LOGGER.warning("No sample images provided, using random images for PCA")
        pbar = tqdm(range(sample_num), desc="PCA computing", total=sample_num)
        memory_monitor.set_progress_bar(pbar)
        memory_monitor.start_monitoring()
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
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    # Test CUDA availability
    if "cuda" in args.device and not torch.cuda.is_available():
        LOGGER.warning(f"{args.device} is not available, using cpu instead")
        args.device = "cpu"

    # Load model
    model = YOLO(args.model).model.to(args.device).eval()

    # Get layers
    if args.layers is None and args.modules is None:
        end_layer = len(model.model) - 1 if args.exclude_head else len(model.model)
        layers = list(range(end_layer))
    else:
        layers = args.layers

    # Get module names directly
    if args.modules is not None:
        modules = args.modules
    else:
        modules = None

    # If specifying samples and labels by --dataset, get the sample dir and label dir
    if args.dataset is not None:
        sample_dirs = []
        label_dirs = []
        for dataset_path in args.dataset:
            dataset_config = YAML.load(dataset_path)
            if 'path' not in dataset_config.keys():
                sample_dir = OSP.join(OSP.dirname(dataset_path), dataset_config['train'])
            else:
                sample_dir = OSP.join(dataset_config['path'], dataset_config['train'])
            label_dir = sample_dir.replace("images", "labels")
            sample_dirs.append(sample_dir)
            label_dirs.append(label_dir)
        # Convert to lists (do_pca supports both list and string, but list is more consistent)
        args.sample_dir = sample_dirs
        args.label_dir = label_dirs
    if args.dataset is None and args.sample_dir is None:
        raise ValueError("Either --dataset or --sample_dir must be provided")

    # Mode
    if args.mode == "unfold":
        unfold = True
    elif args.mode == "fold":
        unfold = False
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Perform PCA
    do_pca(
        model,
        layers,
        modules,
        args.sample_dir,
        args.label_dir,
        args.device,
        args.check,
        args.save_path,
        args.sample_num,
        unfold,
        args.load_hist,
        args.uncentered,
        args.batch_size,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--dataset", nargs="+", type=str, default=None,
        help="Dataset YAML configuration file(s). Can specify multiple datasets.")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
        help="Layers to calculate PCA for, use comma to separate, conv modules within layers are analyzed.")
    parser.add_argument("--modules", nargs="+", type=str, default=None,
        help="Modules to calculate PCA for, use comma to separate, providing more detailed control over the "+
        "modules to calculate PCA.")
    parser.add_argument("--exclude_head", action="store_true",
        help="Exclude the final detection head; NSGP governs the feature extractor only.")
    parser.add_argument("--uncentered", action="store_true",
        help="Accumulate uncentered input covariance as required by NSGP.")
    parser.add_argument("--mode", type=str, choices=["unfold", "fold"], default="unfold",
        help="Mode to calculate PCA, choices: unfold (default), fold.")
    parser.add_argument("--load_hist", type=str, default=None,
        help="Optional path to historical PCA cache file to load as initial state. "
             "Only modules that exist in both current and historical cache will be loaded.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    main(args)