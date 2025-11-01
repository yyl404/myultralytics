"""
Visualize the projection of kernel updates on PCA components.

This script provides three visualization stages for analyzing incremental learning:
    Stage 1: Variance and weight update projection histograms
        - Shows PCA explained variance, rotated means, and cosine similarities
        - Generates histogram plots for each module
    Stage 2: Weight projection polar plots
        - Visualizes kernel updates in polar coordinates
        - Shows projections at different variance milestones (75%, 90%, 95%, 99%)
    Stage 3: Value shift analysis
        - Analyzes value shifts between base and incremental models
        - Requires sample images and optionally label files for bounding box selection
        - Generates horizontal bar charts showing value shift statistics

Usage:
    # Full analysis with all stages (default)
    python proj_vis.py --pca_cache_path <pca_cache_path> \
        --base_model <base_model> --incremental_model <incremental_model> \
        --save_dir <save_dir>
    
    # Only Stage 3 (value shift analysis)
    python proj_vis.py --pca_cache_path <pca_cache_path> \
        --base_model <base_model> --incremental_model <incremental_model> \
        --save_dir <save_dir> --stages 3 --sample_dir <sample_dir> [--label_dir <label_dir>]
    
    # Only Stage 1 and 2 (visualization without value shift)
    python proj_vis.py --pca_cache_path <pca_cache_path> \
        --base_model <base_model> --incremental_model <incremental_model> \
        --save_dir <save_dir> --stages 1 2

Examples:
    # Full analysis
    python proj_vis.py \
        --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
        --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
        --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
        --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1 \
        --sample_dir data/val/images --label_dir data/val/labels
    
    # Only value shift analysis with detailed visualization
    python proj_vis.py \
        --pca_cache_path runs/yolov8l_voc_inc_15_5_fromscratch/task-1/pca_cache.joblib \
        --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
        --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
        --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1 \
        --stages 3 --sample_dir data/val/images --sample_num 200

Required Arguments:
    --pca_cache_path: Path to the PCA cache file (joblib format) containing PCA operators
                      for each convolutional module.
    --save_dir: Directory to save all generated plots and analysis results.

Optional Arguments:
    --base_model: Path to the base model checkpoint file (.pt format).
                  Required for Stages 1, 2, and 3.
    --incremental_model: Path to the incremental model checkpoint file (.pt format).
                        Required for Stages 1, 2, and 3.
    --stages: List of stages to enable. Options: 1 (histograms), 2 (polar plots), 3 (value shift).
             Default: [1, 2, 3] (all stages).
    --unfold: If set, use unfolded kernel representation [c_out, c_in*k[0]*k[1]].
              Otherwise use [c_out*k[0]*k[1], c_in]. Default: False.
    --detailed: If set, plot detailed per-kernel curves in histograms. Default: False.
    --k: Number of top-k kernels to display in detailed mode. Default: 15.
    --sample_dir: Directory or directories containing sample images for Stage 3.
                  Supports multiple directories (space-separated). Required for Stage 3.
    --label_dir: Directory or directories containing label files (YOLO format) for Stage 3.
                 Used for bounding box-based feature selection. Optional.
    --sample_num: Number of samples to use for value shift analysis. Default: 100.

Output:
    For Stage 1:
        - <module_name>_histogram.png: Histogram plots showing variances, means, and cosine similarities
    
    For Stage 2:
        - <module_name>_polar_projection.png: Polar coordinate plots showing kernel update projections
    
    For Stage 3:
        - value_shift_analysis.png: Horizontal bar chart showing value shift statistics across modules

Notes:
    - Models are automatically moved to CUDA if available, otherwise CPU is used.
    - For grouped convolutions, separate plots are generated for each group.
    - Stage 3 requires both models and sample images to function properly.
    - Supported image formats: jpg, png, jpeg, bmp (case-insensitive).
"""

import argparse
import glob
import os
import random
import warnings
from pathlib import Path
from tqdm import tqdm
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


from ultralytics import YOLO

# Constants
MILESTONES = [0.75, 0.90, 0.95, 0.99]
MILESTONE_COLORS = ['red', 'orange', 'green', 'purple']
IMAGE_EXTENSIONS = ['jpg', 'png', 'jpeg', 'bmp']
DEFAULT_IMAGE_SIZE = (640, 640)
MAX_POLAR_GROUPS = 8


def flatten_kernel(module, unfold=False, zeros=False):
    """Flatten convolution kernel weights.
    
    Args:
        module: nn.Conv2d module
        unfold: If True, reshape to [c_out, c_in*k[0]*k[1]], else [c_out*k[0]*k[1], c_in]
        zeros: If True, return zeros tensor with same shape
    
    Returns:
        Flattened kernel tensor or list of tensors for grouped convolutions
    """
    assert isinstance(module, nn.Conv2d), "Module must be a nn.Conv2d module"
    
    weight = module.weight.data  # [c_out, c_in//g, k[0], k[1]]
    groups = module.groups
    
    if groups == 1:
        # Standard convolution
        if unfold:
            kernel = weight.reshape(weight.shape[0], -1)
        else:
            kernel = weight.permute(0, 2, 3, 1).flatten(0, 2)
        return torch.zeros_like(kernel) if zeros else kernel
    else:
        # Grouped convolution - return list of kernels for each group
        kernels = []
        c_out_per_group = weight.shape[0] // groups
        
        for g in range(groups):
            start_out = g * c_out_per_group
            end_out = (g + 1) * c_out_per_group
            group_weight = weight[start_out:end_out]
            
            if unfold:
                group_kernel = group_weight.reshape(group_weight.shape[0], -1)
            else:
                group_kernel = group_weight.permute(0, 2, 3, 1).flatten(0, 2)
            
            kernels.append(torch.zeros_like(group_kernel) if zeros else group_kernel)
        
        return kernels


def create_hook(intermediate_results, name):
    """Create a forward hook to capture intermediate results.
    
    Args:
        intermediate_results: Dictionary to store input and output
        name: module name
    
    Returns:
        Hook function
    """
    def hook_fn(module, feat_in, feat_out):
        intermediate_results[name]["input"] = feat_in
        intermediate_results[name]["output"] = feat_out
        if "call_cnt" not in intermediate_results.keys():
            intermediate_results["call_cnt"] = 0
        if intermediate_results["call_cnt"] != -1:
            intermediate_results[name]["call_idx"] = intermediate_results["call_cnt"]
            intermediate_results["call_cnt"] += 1
    return hook_fn


def window_cumsum(x, window, dim):
    """Calculate the cumulative sum of a tensor with a window.
    
    Optimized: If window==1, directly return the input tensor to avoid
    unnecessary computation.
    
    Args:
        x: Input tensor
        window: Window size
        dim: Dimension to apply the window cumsum
    
    Returns:
        Window cumulative sum of the input tensor
    """
    # Optimization: window=1 means no actual windowing, return as-is
    if window == 1:
        return x
    
    num_dims = len(x.shape)
    
    # Move the target dimension to the last position for easier processing
    if dim != num_dims - 1:
        perm = list(range(num_dims))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_permuted = x.permute(perm)
    else:
        x_permuted = x
        perm = None
    
    # Reshape to 2D for batch processing
    original_shape = x_permuted.shape
    x_2d = x_permuted.reshape(-1, original_shape[-1])
    
    # Apply window cumulative sum using convolution
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    padding = window - 1
    x_padded = F.pad(x_2d.unsqueeze(1), (0, padding), mode='constant', value=0)
    result = F.conv1d(x_padded, kernel, stride=1, padding=0).squeeze(1)
    
    # Reshape back to original shape
    result = result.reshape(original_shape)
    
    # Move the dimension back to its original position
    if perm is not None:
        result = result.permute(perm)
    
    return result


def _get_feature_indices_from_boxes(bboxes, batch_size, h, w, device):
    """Extract feature indices within bounding boxes.
    
    Args:
        bboxes: Bounding boxes in format [batch][bbox_idx][x_min, y_min, x_max, y_max]
        batch_size: Batch size
        h: Feature map height
        w: Feature map width
        device: Device for tensor creation
    
    Returns:
        Tensor of feature indices
    """
    if not bboxes or len(bboxes) == 0:
        return torch.arange(batch_size * h * w, device=device)
    
    # Normalize bboxes format
    if isinstance(bboxes[0], (int, float)):
        bboxes = [[bboxes]]
    elif isinstance(bboxes[0], list) and bboxes[0] and isinstance(bboxes[0][0], (int, float)):
        bboxes = [bboxes]
    
    # Collect all feature indices within bboxes
    all_indices = []
    for batch_id in range(batch_size):
        if batch_id < len(bboxes) and bboxes[batch_id] is not None:
            for bbox in bboxes[batch_id]:
                x_min, y_min, x_max, y_max = bbox
                
                # Scale bbox coordinates to feature map size
                feat_x_min = max(0, min(w - 1, int(x_min * w)))
                feat_y_min = max(0, min(h - 1, int(y_min * h)))
                feat_x_max = max(0, min(w - 1, int(x_max * w)))
                feat_y_max = max(0, min(h - 1, int(y_max * h)))
                
                # Generate indices for all pixels in the box
                for y in range(feat_y_min, feat_y_max + 1):
                    for x in range(feat_x_min, feat_x_max + 1):
                        idx = batch_id * h * w + y * w + x
                        all_indices.append(idx)
    
    if not all_indices:
        return torch.arange(batch_size * h * w, device=device)
    
    return torch.tensor(all_indices, device=device, dtype=torch.long)


def _map_input_indices_to_output(input_indices, batch_size, h_in, w_in, h_out, w_out,
                                  stride, padding, kernel_size, device):
    """Map input feature indices to corresponding output indices.
    
    Args:
        input_indices: Tensor of input feature indices
        batch_size: Batch size
        h_in, w_in: Input feature map dimensions
        h_out, w_out: Output feature map dimensions
        stride: Convolution stride (tuple)
        padding: Convolution padding (tuple)
        kernel_size: Convolution kernel size (tuple)
        device: Device for tensor creation
    
    Returns:
        Tensor of output feature indices
    """
    # Convert flat indices to (batch, y, x) coordinates
    batch_ids = input_indices // (h_in * w_in)
    spatial_indices = input_indices % (h_in * w_in)
    y_in = spatial_indices // w_in
    x_in = spatial_indices % w_in
    
    output_indices_set = set()
    for i in range(len(input_indices)):
        b = batch_ids[i].item()
        y = y_in[i].item()
        x = x_in[i].item()
        
        # Calculate output positions this input contributes to
        for ky in range(kernel_size[0]):
            for kx in range(kernel_size[1]):
                y_padded = y + padding[0]
                x_padded = x + padding[1]
                
                if (y_padded - ky) % stride[0] == 0 and (x_padded - kx) % stride[1] == 0:
                    y_out = (y_padded - ky) // stride[0]
                    x_out = (x_padded - kx) // stride[1]
                    
                    if 0 <= y_out < h_out and 0 <= x_out < w_out:
                        out_idx = b * h_out * w_out + y_out * w_out + x_out
                        output_indices_set.add(out_idx)
    
    return torch.tensor(sorted(output_indices_set), device=device, dtype=torch.long)


def _get_conv_params(module):
    """Extract convolution parameters from module.
    
    Args:
        module: nn.Conv2d module or None
    
    Returns:
        Tuple of (stride, padding, kernel_size)
    """
    if module is not None and isinstance(module, nn.Conv2d):
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        return stride, padding, kernel_size
    return (1, 1), (0, 0), (1, 1)


def cal_value_shift(base_input, incremental_input, base_output, incremental_output,
                    base_kernel, incremental_kernel, name, save_dir, module=None, box=None):
    """Calculate the value shift between the base and incremental models.
    
    This function implements the following steps:
    1. Calculate input shift and reshape to [c_in, bs*h_in*w_in]
    2. Calculate output shift and reshape to [c_out, bs*h_out*w_out]
    3. Select features based on bounding boxes (if provided)
    4. Calculate kernel update
    5-7. Calculate three matrix products:
       - kernel_update @ base_input
       - base_kernel @ input_shift
       - kernel_update @ input_shift
    
    Args:
        base_input: Base model input [bs, c_in, h_in, w_in]
        incremental_input: Incremental model input [bs, c_in, h_in, w_in]
        base_output: Base model output [bs, c_out, h_out, w_out]
        incremental_output: Incremental model output [bs, c_out, h_out, w_out]
        base_kernel: Base kernel [c_out, c_in*k[0]*k[1]]
        incremental_kernel: Incremental kernel [c_out, c_in*k[0]*k[1]]
        name: Module name (unused, kept for compatibility)
        save_dir: Save directory (unused, kept for compatibility)
        module: Conv2d module for extracting parameters
        box: Bounding boxes for feature selection
    
    Returns:
        Tuple of three matrices: (kernel_update_mult_input, base_kernel_mult_input_shift,
                                kernel_update_mult_input_shift)
    """
    device = base_input.device
    
    # Get shapes
    bs, c_in, h_in, w_in = base_input.shape
    _, c_out, h_out, w_out = base_output.shape
    
    # Get convolution parameters
    stride, padding, kernel_size = _get_conv_params(module)
    
    # Calculate input shift and reshape
    input_shift = incremental_input - base_input
    base_input_reshaped = base_input.permute(1, 0, 2, 3).reshape(c_in, -1)
    input_shift_reshaped = input_shift.permute(1, 0, 2, 3).reshape(c_in, -1)
    
    # Calculate output shift and reshape
    output_shift = incremental_output - base_output
    base_output_reshaped = base_output.permute(1, 0, 2, 3).reshape(c_out, -1)
    output_shift_reshaped = output_shift.permute(1, 0, 2, 3).reshape(c_out, -1)
    
    # Select feature indices based on bounding boxes
    input_indices = _get_feature_indices_from_boxes(box, bs, h_in, w_in, device)
    output_indices = _map_input_indices_to_output(
        input_indices, bs, h_in, w_in, h_out, w_out, stride, padding, kernel_size, device
    )
    
    # Select features based on indices
    base_input_selected = base_input_reshaped[:, input_indices]
    input_shift_selected = input_shift_reshaped[:, input_indices]
    
    # Calculate kernel update
    kernel_update = incremental_kernel - base_kernel
    
    # Calculate three matrix products
    kernel_update_mult_input = kernel_update @ base_input_selected
    base_kernel_mult_input_shift = base_kernel @ input_shift_selected
    kernel_update_mult_input_shift = kernel_update @ input_shift_selected
    
    return kernel_update_mult_input, base_kernel_mult_input_shift, kernel_update_mult_input_shift


def _calculate_milestone_indices(variances_cumsum_ratio, milestones):
    """Calculate milestone indices from cumulative variance ratios.
    
    Args:
        variances_cumsum_ratio: Cumulative variance ratio tensor
        milestones: List of milestone thresholds
    
    Returns:
        List of milestone indices
    """
    milestones_indices = []
    for milestone in milestones:
        for i in range(len(variances_cumsum_ratio)):
            if variances_cumsum_ratio[i] >= milestone:
                milestones_indices.append(i)
                break
    return milestones_indices


def _calculate_pca_statistics(pca_operator, kernel_update, device, milestones):
    """Calculate PCA statistics for each group.
    
    Args:
        pca_operator: PCA operator or list of PCA operators
        kernel_update: Kernel update tensor or list of tensors
        device: Device for computation
        milestones: List of milestone thresholds
    
    Returns:
        Tuple of lists: (variances_list, means_rotated_list, milestones_indices_list,
                        cosine_similarity_windowed_list, cosine_similarity_cumsum_list,
                        kernel_update_list)
    """
    # Normalize to list format
    if isinstance(pca_operator, list):
        num_groups = len(pca_operator)
        pca_list = pca_operator
    else:
        num_groups = 1
        pca_list = [pca_operator]
    
    variances_list = []
    means_rotated_list = []
    milestones_indices_list = []
    cosine_similarity_windowed_list = []
    cosine_similarity_cumsum_list = []
    kernel_update_list = []
    
    for group_idx in range(num_groups):
        group_pca = pca_list[group_idx]
        
        # Calculate variances and milestones
        variances = group_pca.explained_variance_.to(device)
        variances_cumsum = torch.cumsum(variances, dim=0)
        variances_cumsum_ratio = variances_cumsum / variances_cumsum[-1]
        milestones_indices = _calculate_milestone_indices(variances_cumsum_ratio, milestones)
        
        # Calculate means
        means = group_pca.mean_.to(device)
        components = F.normalize(group_pca.components_.to(device), p=2, dim=1)
        means_rotated = torch.matmul(means.unsqueeze(0), components.T).squeeze(0)
        
        variances_list.append(variances)
        means_rotated_list.append(means_rotated)
        milestones_indices_list.append(milestones_indices)
        
        # Process kernel updates if available
        if kernel_update is not None:
            if isinstance(kernel_update, list):
                group_kernel_update = kernel_update[group_idx]
            else:
                group_kernel_update = kernel_update
            
            if group_kernel_update is not None and group_kernel_update.shape[0] > 0:
                # Calculate kernel update projections
                proj = torch.matmul(group_kernel_update, components.T)
                kernel_norms = torch.norm(group_kernel_update, p=2, dim=1, keepdim=True) + 1e-12
                cosine_similarity = proj / kernel_norms
                cosine_similarity_cumsum = (cosine_similarity ** 2).cumsum(dim=1).sqrt()
                
                # Window size is always 1 (optimized in window_cumsum)
                window_size = 1
                cosine_similarity_windowed = window_cumsum(
                    cosine_similarity ** 2, window=window_size, dim=1
                ).sqrt()
                
                cosine_similarity_windowed_list.append(cosine_similarity_windowed)
                cosine_similarity_cumsum_list.append(cosine_similarity_cumsum)
                kernel_update_list.append(group_kernel_update)
            else:
                cosine_similarity_windowed_list.append(None)
                cosine_similarity_cumsum_list.append(None)
                kernel_update_list.append(None)
        else:
            # PCA-only analysis: create dummy data
            dummy_cosine_similarity_windowed = torch.zeros(1, len(variances), device=device)
            cosine_similarity_windowed_list.append(dummy_cosine_similarity_windowed)
            cosine_similarity_cumsum_list.append(None)
            kernel_update_list.append(None)
    
    return (variances_list, means_rotated_list, milestones_indices_list,
            cosine_similarity_windowed_list, cosine_similarity_cumsum_list,
            kernel_update_list)


def plot_histogram(cosine_similarity_windowed_list, variances_list, means_rotated_list,
                   module_name, milestones, milestones_indices_list, save_dir,
                   detailed=False, top_k=15, kernel_update_list=None):
    """Plot histogram of PCA variances, means and windowed cumulative cosine similarities.
    
    Supports grouped convolutions. Plots are arranged horizontally with separators
    between groups.
    
    Args:
        cosine_similarity_windowed_list: List of windowed cumulative cosine similarities
            for each group. Each element shape: (num_kernels, num_components).
        variances_list: List of explained variances for each group.
        means_rotated_list: List of rotated means for each group.
        module_name: Name of the module.
        milestones: Milestones of the accumulated variance ratio.
        milestones_indices_list: List of milestone indices for each group.
        save_dir: Directory to save the plot.
        detailed: If True, plot detailed per-kernel curves.
        top_k: Number of top kernels to plot in detailed mode.
        kernel_update_list: List of kernel updates for each group.
    """
    num_groups = len(variances_list)
    
    # Create a 2 x num_groups grid of subplots
    fig, axes = plt.subplots(2, num_groups, figsize=(15 * num_groups, 12), squeeze=False)
    
    # Plot per-group subplots
    for group_idx in range(num_groups):
        variances_np = variances_list[group_idx].cpu().numpy()
        means_rotated_np = means_rotated_list[group_idx].cpu().numpy()
        milestones_indices = milestones_indices_list[group_idx]

        # Setup axes
        ax_var = axes[0, group_idx]
        ax_sim = ax_var.twinx()
        ax_mean = axes[1, group_idx]

        # Convert cosine similarity to numpy
        if cosine_similarity_windowed_list[group_idx] is not None:
            cosine_sim_windowed_np = cosine_similarity_windowed_list[group_idx].cpu().numpy()
            cosine_sim_mean = np.mean(cosine_sim_windowed_np, axis=0)
            cosine_sim_std = np.std(cosine_sim_windowed_np, axis=0)
        else:
            cosine_sim_mean = np.zeros(len(variances_np))
            cosine_sim_std = np.zeros(len(variances_np))

        x = np.arange(len(variances_np))

        # Top row: variances
        ax_var.bar(x, variances_np, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)

        # Cosine similarities
        if np.any(cosine_sim_mean != 0):
            upper = cosine_sim_mean + cosine_sim_std
            lower = cosine_sim_mean - cosine_sim_std
            ax_sim.fill_between(x, lower, upper, color='red', alpha=0.15)
            ax_sim.plot(x, cosine_sim_mean, 'o-', color='red', linewidth=2, markersize=4, alpha=0.9)

            # Optional detailed per-kernel curves
            if detailed and kernel_update_list and kernel_update_list[group_idx] is not None:
                cosines_np_mag = cosine_sim_windowed_np
                ku = kernel_update_list[group_idx]
                if hasattr(ku, 'cpu'):
                    ku = ku.cpu().numpy()
                
                if cosines_np_mag.shape[0] > 0:
                    # Select top-k kernels by L1 norm
                    if top_k and isinstance(top_k, int) and 0 < top_k < cosines_np_mag.shape[0]:
                        l1_norms = np.sum(np.abs(ku), axis=1)
                        top_indices = np.argsort(-l1_norms)[:top_k]
                    else:
                        top_indices = np.arange(cosines_np_mag.shape[0])
                    
                    cmap = plt.get_cmap('tab20')
                    for j, idx in enumerate(top_indices):
                        color = cmap(j % 20)
                        ax_sim.plot(x, cosines_np_mag[idx], color=color, linewidth=1.0, alpha=0.6)

        # Milestone vertical lines
        for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
            color = MILESTONE_COLORS[i]
            ax_var.axvline(x=milestone_idx, color=color, linestyle='--', linewidth=2, alpha=0.7)

        # Axis labels for top row
        if group_idx == 0:
            ax_var.set_ylabel('Explained Variance', color='blue')
            ax_var.tick_params(axis='y', labelcolor='blue')
            if np.any(cosine_sim_mean != 0):
                ax_sim.set_ylabel('Windowed Cosine Similarity', color='red')
                ax_sim.tick_params(axis='y', labelcolor='red')
        ax_var.set_xlabel('PCA Component Index')

        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.7,
                         label='Explained Variance'),
        ]
        num_components = len(variances_np)
        for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
            idx_ratio_pct = (milestone_idx / max(1, num_components)) * 100
            legend_elements.append(
                plt.Line2D([0], [0], color=MILESTONE_COLORS[i], linestyle='--',
                           label=f'{int(milestone*100)}% @ idx {milestone_idx} ({idx_ratio_pct:.0f}%)')
            )
        if np.any(cosine_sim_mean != 0):
            legend_elements.extend([
                plt.Line2D([0], [0], color='red', marker='o', linestyle='-',
                          label='Mean Cosine Similarity'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.15,
                             label='Mean ± 1 Std. Dev.'),
            ])
        ax_var.legend(handles=legend_elements, loc='upper right')

        # Bottom row: means in rotated space
        ax_mean.bar(x, -means_rotated_np, alpha=0.7, color='lightcoral',
                   edgecolor='darkred', linewidth=0.5)
        for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
            ax_mean.axvline(x=milestone_idx, color=MILESTONE_COLORS[i],
                           linestyle='--', linewidth=2, alpha=0.7)
        ax_mean.set_xlabel('PCA Component Index')
        if group_idx == 0:
            ax_mean.set_ylabel('Means in Rotated Space (Negative)')
    
    # Aesthetics
    for r in range(2):
        for c in range(num_groups):
            axes[r, c].grid(True, alpha=0.3)
    for c in range(num_groups):
        axes[1, c].invert_yaxis()
    
    plt.suptitle(f'PCA Analysis Summary\nModule: {module_name} ({num_groups} groups)',
                fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / f"{module_name}_histogram.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_polar(cosine_similarity_cumsum_list, kernel_update_list, module_name,
               milestones, milestones_indices_list, save_dir):
    """Plot kernel updates in a polar coordinates system.
    
    Supports grouped convolutions. Creates separate polar plots for each group
    (max 8 groups). Each group has subplots for each milestone.
    
    Args:
        cosine_similarity_cumsum_list: List of cumulative cosine similarities
            for each group. Each element shape: (num_kernels, num_components).
        kernel_update_list: List of kernel updates for each group.
        module_name: Name of the module.
        milestones: Milestones of the accumulated variance ratio.
        milestones_indices_list: List of milestone indices for each group.
        save_dir: Directory to save the plot.
    """
    num_groups = min(len(kernel_update_list), MAX_POLAR_GROUPS)
    num_milestones = len(milestones)
    
    if len(kernel_update_list) > MAX_POLAR_GROUPS:
        print(f"Warning: Module {module_name} has {len(kernel_update_list)} groups, "
              f"limiting visualization to first {MAX_POLAR_GROUPS} groups")
    
    # Create figure with groups arranged vertically
    fig, axes = plt.subplots(num_groups, num_milestones,
                            figsize=(6 * num_milestones, 6 * num_groups),
                            subplot_kw=dict(projection='polar'))
    
    # Handle single group/milestone case
    if num_groups == 1:
        axes = axes.reshape(1, -1)
    if num_milestones == 1:
        axes = axes.reshape(-1, 1)
    
    # Create colormap
    colormap = plt.get_cmap('RdYlBu_r')
    
    for group_idx in range(num_groups):
        if (cosine_similarity_cumsum_list[group_idx] is None or
                kernel_update_list[group_idx] is None):
            # Skip empty groups
            for milestone_idx in range(num_milestones):
                ax = axes[group_idx, milestone_idx]
                ax.set_title(f'Group {group_idx}\nNo Data', pad=20)
                ax.set_ylim(0, 1.2)
                ax.grid(True, alpha=0.3)
            continue
            
        cosine_similarity_cumsum = cosine_similarity_cumsum_list[group_idx]
        kernel_update = kernel_update_list[group_idx]
        milestones_indices = milestones_indices_list[group_idx]
        
        # Get kernel norms and normalize
        kernel_norms = torch.norm(kernel_update, p=2, dim=1).cpu().numpy()
        if len(kernel_norms) == 0:
            continue
        max_norm = np.max(kernel_norms)
        normalized_norms = kernel_norms / max_norm
        
        # Convert to numpy
        cosine_sim_cumsum_np = cosine_similarity_cumsum.cpu().numpy()
        
        for milestone_idx, (milestone, milestone_idx_val) in enumerate(
                zip(milestones, milestones_indices)):
            ax = axes[group_idx, milestone_idx]
            
            # Calculate cumulative cosine similarity up to milestone
            x_cosine_similarities = cosine_sim_cumsum_np[:, milestone_idx_val]
            
            # Calculate remaining cosine similarity
            if milestone_idx_val < cosine_sim_cumsum_np.shape[1] - 1:
                y_cosine_similarities = (cosine_sim_cumsum_np[:, -1] -
                                        cosine_sim_cumsum_np[:, milestone_idx_val])
            else:
                y_cosine_similarities = np.zeros_like(x_cosine_similarities)
            
            # Calculate angles
            angles = np.arctan2(y_cosine_similarities, x_cosine_similarities)
            angles = np.where(angles < 0, angles + 2 * np.pi, angles)
            
            # Normalize cosine similarities for colormap
            x_cosine_normalized = np.clip((x_cosine_similarities + 1) / 2, 0, 1)
            
            # Plot vectors
            for j in range(len(cosine_sim_cumsum_np)):
                angle = angles[j]
                radius = normalized_norms[j]
                color = colormap(x_cosine_normalized[j])
                ax.plot([0, angle], [0, radius], color=color, linewidth=2, alpha=0.8)
                ax.scatter(angle, radius, color=color, s=20, alpha=0.8)
            
            # Set polar plot properties
            ax.set_ylim(0, 1.2)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Group {group_idx}\n{milestone*100:.0f}% Milestone\n'
                        f'Split at Component {milestone_idx_val}', pad=20)
    
    plt.suptitle(f'Kernel Update Projections in Polar Coordinates\n'
                f'Module: {module_name} ({len(kernel_update_list)} groups)', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / f"{module_name}_polar_projection.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _filter_modules(module_names, module_filter):
    """Filter module names based on various filter criteria.
    
    Args:
        module_names: List of all module names (sorted by call_idx)
        module_filter: Filter specification. Can be:
            - String "model start:end": match modules with prefix "model.i" where i in [start, end)
              (e.g., "model 0:5" matches "model.0.*", "model.1.*", ..., "model.4.*")
            - String "start:end": index range (e.g., "0:10" for first 10 layers)
            - String "start:": from start index to end (e.g., "10:" for layers after index 10)
            - String ":end": from beginning to end index (e.g., ":20" for first 20 layers)
            - List of strings: module names to include (supports wildcards with '*')
            - String: single module name or pattern (supports wildcards with '*')
    
    Returns:
        List of filtered module names
    """
    import fnmatch
    import re
    
    if isinstance(module_filter, str):
        # Check if it's "model start:end" format (e.g., "model 0:5")
        model_pattern = r'^model\s+(\d+):(\d+)$'
        match = re.match(model_pattern, module_filter.strip())
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            filtered = []
            for i in range(start, end):
                pattern = f"model.{i}.*"
                filtered.extend([name for name in module_names if fnmatch.fnmatch(name, pattern)])
            # Remove duplicates while preserving order
            seen = set()
            return [name for name in filtered if not (name in seen or seen.add(name))]
        
        # Check if it's an index range format (e.g., "0:10", "10:", ":20")
        if ':' in module_filter:
            parts = module_filter.split(':')
            if len(parts) == 2:
                start_str, end_str = parts
                try:
                    start = int(start_str) if start_str else 0
                    end = int(end_str) if end_str else len(module_names)
                    return module_names[start:end]
                except ValueError:
                    pass  # Not a numeric range, continue to pattern matching
        
        # Otherwise treat as a pattern (supports wildcards)
        return [name for name in module_names if fnmatch.fnmatch(name, module_filter)]
    
    elif isinstance(module_filter, list):
        # List of patterns/names - recursively filter for each pattern
        filtered = []
        for pattern in module_filter:
            # Recursively call _filter_modules to handle all pattern types
            filtered.extend(_filter_modules(module_names, pattern))
        # Remove duplicates while preserving order
        seen = set()
        return [name for name in filtered if not (name in seen or seen.add(name))]
    
    else:
        return module_names


def plot_value_shift_bars(module_means, save_dir, xlim_main=None, xlim_kernel_update=None, 
                          module_filter=None):
    """Plot horizontal bar chart showing value shifts for each module.
    
    Args:
        module_means: Dictionary mapping module names to their mean values
        save_dir: Directory to save the plot
        xlim_main: Tuple (min, max) for main x-axis range, or None for auto
        xlim_kernel_update: Tuple (min, max) for kernel update x-axis range, or None for auto
        module_filter: Filter for modules to visualize. Can be:
            - None: visualize all modules
            - String "model start:end": match modules with prefix "model.i" where i in [start, end)
              (e.g., "model 0:5" matches "model.0.*", "model.1.*", ..., "model.4.*")
            - String "start:end": index range (e.g., "0:10" for first 10 layers)
            - String "start:": from start index to end (e.g., "10:" for layers after index 10)
            - String ":end": from beginning to end index (e.g., ":20" for first 20 layers)
            - List of strings: module names to include (supports wildcards with '*' and all above formats)
            - String: single module name or pattern (supports wildcards with '*')
    """
    if not module_means:
        print("No module statistics to plot")
        return
    
    # Sort modules by name for consistent ordering
    all_module_names = sorted(module_means.keys(), key=lambda x:module_means[x]["call_idx"])
    
    # Apply module filter
    if module_filter is None:
        module_names = all_module_names
    else:
        module_names = _filter_modules(all_module_names, module_filter)
        if not module_names:
            print(f"Warning: No modules match the filter '{module_filter}'. Using all modules.")
            module_names = all_module_names
        else:
            print(f"Filtered to {len(module_names)}/{len(all_module_names)} modules based on filter: {module_filter}")
    
    # Extract values for each metric
    kernel_update_mult_input_vals = [
        module_means[name]['kernel_update_mult_input'] for name in module_names
    ]
    base_kernel_mult_input_shift_vals = [
        module_means[name]['base_kernel_mult_input_shift'] for name in module_names
    ]
    kernel_update_mult_input_shift_vals = [
        module_means[name]['kernel_update_mult_input_shift'] for name in module_names
    ]
    kernel_update_abs_mean_vals = [
        module_means[name]['kernel_update_abs_mean'] for name in module_names
    ]
    input_update_abs_mean_vals = [
        module_means[name]['input_update_abs_mean'] for name in module_names
    ]
    
    # Create figure with primary axis
    fig, ax1 = plt.subplots(figsize=(12, max(8, len(module_names) * 0.3)))
    
    # Set up y-axis positions
    y_pos = np.arange(len(module_names))
    bar_height = 0.16
    
    # Plot horizontal bars on primary axis (excluding kernel_update_abs_mean)
    ax1.barh(y_pos - 2 * bar_height, kernel_update_mult_input_vals, bar_height,
                    label='Kernel Update × Base Input', color='red', alpha=0.8)
    ax1.barh(y_pos - bar_height, base_kernel_mult_input_shift_vals, bar_height,
                    label='Base Kernel × Input Shift', color='blue', alpha=0.8)
    ax1.barh(y_pos, kernel_update_mult_input_shift_vals, bar_height,
                    label='Kernel Update × Input Shift', color='brown', alpha=0.8)
    ax1.barh(y_pos + 2 * bar_height, input_update_abs_mean_vals, bar_height,
                    label='Input Update Abs Mean', color='purple', alpha=0.8)
    
    # Set fixed x-axis range for main axis if provided
    if xlim_main is not None:
        ax1.set_xlim(xlim_main)
    
    # Customize primary axis
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(module_names, fontsize=1)
    ax1.set_xlabel('Mean Absolute Value', fontsize=12, color='black')
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.tick_params(axis='y', labelsize=8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Create secondary axis for kernel_update_abs_mean
    ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
    
    # Set same y-axis range and ticks for secondary axis
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y_pos)
    
    # Ensure y-axis labels are visible on the left side
    ax1.yaxis.set_ticks_position('left')
    ax1.yaxis.set_label_position('left')
    
    ax2.barh(y_pos + bar_height, kernel_update_abs_mean_vals, bar_height,
                    label='Kernel Update Abs Mean', color='green', alpha=0.8)
    
    # Set fixed x-axis range for kernel update axis if provided
    if xlim_kernel_update is not None:
        ax2.set_xlim(xlim_kernel_update)
    
    # Customize secondary axis
    ax2.set_xlabel('Kernel Update Abs Mean', fontsize=12, color='green')
    ax2.tick_params(axis='x', labelcolor='green')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    ax1.set_title('Value Shift Analysis Across Modules', fontsize=14, fontweight='bold')
    
    # Adjust layout to ensure y-axis labels are visible on the left
    # Use tight_layout with left margin to preserve module name labels
    plt.tight_layout(rect=[0.2, 0, 0.98, 0.98])  # [left, bottom, right, top]
    
    # Save figure with padding to ensure labels are not clipped
    save_path = Path(save_dir) / "value_shift_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print(f"Saved value shift analysis to {save_path}")


def _load_sample_images(sample_dirs, sample_num):
    """Load sample images from directories.
    
    Args:
        sample_dirs: Single directory path or list of directory paths
        sample_num: Maximum number of samples to load
    
    Returns:
        List of image file paths
    """
    sample_files = []
    dirs = sample_dirs if isinstance(sample_dirs, (list, tuple)) else [sample_dirs]
    
    for directory in dirs:
        for ext in IMAGE_EXTENSIONS:
            sample_files.extend(glob.glob(os.path.join(directory, f'*.{ext.lower()}')))
            sample_files.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    
    random.shuffle(sample_files)
    return sample_files[:sample_num]


def _load_label_files(sample_files, label_dirs):
    """Load label files corresponding to sample images.
    
    Args:
        sample_files: List of sample image file paths
        label_dirs: Single directory path or list of directory paths
    
    Returns:
        List of label file paths (or None if not found)
    """
    if label_dirs is None:
        return [None] * len(sample_files)
    
    label_files = []
    dirs = label_dirs if isinstance(label_dirs, (list, tuple)) else [label_dirs]
    
    for sample_file in sample_files:
        label_name = os.path.basename(sample_file).split('.')[0] + '.txt'
        found = False
        
        for label_dir in dirs:
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                label_files.append(label_path)
                found = True
                break
        
        if not found:
            label_files.append(None)
            warnings.warn(f"Label file {label_name} not found in {label_dirs}")
    
    return label_files


def _parse_bboxes_from_label(label_file):
    """Parse bounding boxes from label file.
    
    Args:
        label_file: Path to label file (YOLO format)
    
    Returns:
        List of bounding boxes [x_min, y_min, x_max, y_max]
    """
    if label_file is None or not os.path.exists(label_file):
        return []
    
    bboxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x_min, y_min = x - w / 2, y - h / 2
                x_max, y_max = x + w / 2, y + h / 2
                bboxes.append([x_min, y_min, x_max, y_max])
    
    return bboxes


def _preprocess_image(image_path, device):
    """Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        device: Device for tensor
    
    Returns:
        Preprocessed image tensor [1, 3, 640, 640]
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
    image = image.transpose(2, 0, 1) / 255.0
    image = torch.from_numpy(image).float().to(device)
    return image.unsqueeze(0)


def _process_module_for_value_shift(base_name, base_module, incremental_name,
                                   incremental_module, intermediate_results,
                                   base_input, incremental_input, base_output,
                                   incremental_output, bboxes, save_dir):
    """Process a single module for value shift analysis.
    
    Args:
        base_name: Base module name
        base_module: Base module
        incremental_name: Incremental module name
        incremental_module: Incremental module
        intermediate_results: Dictionary containing intermediate results
        base_input: Base model input (extracted from intermediate_results)
        incremental_input: Incremental model input (extracted from intermediate_results)
        base_output: Base model output (extracted from intermediate_results)
        incremental_output: Incremental model output (extracted from intermediate_results)
        bboxes: Bounding boxes
        save_dir: Save directory
    
    Returns:
        Tuple of four mean absolute values: (kernel_update_mult_input, 
        base_kernel_mult_input_shift, kernel_update_mult_input_shift, 
        kernel_update_abs_mean, input_update_abs_mean) or None if error
    """
    if base_name != incremental_name:
        warnings.warn(f"Module names do not match: {base_name} vs {incremental_name}")
        return None
    
    if base_name not in intermediate_results["base_model"]:
        return None
    
    if base_name not in intermediate_results["incremental_model"]:
        warnings.warn(f"Module {base_name} not found in incremental model hooks")
        return None
    
    base_input_raw = intermediate_results["base_model"][base_name]["input"]
    incremental_input_raw = intermediate_results["incremental_model"][base_name]["input"]
    
    base_input = base_input_raw[0] if isinstance(base_input_raw, tuple) else base_input_raw
    incremental_input = (incremental_input_raw[0] if isinstance(incremental_input_raw, tuple)
                        else incremental_input_raw)
    
    base_output = intermediate_results["base_model"][base_name]["output"]
    incremental_output = intermediate_results["incremental_model"][base_name]["output"]
    
    # Check shapes match
    if base_input.shape != incremental_input.shape:
        warnings.warn(f"Module {base_name}: Input shapes do not match "
                        f"{base_input.shape} vs {incremental_input.shape}")
        return None
    
    if base_output.shape != incremental_output.shape:
        warnings.warn(f"Module {base_name}: Output shapes do not match "
                        f"{base_output.shape} vs {incremental_output.shape}")
        return None
    
    # Flatten kernels
    base_kernel = flatten_kernel(base_module)
    incremental_kernel = flatten_kernel(incremental_module)
    
    # Handle grouped convolutions (take first group for now)
    if isinstance(base_kernel, list):
        base_kernel = base_kernel[0]
    if isinstance(incremental_kernel, list):
        incremental_kernel = incremental_kernel[0]
    
    # Calculate value shift
    kernel_update_mult_input, base_kernel_mult_input_shift, kernel_update_mult_input_shift = \
        cal_value_shift(base_input, incremental_input, base_output, incremental_output,
                        base_kernel, incremental_kernel, base_name, save_dir,
                        module=base_module, box=bboxes)
    
    # Calculate kernel update absolute mean
    kernel_update = incremental_kernel - base_kernel
    kernel_update_abs_mean = torch.mean(torch.abs(kernel_update)).item()
    # kernel_update_abs_mean = 0.

    # Calculate input update absolute mean
    input_update = incremental_input - base_input
    input_update_abs_mean = torch.mean(torch.abs(input_update)).item()
    # input_update_abs_mean = 0.
    
    # Return mean absolute values
    return (
        torch.mean(torch.abs(kernel_update_mult_input)).item(),
        torch.mean(torch.abs(base_kernel_mult_input_shift)).item(),
        torch.mean(torch.abs(kernel_update_mult_input_shift)).item(),
        kernel_update_abs_mean,
        input_update_abs_mean
    )


def _run_value_shift_analysis(base_model, incremental_model, args, device):
    """Run value shift analysis (Stage 3).
    
    Args:
        base_model: Base model
        incremental_model: Incremental model
        args: Command line arguments
        device: Device for computation
    """
    print("\nStage 3: Starting value shift analysis...")
    
    # Register hooks
    intermediate_results = {"base_model": {}, "incremental_model": {}}
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d):
            intermediate_results["base_model"][name] = {}
            module.register_forward_hook(create_hook(intermediate_results["base_model"], name))
    
    for name, module in incremental_model.named_modules():
        if isinstance(module, nn.Conv2d):
            intermediate_results["incremental_model"][name] = {}
            module.register_forward_hook(create_hook(intermediate_results["incremental_model"], name))
    
    # Load sample images and labels
    sample_files = _load_sample_images(args.sample_dir, args.sample_num)
    label_files = _load_label_files(sample_files, args.label_dir) if args.label_dir else [None] * len(sample_files)
    
    # Initialize accumulator for each module
    module_stats = {
        name: {
            'kernel_update_mult_input': [],
            'base_kernel_mult_input_shift': [],
            'kernel_update_mult_input_shift': [],
            'kernel_update_abs_mean': [],
            'input_update_abs_mean': []
        }
        for name in intermediate_results["base_model"].keys()
    }
    
    # Process each sample
    sample_pbar = tqdm(range(len(sample_files)), desc="Processing samples for value shift analysis")
    for i in sample_pbar:
        # Preprocess image
        image = _preprocess_image(sample_files[i], device)
        
        # Parse bounding boxes
        bboxes = _parse_bboxes_from_label(label_files[i]) if args.label_dir else []
        
        # Forward pass
        with torch.no_grad():
            base_model(image)
            incremental_model(image)
            intermediate_results["base_model"]["call_cnt"] = -1
            intermediate_results["incremental_model"]["call_cnt"] = -1
        
        # Process each module
        for (base_name, base_module), (inc_name, inc_module) in zip(
                base_model.named_modules(), incremental_model.named_modules()):

            if base_name not in intermediate_results["base_model"].keys():
                continue

            result = _process_module_for_value_shift(
                base_name, base_module, inc_name, inc_module, intermediate_results,
                None, None, None, None, bboxes, args.save_dir
            )
            
            if result is not None:
                kernel_update_mult_input, base_kernel_mult_input_shift, kernel_update_mult_input_shift,\
                     kernel_update_abs_mean, input_update_abs_mean = result
                module_stats[base_name]['kernel_update_mult_input'].append(kernel_update_mult_input)
                module_stats[base_name]['base_kernel_mult_input_shift'].append(base_kernel_mult_input_shift)
                module_stats[base_name]['kernel_update_mult_input_shift'].append(kernel_update_mult_input_shift)
                module_stats[base_name]['kernel_update_abs_mean'].append(kernel_update_abs_mean)
                module_stats[base_name]['input_update_abs_mean'].append(input_update_abs_mean)
            
            module_stats[base_name]["call_idx"] = intermediate_results['base_model'][base_name]["call_idx"]
    
    # Calculate mean across all samples for each module
    module_means = {}
    for name in module_stats:
        num_samples = len(module_stats[name]['kernel_update_mult_input'])
        if num_samples > 0:
            module_means[name] = {
                'kernel_update_mult_input': np.mean(module_stats[name]['kernel_update_mult_input']),
                'base_kernel_mult_input_shift': np.mean(module_stats[name]['base_kernel_mult_input_shift']),
                'kernel_update_mult_input_shift': np.mean(module_stats[name]['kernel_update_mult_input_shift']),
                'kernel_update_abs_mean': np.mean(module_stats[name]['kernel_update_abs_mean']),
                'input_update_abs_mean': np.mean(module_stats[name]['input_update_abs_mean']),
                'call_idx': module_stats[name]['call_idx']
            }
            print(f"Module {name}: processed {num_samples}/{len(sample_files)} samples")
        else:
            warnings.warn(f"Module {name}: No valid samples processed")
    
    if not module_means:
        print("Warning: No modules have valid statistics. Cannot generate value shift analysis plot.")
    else:
        print(f"\nGenerating value shift analysis plot for {len(module_means)} modules...")
        
        # Parse xlim_main if provided
        xlim_main = None
        if args.xlim_main is not None:
            try:
                min_val, max_val = map(float, args.xlim_main.split(','))
                xlim_main = (min_val, max_val)
            except ValueError:
                warnings.warn(f"Invalid xlim_main format: {args.xlim_main}. Expected 'min,max'. Using auto scale.")
        
        # Parse xlim_kernel_update if provided
        xlim_kernel_update = None
        if args.xlim_kernel_update is not None:
            try:
                min_val, max_val = map(float, args.xlim_kernel_update.split(','))
                xlim_kernel_update = (min_val, max_val)
            except ValueError:
                warnings.warn(f"Invalid xlim_kernel_update format: {args.xlim_kernel_update}. Expected 'min,max'. Using auto scale.")
        
        # Parse module_filter if provided
        module_filter = None
        if args.module_filter is not None:
            # Check if it's a comma-separated list of patterns
            if ',' in args.module_filter:
                module_filter = [pattern.strip() for pattern in args.module_filter.split(',')]
            else:
                module_filter = args.module_filter
        
        plot_value_shift_bars(module_means, args.save_dir, xlim_main=xlim_main, 
                              xlim_kernel_update=xlim_kernel_update, module_filter=module_filter)


def _process_modules_for_pca_visualization(modules_to_process, pca_cache, args,
                                          enabled_stages, has_both_models, device):
    """Process modules for PCA visualization (Stages 1 and 2).
    
    Args:
        modules_to_process: Iterable of modules to process
        pca_cache: PCA cache dictionary
        args: Command line arguments
        enabled_stages: Set of enabled stage numbers
        has_both_models: Whether both models are available
        device: Device for computation
    """
    pbar = tqdm(modules_to_process, desc="Processing modules", total=len(pca_cache))
    
    for item in pbar:
        # Extract module information
        if has_both_models:
            (module_name, base_module), (inc_module_name, inc_module) = item
            assert module_name == inc_module_name, "Module names do not match"
            pca_operator = pca_cache[module_name]
            base_kernel = flatten_kernel(base_module, args.unfold)
            incremental_kernel = flatten_kernel(inc_module, args.unfold)
            
            # Handle grouped and single kernels
            if isinstance(base_kernel, list) and isinstance(incremental_kernel, list):
                if len(base_kernel) != len(incremental_kernel):
                    pbar.set_postfix_str("Group count mismatch, skipping")
                    continue
                kernel_update = [i_k - b_k for b_k, i_k in zip(base_kernel, incremental_kernel)
                               if b_k.shape == i_k.shape]
                if len(kernel_update) != len(base_kernel):
                    pbar.set_postfix_str("Shape mismatch, skipping")
                    continue
            else:
                if base_kernel.shape != incremental_kernel.shape:
                    pbar.set_postfix_str("Shape mismatch, skipping")
                    continue
                kernel_update = incremental_kernel - base_kernel
        else:
            # Base-only or PCA-only mode
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], nn.Conv2d):
                module_name, base_module = item
                pca_operator = pca_cache[module_name]
                kernel_update = flatten_kernel(base_module, args.unfold)
            else:
                module_name, pca_operator = item
                kernel_update = None
        
        pbar.set_description(f"Processing module: {module_name}")
        
        # Validate PCA operator
        if isinstance(pca_operator, list):
            num_groups = len(pca_operator)
            if (num_groups == 0 or not hasattr(pca_operator[0], 'explained_variance_') or
                    len(pca_operator[0].explained_variance_) == 0):
                pbar.set_postfix_str("No PCA data, skipping")
                continue
        else:
            num_groups = 1
            if (not hasattr(pca_operator, 'explained_variance_') or
                    len(pca_operator.explained_variance_) == 0):
                pbar.set_postfix_str("No PCA data, skipping")
                continue
            pca_operator = [pca_operator]
        
        # Calculate PCA statistics
        (variances_list, means_rotated_list, milestones_indices_list,
         cosine_similarity_windowed_list, cosine_similarity_cumsum_list,
         kernel_update_list) = _calculate_pca_statistics(
            pca_operator, kernel_update, device, MILESTONES
        )
        
        # Stage 1: Plot histogram
        generated_plots = []
        if 1 in enabled_stages:
            plot_histogram(cosine_similarity_windowed_list, variances_list,
                          means_rotated_list, module_name, MILESTONES,
                          milestones_indices_list, args.save_dir,
                          detailed=args.detailed, top_k=args.k,
                          kernel_update_list=kernel_update_list)
            generated_plots.append("histogram")
        
        # Stage 2: Plot polar projection
        if (2 in enabled_stages and has_both_models and
                any(x is not None for x in cosine_similarity_cumsum_list)):
            plot_polar(cosine_similarity_cumsum_list, kernel_update_list, module_name,
                      MILESTONES, milestones_indices_list, args.save_dir)
            generated_plots.append("polar")
        
        if generated_plots:
            pbar.set_postfix_str(f"Generated {', '.join(generated_plots)} for {module_name} "
                               f"({num_groups} groups)")
        else:
            pbar.set_postfix_str(f"Skipped {module_name} (no stages enabled)")
    

def main(args):
    """Main function.
    
    Args:
        args: Command line arguments
    """
    # Print enabled stages
    enabled_stages = set(args.stages)
    print(f"Enabled stages: {sorted(enabled_stages)}")
    print("  Stage 1: Variance and weight update projection histograms")
    print("  Stage 2: Weight projection polar plots")
    print("  Stage 3: Value shift histograms")
    
    print(f"Saving results to: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load PCA cache
    print(f"Loading PCA cache from: {args.pca_cache_path}")
    pca_cache = joblib.load(args.pca_cache_path)
    print(f"Successfully loaded PCA cache with {len(pca_cache)} modules")
    
    if len(pca_cache) == 0:
        print("Error: PCA cache is empty. No plots will be generated.")
        return
    
    # Load models
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is set to device but unavailable, using CPU instead.")
    
    base_model = None
    if args.base_model:
        print(f"Loading base model from: {args.base_model}")
        base_model = YOLO(args.base_model).model
        print("Successfully loaded base model")
    
    incremental_model = None
    if args.incremental_model:
        print(f"Loading incremental model from: {args.incremental_model}")
        incremental_model = YOLO(args.incremental_model).model
        print("Successfully loaded incremental model")
    
    has_both_models = base_model is not None and incremental_model is not None
    
    # Prepare modules for processing
    need_main_loop = (1 in enabled_stages) or (2 in enabled_stages)
    
    if need_main_loop:
        if has_both_models:
            print("Both models provided. Performing full kernel update analysis.")
            base_model.to(device).eval()
            incremental_model.to(device).eval()
            base_modules = [(name, module) for name, module in base_model.named_modules()
                          if isinstance(module, nn.Conv2d) and name in pca_cache]
            inc_modules = [(name, module) for name, module in incremental_model.named_modules()
                         if isinstance(module, nn.Conv2d) and name in pca_cache]
            modules_to_process = zip(base_modules, inc_modules)
        elif base_model:
            print("Incremental model not provided. Visualizing base model weights.")
            base_model.to(device).eval()
            modules_to_process = [(name, module) for name, module in base_model.named_modules()
                                if isinstance(module, nn.Conv2d) and name in pca_cache]
        else:
            print("Warning: No models provided. Only PCA variance and mean plots will be generated.")
            modules_to_process = pca_cache.items()
        
        print(f"Found {len(pca_cache)} modules in PCA cache: {list(pca_cache.keys())}")
        
        # Process modules for PCA visualization
        _process_modules_for_pca_visualization(
            modules_to_process, pca_cache, args, enabled_stages, has_both_models, device
        )
    else:
        print("Stages 1 and 2 are disabled. Skipping main processing loop.")
    
    # Stage 3: Value shift analysis
    if 3 in enabled_stages:
        if not has_both_models:
            print("\nStage 3: Skipped (requires both base_model and incremental_model)")
        elif args.sample_dir is None:
            print("\nStage 3: Skipped (requires --sample_dir)")
        else:
            # Ensure models are moved to device before value shift analysis
            if base_model is not None:
                base_model.to(device).eval()
            if incremental_model is not None:
                incremental_model.to(device).eval()
            _run_value_shift_analysis(base_model, incremental_model, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_cache_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--incremental_model", type=str, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unfold", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--detailed", action="store_true", default=False)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--sample_dir", type=str, nargs='+', default=None,
                       help="Directory or directories containing sample images")
    parser.add_argument("--label_dir", type=str, nargs='+', default=None,
                        help="Directory or directories containing label files (optional)")
    parser.add_argument("--sample_num", type=int, default=100,
                        help="Number of samples to use for value shift analysis")
    parser.add_argument("--stages", type=int, nargs='+', default=[1, 2, 3],
                       help="Stages to enable: 1=histogram plots, 2=polar plots, "
                           "3=value shift analysis (default: all)")
    parser.add_argument("--xlim_main", type=str, default=None,
                       help="Main x-axis range as 'min,max' (e.g., '0,100')")
    parser.add_argument("--xlim_kernel_update", type=str, default=None,
                       help="Kernel update x-axis range as 'min,max' (e.g., '0,10')")
    parser.add_argument("--module_filter", type=str, default=None,
                       help="Filter modules to visualize. Options:\n"
                            "  - Model range: 'model start:end' (e.g., 'model 0:5' matches model.0.* to model.4.*)\n"
                            "  - Index range: 'start:end' (e.g., '0:10' for first 10 layers)\n"
                            "  - From index: 'start:' (e.g., '10:' for layers after index 10)\n"
                            "  - To index: ':end' (e.g., ':20' for first 20 layers)\n"
                            "  - Pattern: module name pattern with wildcards (e.g., 'model.*.conv')\n"
                            "  - Multiple patterns: comma-separated (e.g., 'model.0.*,model.1.*')")
    args = parser.parse_args()
    main(args)
