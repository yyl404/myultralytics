"""
Visualize the projection of kernel updates on PCA components.

Usage:
python proj_vis.py --pca_cache_path <pca_cache_path> --base_model <base_model> --incremental_model <incremental_model> --save_dir <save_dir>

Example:
python proj_vis.py --pca_cache_path runs/yolov8l_voc_inc_15_1x5_fromscratch/task-1/pca_cache.joblib \
    --base_model runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best.pt \
    --incremental_model runs/yolov8l_voc_inc_15_5_fromscratch/task-2/best.pt \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/proj_vis_model_2-1

Arguments:
    --pca_cache_path: The path to the PCA cache file.
    --base_model: The path to the base model file.
    --incremental_model: The path to the incremental model file.
    --save_dir: The directory to save the results.
"""

import joblib
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import Conv


def flatten_augment_kernel(module):
    assert isinstance(module, Conv), "Module must be a convolutional module"
    if module.conv.bias is not None:
        # If bias exists, augment the weight matrix to represent convolution
        # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
        weight = module.conv.weight.reshape(module.conv.weight.shape[0], -1)
        bias = module.conv.bias.reshape(module.conv.bias.shape[0], -1)
        kernel = torch.cat([weight, bias], dim=1)
    else:
        kernel = module.conv.weight.reshape(module.conv.weight.shape[0], -1)
    return kernel


def window_cumsum(x, window, dim):
    """
    Calculate the cumulative sum of a tensor with a window.
    Args:
        x(Tensor): The input tensor.
        window(int): The window size.
        dim(int): The dimension to apply the window cumsum.
    Returns:
        Tensor: The window cumulative sum of the input tensor.
    """
    # Get the shape of the input tensor
    shape = x.shape
    num_dims = len(shape)
    
    # Move the target dimension to the last position for easier processing
    if dim != num_dims - 1:
        # Create permutation to move dim to last position
        perm = list(range(num_dims))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_permuted = x.permute(perm)
    else:
        x_permuted = x
    
    # Reshape to 2D for batch processing: (batch_size, sequence_length)
    original_shape = x_permuted.shape
    x_2d = x_permuted.reshape(-1, original_shape[-1])
    
    # Apply window cumulative sum using convolution
    # Create a 1D convolution kernel of ones with size window
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    
    # Apply convolution with appropriate padding
    # We need to pad to maintain the same length
    padding = window - 1
    x_padded = F.pad(x_2d.unsqueeze(1), (0, padding), mode='constant', value=0)
    
    # Apply convolution
    result = F.conv1d(x_padded, kernel, stride=1, padding=0)
    result = result.squeeze(1)  # Remove the channel dimension
    
    # Reshape back to original shape
    result = result.reshape(original_shape)
    
    # Move the dimension back to its original position
    if dim != num_dims - 1:
        result = result.permute(perm)
    
    return result

def plot_histgram(cosine_similarity_windowed, window_size, variances, means_rotated, module_name, milestones, milestones_indices, save_dir):
    """
    Plot histogram of PCA variances, means and windowed cumulative cosine similarities between kernel updates
    and PCA components subspaces.
    
    Args:
        cosine_similarity_windowed(Tensor): The windowed cumulative cosine similarity between
            kernel updates and PCA components subspaces. Shape: (num_kernels, num_components).
        window_size(int): The window size used to calculate the windowed cumulative cosine similarities.
        variances(Tensor): The explained variance of the PCA components. Shape: (num_components).
        means_rotated(Tensor): The rotated means in PCA coordinate space. Shape: (num_components).
        module_name(str): The name of the module.
        milestones(list): The milestones of the accumulated variance ratio.
        milestones_indices(list): The indices of the PCA components according to the milestones.
        save_dir(str): The directory to save the plot.
    
    Plot two histograms.
    The first histogram is composed of variances and windowed cumulative cosine similarities. The x-axis is the PCA component index.
    The y-axis is the value of the variance or windowed cumulative cosine similarities. The variances are represented by vertical
    color bars. The windowed cumulative cosine similarities are represented by points and a approximated curve.
    The variances and similarities have separate y-axes. Each adjust to appropriate scale.

    The second histogram is composed of means in rotated space. It is similar to variances histogram, but with negative y-axis.
    """
    # Convert to numpy
    cosine_sim_windowed_np = cosine_similarity_windowed.cpu().numpy()
    variances_np = variances.cpu().numpy()
    means_rotated_np = means_rotated.cpu().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Variances and windowed cumulative cosine similarities with separate y-axes
    x = np.arange(len(variances_np))
    
    # Create first y-axis for variances
    ax1_var = ax1
    bars = ax1_var.bar(x, variances_np, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5, label='Explained Variance')
    ax1_var.set_ylabel('Explained Variance', color='blue')
    ax1_var.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for windowed cumulative cosine similarities
    ax1_sim = ax1_var.twinx()
    
    # Calculate mean across kernels for each component
    cosine_sim_mean = np.mean(cosine_sim_windowed_np, axis=0)
    
    # Plot points of cosine sim (only if there's actual data)
    if np.any(cosine_sim_mean != 0):
        ax1_sim.plot(x, cosine_sim_mean, 'o-', color='red', 
                    linewidth=2, markersize=4, alpha=0.8, label='Windowed Cosine Similarity')
    else:
        # If no cosine similarity data, plot a horizontal line at y=0 with different styling
        ax1_sim.plot(x, cosine_sim_mean, '--', color='gray', 
                    linewidth=1, alpha=0.5, label='No Kernel Update Data')
    
    # Add milestone lines
    colors = ['red', 'orange', 'green', 'purple']
    for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
        ax1_var.axvline(x=milestone_idx, color=colors[i], linestyle='--', linewidth=2, 
                       label=f'{milestone*100:.0f}% at component {milestone_idx}')
    
    ax1_var.set_xlabel('PCA Component Index')
    ax1_var.set_title(f'Variances and Windowed Cumulative Cosine Similarities(window size: {window_size})')
    
    # Combine legends from both axes
    lines1, labels1 = ax1_var.get_legend_handles_labels()
    lines2, labels2 = ax1_sim.get_legend_handles_labels()
    ax1_var.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1_var.grid(True, alpha=0.3)
    
    # Plot 2: Means in rotated space (similar to variances histogram but with negative y-axis)
    ax2.bar(x, -means_rotated_np, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=0.5, label='Means in Rotated Space')
    
    # Add milestone lines
    for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
        ax2.axvline(x=milestone_idx, color=colors[i], linestyle='--', linewidth=2, 
                   label=f'{milestone*100:.0f}% at component {milestone_idx}')
    
    ax2.set_xlabel('PCA Component Index')
    ax2.set_ylabel('Means in Rotated Space (Negative)')
    ax2.set_title('Means in Rotated Space (Negative Y-axis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Invert y-axis to show negative values properly
    ax2.invert_yaxis()
    
    plt.suptitle(f'PCA Analysis Summary\nModule: {module_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / f"{module_name}_histogram.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_polar(cosine_similarity_cumsum, kernel_update, module_name, milestones, milestones_indices, save_dir):
    """
    Plot kernel updates in a polar coordinates system.

    Args:
        cosine_similarity_cumsum(Tensor): The cumulative cosine similarity between kernel updates and PCA components. Shape: (num_kernels, num_components).
        kernel_update(Tensor): The kernel update. Shape: (num_kernels, dim_feat).
        module_name(str): The name of the module.
        milestones(list): The milestones of the accumulated variance ratio by which the subspaces and their
        milestones_indices(list): The indices of the PCA components according to the milestones.
        save_dir(str): The directory to save the plot.

    For each milestone, create a subplot with a polar coordinates. The positive x-axis represents subspace expanded
    by components that are less than the milestone index. The positive y-axis represents subspace expanded by components
    that are greater than or equal to the milestone index.
    For each kernel update, plot a vector on the subplot. Vector's angle with positive x-axis is determined by the cumulative
    cosine similarity in regard of the milestone. Vector's length is determined by its L2 norm. All vectors' lengths are
    normalized by the max.
    Use a cold-warm color scheme to represent the vectors' unsigned cosine similarity with x-axis.
    """
    num_milestones = len(milestones)
    # Increase figure size to avoid overlap
    fig, axes = plt.subplots(1, num_milestones, figsize=(6 * num_milestones, 8), subplot_kw=dict(projection='polar'))
    if num_milestones == 1:
        axes = [axes]
    
    # Get kernel norms for vector lengths and normalize by max
    kernel_norms = torch.norm(kernel_update, p=2, dim=1).cpu().numpy()
    max_norm = np.max(kernel_norms)
    normalized_norms = kernel_norms / max_norm
    
    # Convert to numpy
    cosine_sim_cumsum_np = cosine_similarity_cumsum.cpu().numpy()
    
    # Create colormap for warm-cold color scheme
    import matplotlib.cm as cm
    colormap = cm.get_cmap('RdYlBu_r')  # Red-Yellow-Blue reversed (warm to cold)
    
    for i, (milestone, milestone_idx) in enumerate(zip(milestones, milestones_indices)):
        ax = axes[i]
        
        # For each milestone, calculate the cumulative cosine similarity up to that point
        # This represents the projection on the subspace of components < milestone_idx
        x_cosine_similarities = cosine_sim_cumsum_np[:, milestone_idx]  # Shape: (num_kernels,)
        
        # Calculate the remaining cosine similarity (components >= milestone_idx)
        # This is the total similarity minus the cumulative up to milestone
        if milestone_idx < cosine_sim_cumsum_np.shape[1] - 1:
            y_cosine_similarities = cosine_sim_cumsum_np[:, -1] - cosine_sim_cumsum_np[:, milestone_idx]
        else:
            y_cosine_similarities = np.zeros_like(x_cosine_similarities)
        
        # Calculate angles from x-axis (0 to 2π)
        # Angle is determined by the ratio of y to x components
        angles = np.arctan2(y_cosine_similarities, x_cosine_similarities)
        # Convert to [0, 2π] range
        angles = np.where(angles < 0, angles + 2*np.pi, angles)
        
        # Normalize cosine similarities to [0, 1] for colormap
        x_cosine_normalized = (x_cosine_similarities + 1) / 2  # Map [-1, 1] to [0, 1]
        x_cosine_normalized = np.clip(x_cosine_normalized, 0, 1)
        
        # Plot vectors for each kernel update
        for j in range(len(cosine_sim_cumsum_np)):
            angle = angles[j]
            radius = normalized_norms[j]
            
            # Get color based on cosine similarity with positive x-axis
            color = colormap(x_cosine_normalized[j])
            
            # Plot vector in polar coordinates
            ax.plot([0, angle], [0, radius], color=color, linewidth=2, alpha=0.8)
            ax.scatter(angle, radius, color=color, s=20, alpha=0.8)
        
        # Set polar plot properties
        ax.set_ylim(0, 1.2)
        ax.set_theta_zero_location('E')  # Set 0 degrees to the right (positive x-axis)
        ax.set_theta_direction(1)  # Counterclockwise
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_title(f'{milestone*100:.0f}% Milestone\nSplit at Component {milestone_idx}', pad=20)
        # ax.set_xlabel('Subspace 1 (Components < milestone)')
        # ax.set_ylabel('Subspace 2 (Components >= milestone)')
    
    # Add colorbar to show the warm-cold color scheme
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=-1, vmax=1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=axes, shrink=0.8, aspect=20)
    # cbar.set_label('Cosine Similarity with Positive X-axis', rotation=270, labelpad=40)
    # cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # cbar.set_ticklabels(['-1 (Cold)', '-0.5', '0', '0.5', '1 (Warm)'])
    
    plt.suptitle(f'Kernel Update Projections in Polar Coordinates\nModule: {module_name}\n(Color represents cosine similarity with positive x-axis)', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / f"{module_name}_polar_projection.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    """Main function.
    
    Args:
        args(Namespace): The arguments.
    """
    # Load the PCA cache
    try:
        print(f"Loading PCA cache from: {args.pca_cache_path}")
        pca_cache = joblib.load(args.pca_cache_path)
        print(f"Successfully loaded PCA cache with {len(pca_cache)} modules")
    except FileNotFoundError:
        print(f"Error: PCA cache file not found: {args.pca_cache_path}")
        return
    except Exception as e:
        print(f"Error loading PCA cache: {e}")
        return
    
    # Load the base model
    if args.base_model is not None:
        try:
            print(f"Loading base model from: {args.base_model}")
            base_model = YOLO(args.base_model).model
            print("Successfully loaded base model")
        except FileNotFoundError:
            print(f"Error: Base model file not found: {args.base_model}")
            base_model = None
        except Exception as e:
            print(f"Error loading base model: {e}")
            base_model = None
    else:
        base_model = None
        
    # Load the incremental model
    if args.incremental_model is not None:
        try:
            print(f"Loading incremental model from: {args.incremental_model}")
            incremental_model = YOLO(args.incremental_model).model
            print("Successfully loaded incremental model")
        except FileNotFoundError:
            print(f"Error: Incremental model file not found: {args.incremental_model}")
            incremental_model = None
        except Exception as e:
            print(f"Error loading incremental model: {e}")
            incremental_model = None
    else:
        incremental_model = None

    # auto set device to cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Only process kernel updates if both models are provided
    if base_model is not None and incremental_model is not None:
        base_model.to(device).eval()
        incremental_model.to(device).eval()

        base_modules = [(name, module) for name, module in base_model.named_modules() if isinstance(module, Conv) and name in pca_cache.keys()]
        incremental_modules = [(name, module) for name, module in incremental_model.named_modules() if isinstance(module, Conv) and name in pca_cache.keys()]
        pbar = tqdm(zip(base_modules, incremental_modules), desc="Processing modules", total=len(base_modules))
        for (module_name, base_module), (incremental_module_name, incremental_module) in pbar:
            assert module_name == incremental_module_name, "Module names do not match"
            pbar.set_description(f"Processing module: {module_name}")
            pca_operator = pca_cache[module_name]
            base_kernel = flatten_augment_kernel(base_module)
            incremental_kernel = flatten_augment_kernel(incremental_module)
            if incremental_kernel.shape != base_kernel.shape:
                continue
            kernel_update = incremental_kernel - base_kernel # (num_kernels, dim_feat)

            # calculate projection of kernel updates on each pca components(normalized by L2 norm)
            components = F.normalize(pca_operator.components_.to(device), p=2, dim=1) # (num_components, dim_feat)
            proj = torch.matmul(kernel_update, components.T) # (num_kernels, num_components)

            # calculate cosine similarity
            cosine_similarity = proj / (torch.norm(kernel_update, p=2, dim=1, keepdim=True) + 1e-12) # (num_kernels, num_components)
            cosine_similarity_cumsum = (cosine_similarity**2).cumsum(dim=1).sqrt() # (num_kernels, num_components)
            window_size = 2 ** int(math.log2(cosine_similarity.shape[1] / 100))
            cosine_similarity_windowed = window_cumsum(cosine_similarity**2, window=window_size, dim=1).sqrt() # (num_kernels, num_components)

            # calculate milestones indices
            milestones = [0.75, 0.90, 0.95, 0.99]
            variances = pca_operator.explained_variance_.to(device) # (num_components)
            variances_cumsum = torch.cumsum(variances, dim=0) # (num_components)
            variances_cumsum_ratio = variances_cumsum / variances_cumsum[-1] # (num_components)
            milestones_indices = []
            for milestone in milestones:
                for i in range(len(variances_cumsum_ratio)):
                    if variances_cumsum_ratio[i] >= milestone:
                        milestones_indices.append(i)
                        break

            # other statistics
            means = pca_operator.mean_.to(device) # (dim_feat)
            # Transform means to the PCA coordinate space using the rotation matrix (components_)
            # means_rotated = means @ components.T, where components is the rotation matrix
            means_rotated = torch.matmul(means.unsqueeze(0), components.T).squeeze(0) # (num_components)

            # plot
            plot_histgram(cosine_similarity_windowed, window_size, variances, means_rotated, module_name, milestones, milestones_indices, args.save_dir)
            plot_polar(cosine_similarity_cumsum, kernel_update, module_name, milestones, milestones_indices, args.save_dir)
    else:
        print("Warning: base_model or incremental_model not provided. Skipping kernel update projection plots.")
        print("Only PCA variance and mean plots will be generated.")
        
        # Generate plots for PCA statistics only (without kernel updates)
        print(f"Found {len(pca_cache)} modules in PCA cache: {list(pca_cache.keys())}")
        
        if len(pca_cache) == 0:
            print("Error: PCA cache is empty. No plots will be generated.")
            return
            
        for module_name in pca_cache.keys():
            print(f"Processing module: {module_name}")
            pca_operator = pca_cache[module_name]
            
            # Check if PCA operator has valid data
            if not hasattr(pca_operator, 'explained_variance_') or len(pca_operator.explained_variance_) == 0:
                print(f"Warning: Module {module_name} has no PCA data. Skipping.")
                continue
                
            # calculate milestones indices
            milestones = [0.75, 0.90, 0.95, 0.99]
            variances = pca_operator.explained_variance_.to(device) # (num_components)
            variances_cumsum = torch.cumsum(variances, dim=0) # (num_components)
            variances_cumsum_ratio = variances_cumsum / variances_cumsum[-1] # (num_components)
            milestones_indices = []
            for milestone in milestones:
                for i in range(len(variances_cumsum_ratio)):
                    if variances_cumsum_ratio[i] >= milestone:
                        milestones_indices.append(i)
                        break

            # other statistics
            means = pca_operator.mean_.to(device) # (dim_feat)
            components = F.normalize(pca_operator.components_.to(device), p=2, dim=1) # (num_components, dim_feat)
            # Transform means to the PCA coordinate space using the rotation matrix (components_)
            # means_rotated = means @ components.T, where components is the rotation matrix
            means_rotated = torch.matmul(means.unsqueeze(0), components.T).squeeze(0) # (num_components)

            print(f"  - Variances shape: {variances.shape}")
            print(f"  - Means rotated shape: {means_rotated.shape}")
            print(f"  - Milestones indices: {milestones_indices}")
            
            # Create dummy cosine similarity data for plotting (all zeros)
            dummy_cosine_similarity_windowed = torch.zeros(1, len(variances), device=device)
            
            # plot only variance and mean histograms
            plot_histgram(dummy_cosine_similarity_windowed, 1, variances, means_rotated, module_name, milestones, milestones_indices, args.save_dir)
            print(f"  - Generated plot for {module_name}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_cache_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--incremental_model", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)