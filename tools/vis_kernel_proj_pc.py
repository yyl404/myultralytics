"""
Visualize kernel projection on principal components.

This script visualizes the projection of kernel updates onto principal components
for each convolutional layer and group. It generates plots showing:
- Variance distribution (histogram)
- Elbow point annotation
- Projection magnitude curve with error bands

Usage:
    python tools/vis_kernel_proj_pc.py \
        --base_model <path/to/base_model.pt> \
        --incremental_model <path/to/incremental_model.pt> \
        --pca_cache <path/to/pca_cache.pkl> \
        --save_dir <path/to/save_dir> \
        [--device cuda] \
        [--layers "0,1,2"]
"""

import argparse
import warnings
import joblib
import os

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.engine.espreg import find_elbow_point


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available, using CPU instead.")
    
    base_model = YOLO(args.base_model)
    incremental_model = YOLO(args.incremental_model)
    base_model.to(args.device).eval()
    incremental_model.to(args.device).eval()

    if args.layers is None or args.layers == "":
        args.layers = list(range(len(base_model.model.model)))
    else:
        args.layers = [int(x.strip()) for x in args.layers.split(",")]
    
    kernel_updates = {}
    pca_cache = joblib.load(args.pca_cache)
    print(f"Loaded PCA cache from {args.pca_cache}")
    
    for name, module in base_model.model.named_modules():
        for layer in args.layers:
            if f"model.{layer}" in name and isinstance(module, nn.Conv2d) and "dfl" not in name:
                base_k, base_g, base_c_in, base_c_out = module.kernel_size, module.groups, module.in_channels, module.out_channels
                base_kernel = module.weight.data.reshape(base_g, base_c_out//base_g, base_c_in//base_g * base_k[0] * base_k[1])
                
                if module.bias is not None:
                    base_kernel = torch.concat((base_kernel, module.bias.data.reshape(base_g, base_c_out//base_g, 1)), dim=2)
                
                incremental_module = incremental_model.model.get_submodule(name)
                inc_k, inc_g, inc_c_in, inc_c_out = incremental_module.kernel_size, incremental_module.groups, incremental_module.in_channels, incremental_module.out_channels
                
                if base_c_in == inc_c_in and base_g == inc_g and base_k == inc_k:
                    incremental_kernel = incremental_module.weight.data.reshape(inc_g, inc_c_out//inc_g, inc_c_in//inc_g * inc_k[0] * inc_k[1])
                    
                    if incremental_module.bias is not None:
                        incremental_kernel = torch.concat((incremental_kernel, incremental_module.bias.data.reshape(inc_g, inc_c_out//inc_g, 1)), dim=2)
                    
                    min_c_out = min(base_c_out, inc_c_out)
                    base_kernel_trimmed = base_kernel[:, :min_c_out//base_g, :]
                    incremental_kernel_trimmed = incremental_kernel[:, :min_c_out//inc_g, :]
                    kernel_updates[name] = incremental_kernel_trimmed - base_kernel_trimmed
                else:
                    warnings.warn(f"Skipping {name}: channel/groups/kernel_size mismatch "
                                f"(base: c_in={base_c_in}, c_out={base_c_out}, g={base_g}, k={base_k}; "
                                f"inc: c_in={inc_c_in}, c_out={inc_c_out}, g={inc_g}, k={inc_k})")

                break
    
    print(f"Found {len(kernel_updates)} matching layers for visualization")

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Count total visualizations to create
    total_visualizations = 0
    for name in kernel_updates.keys():
        if name in pca_cache:
            total_visualizations += kernel_updates[name].shape[0]
    
    print(f"Generating {total_visualizations} visualizations...")
    
    visualization_count = 0
    skipped_count = 0
    
    for name in kernel_updates.keys():
        if name in pca_cache:
            for ig in range(kernel_updates[name].shape[0]):
                visualization_count += 1
                if total_visualizations > 0:
                    progress = (visualization_count / total_visualizations) * 100
                    print(f"[{visualization_count}/{total_visualizations} ({progress:.1f}%)] Processing {name}, group {ig}...", end='\r')
                entry = pca_cache[name][ig]
                if not isinstance(entry, dict) or "state" not in entry:
                    raise TypeError(
                        f"PCA cache entry for module '{name}' group {ig} has an unexpected format "
                        f"(expected a dict with a 'state' key, got {type(entry)}). "
                        f"Regenerate the cache with tools/pca.py."
                    )
                components = entry["state"]["components_"]
                variances = entry["state"]["explained_variance_"]
                
                if not isinstance(components, torch.Tensor):
                    components = torch.from_numpy(components).float()
                if not isinstance(variances, torch.Tensor):
                    variances = torch.from_numpy(variances).float()
                
                kernel_update_ig = kernel_updates[name][ig]
                # PCA caches are serialized on CPU; move to the kernel update's device
                components = components.to(kernel_update_ig.device)
                variances = variances.to(kernel_update_ig.device)
                proj_norm = kernel_update_ig @ components.T
                proj_magnitudes = torch.abs(proj_norm)
                proj_mean = torch.mean(proj_magnitudes, dim=0)
                proj_std = torch.std(proj_magnitudes, dim=0)
                
                elbow_idx = find_elbow_point(variances)
                elbow_percentage = (elbow_idx + 1) / len(variances) * 100
                
                fig, ax = plt.subplots(figsize=(12, 8))
                variances_np = variances.cpu().numpy() if isinstance(variances, torch.Tensor) else variances
                x_indices = np.arange(len(variances_np))
                ax.bar(x_indices, variances_np, alpha=0.6, color='blue', label='Variances')
                ax.axvline(x=elbow_idx, color='red', linestyle='--', linewidth=2, 
                          label=f'Elbow point (index {elbow_idx}, {elbow_percentage:.2f}%)')
                ax.text(elbow_idx, ax.get_ylim()[1] * 0.95, f'{elbow_percentage:.2f}%', 
                       rotation=0, horizontalalignment='center', verticalalignment='top', fontsize=25,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
                
                ax2 = ax.twinx()
                ax2.plot(x_indices, proj_mean.cpu().numpy(), 'g-', linewidth=2, label='Projection mean')
                ax2.fill_between(x_indices, 
                                (proj_mean - proj_std).cpu().numpy(),
                                (proj_mean + proj_std).cpu().numpy(),
                                alpha=0.3, color='green', label='±1 std')
                
                ax.set_xlabel('Principal Component Index', fontsize=25, labelpad=15)
                ax.set_ylabel('Eigenvalue', fontsize=25, color='blue', labelpad=15)
                ax2.set_ylabel('Projection Magnitude', fontsize=25, color='green', labelpad=15)
                ax.tick_params(axis='x', labelsize=25)
                ax.tick_params(axis='y', labelsize=25, labelcolor='blue')
                ax2.tick_params(axis='y', labelsize=25, labelcolor='green')
                
                safe_name = name.replace('.', '_').replace('/', '_')
                ax.set_title(f'Layer: {name}, Group: {ig}\nEigenvalue Distribution with Projection Magnitude', 
                           fontsize=25, fontweight='bold')
                
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=25, 
                         framealpha=0.9)
                plt.tight_layout()
                
                save_path = os.path.join(args.save_dir, f'{safe_name}_group_{ig}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
        else:
            skipped_count += 1
            warnings.warn(f"Layer {name} not found in PCA cache, skipping")
    
    print()  # New line after progress output
    print(f"Total visualizations created: {visualization_count}")
    if skipped_count > 0:
        print(f"Warning: {skipped_count} layer(s) skipped (not found in PCA cache)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize kernel projection on principal components")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model checkpoint (.pt file)")
    parser.add_argument("--incremental_model", type=str, required=True,
                       help="Path to the incremental model checkpoint (.pt file)")
    parser.add_argument("--pca_cache", type=str, required=True,
                       help="Path to the PCA cache file (.pkl file)")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save visualization plots")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for model inference (default: cuda)")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices to visualize (e.g., '0,1,2'). If not specified, all layers will be visualized")
    args = parser.parse_args()
    
    main(args)