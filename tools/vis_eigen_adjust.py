"""
Visualize eigen value adjustment after log and sigmoid transformation.

This script visualizes the adjustment of eigen values (variances) using the same
transformation as in ESPReg loss: log translation + sigmoid scaling. It generates plots showing:
- Original eigen values (histogram)
- Adjusted eigen values (curve after log + sigmoid)
- Elbow point annotation
- Hyperparameter values

Usage:
    python tools/vis_eigen_adjust.py \
        --pca_cache <path/to/pca_cache.pkl> \
        --save_dir <path/to/save_dir>
"""

import argparse
import warnings
import joblib
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics.engine.espreg import find_elbow_point, adjust_eigen_values


def main(args):
    pca_cache = joblib.load(args.pca_cache)
    print(f"Loaded PCA cache from {args.pca_cache}")
    
    eigen_values = {}
    for name in pca_cache.keys():
        _eigen_values = []
        for ig in range(len(pca_cache[name])):
            eigen_val = pca_cache[name][ig].explained_variance_
            if not isinstance(eigen_val, torch.Tensor):
                eigen_val = torch.from_numpy(eigen_val).float()
            _eigen_values.append(eigen_val)
        eigen_values[name] = torch.stack(_eigen_values)
    
    adjusted_eigen_values = adjust_eigen_values(eigen_values)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Count total visualizations to create
    total_visualizations = sum(eigen_vals.shape[0] for eigen_vals in eigen_values.values())
    print(f"Generating {total_visualizations} visualizations...")
    
    visualization_count = 0
    
    for name, eigen_vals in eigen_values.items():
        g, n = eigen_vals.shape
        adjusted_vals = adjusted_eigen_values[name]
        
        for ig in range(g):
            visualization_count += 1
            if total_visualizations > 0:
                progress = (visualization_count / total_visualizations) * 100
                print(f"[{visualization_count}/{total_visualizations} ({progress:.1f}%)] Processing {name}, group {ig}...", end='\r')
            original_vals = eigen_vals[ig]
            adjusted_vals_group = adjusted_vals[ig]
            
            elbow_idx = find_elbow_point(original_vals)
            elbow_percentage = (elbow_idx + 1) / len(original_vals) * 100
            
            original_vals_np = original_vals.cpu().numpy() if isinstance(original_vals, torch.Tensor) else original_vals
            adjusted_vals_np = adjusted_vals_group.cpu().numpy() if isinstance(adjusted_vals_group, torch.Tensor) else adjusted_vals_group
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            x_indices = np.arange(len(original_vals_np))
            ax1.bar(x_indices, original_vals_np, alpha=0.6, color='blue', label='Original Eigen Values')
            ax1.axvline(x=elbow_idx, color='red', linestyle='--', linewidth=2, 
                       label=f'Elbow point (index {elbow_idx}, {elbow_percentage:.2f}%)')
            ax1.text(elbow_idx, ax1.get_ylim()[1] * 0.95, f'{elbow_percentage:.2f}%', 
                    rotation=90, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax2 = ax1.twinx()
            ax2.plot(x_indices, adjusted_vals_np, 'g-', linewidth=2, label='Adjusted Eigen Values')
            
            ax1.set_xlabel('Principal Component Index', fontsize=12)
            ax1.set_ylabel('Original Eigen Value', fontsize=12, color='blue')
            ax2.set_ylabel('Adjusted Eigen Value (normalized)', fontsize=12, color='green')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='green')
            
            safe_name = name.replace('.', '_').replace('/', '_')
            ax1.set_title(f'Layer: {name}, Group: {ig}\nEigen Value Adjustment', 
                         fontsize=14, fontweight='bold')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            textstr = f'Elbow index: {elbow_idx}\nElbow percentage: {elbow_percentage:.2f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            save_path = os.path.join(args.save_dir, f'{safe_name}_group_{ig}_eigen_adjust.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    print()  # New line after progress output
    print(f"Total visualizations created: {visualization_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize eigen value adjustment after normalization")
    parser.add_argument("--pca_cache", type=str, required=True,
                       help="Path to the PCA cache file (.pkl file)")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save visualization plots")
    args = parser.parse_args()
    
    main(args)

