""" Expand the importance file to support expanded model head.

When the model head is expanded to support more classes, the importance file
needs to be expanded accordingly. This script expands the importance values
for the detection head output channels (cv3.2.weight and cv3.2.bias).

Usage:
    $ python tools/expand_importance.py \
        --old_importance <path/to/old_importance.pth> \
        --old_model <path/to/old_model.pt> \
        --new_model <path/to/new_model.pt> \
        --save_path <path/to/expanded_importance.pth>

    Arguments:
        --old_importance: Path to the old importance file (.pth file)
        --old_model: Path to the old model checkpoint (.pt file)
        --new_model: Path to the expanded model checkpoint (.pt file)
        --save_path: Path where the expanded importance will be saved
        --zero_importance_init: Whether to initialize importance of new channels to 0
        --copy_importance_init: Whether to copy importance from old channels to new channels

Examples:
    $ python tools/expand_importance.py \
        --old_importance runs/task-1/importance.pth \
        --old_model runs/task-1/best.pt \
        --new_model runs/task-2/task-1-best-expanded.pt \
        --save_path runs/task-2/task-1-importance-expanded.pth
"""

import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def expand_importance(old_importance_path, old_model_path, new_model_path, save_path,
                      zero_importance_init=False, copy_importance_init=False):
    """Expand importance file to match expanded model head.
    
    Args:
        old_importance_path: Path to the old importance file
        old_model_path: Path to the old model checkpoint
        new_model_path: Path to the expanded model checkpoint
        save_path: Path to save the expanded importance file
        zero_importance_init: If True, initialize new channel importance to 0
        copy_importance_init: If True, copy importance from old channels to new channels
            (uses the same channel mapping as expand_model_head.py)
    """
    # Load old importance
    LOGGER.info(f"Loading old importance from {old_importance_path}...")
    with open(old_importance_path, "rb") as f:
        old_state = torch.load(f)
    
    old_importance = old_state.get('running_importance', {})
    old_n_batch = old_state.get('n_batch', {})
    old_modules = old_state.get('modules', [])
    device = old_state.get('device', 'cpu')
    
    # Load models to determine channel mapping
    LOGGER.info(f"Loading old model from {old_model_path}...")
    old_model = YOLO(old_model_path)
    old_model_dict = old_model.model.state_dict()
    
    LOGGER.info(f"Loading new model from {new_model_path}...")
    new_model = YOLO(new_model_path)
    new_model_dict = new_model.model.state_dict()
    
    # Determine the last layer index (detect layer)
    # Find cv3.2.weight in both models to determine channel mapping
    old_cv3_weight_key = None
    new_cv3_weight_key = None
    
    for key in old_model_dict.keys():
        if 'cv3' in key and key.endswith('.2.weight'):
            old_cv3_weight_key = key
            break
    
    for key in new_model_dict.keys():
        if 'cv3' in key and key.endswith('.2.weight'):
            new_cv3_weight_key = key
            break
    
    # Initialize channel mapping variables
    channel_map = {}
    old_num_classes = 0
    new_num_classes = 0
    
    if old_cv3_weight_key is None or new_cv3_weight_key is None:
        LOGGER.warning("Could not find cv3.2.weight in models, skipping importance expansion for detect layer")
    else:
        old_cv3_weight = old_model_dict[old_cv3_weight_key]
        new_cv3_weight = new_model_dict[new_cv3_weight_key]
        
        old_num_classes = old_cv3_weight.shape[0]
        new_num_classes = new_cv3_weight.shape[0]
        
        LOGGER.info(f"Old model has {old_num_classes} output channels, new model has {new_num_classes} output channels")
        
        # Determine channel mapping (same logic as expand_model_head.py)
        # Get class names to determine mapping
        old_classes = [old_model.names[i] for i in sorted(old_model.names.keys())]
        new_classes = [new_model.names[i] for i in sorted(new_model.names.keys())]
        
        # Create mapping from old class index to new class index
        for old_idx, old_cls in enumerate(old_classes):
            if old_cls in new_classes:
                new_idx = new_classes.index(old_cls)
                channel_map[old_idx] = new_idx
        
        LOGGER.info(f"Channel mapping: {len(channel_map)} old channels mapped to new channels")
    
    # Create expanded importance dictionary
    new_importance = {}
    new_n_batch = {}
    
    # Copy all non-detect layer importance values
    for param_name, importance_val in old_importance.items():
        # Check if this is a detect layer parameter (cv3.2.weight or cv3.2.bias)
        is_detect_weight = 'cv3' in param_name and param_name.endswith('.2.weight')
        is_detect_bias = 'cv3' in param_name and param_name.endswith('.2.bias')
        
        if is_detect_weight or is_detect_bias:
            # This is a detect layer parameter, need to expand it
            if old_cv3_weight_key is None:
                # Skip if we couldn't find the detect layer
                continue
            
            # Get the corresponding parameter name in new model
            # Extract the layer prefix (e.g., "model.22.cv3")
            parts = param_name.split('.')
            # Find cv3 index
            cv3_idx = None
            for i, part in enumerate(parts):
                if part == 'cv3':
                    cv3_idx = i
                    break
            
            if cv3_idx is None:
                # Not a cv3 parameter, copy as is
                new_importance[param_name] = importance_val.clone()
                if param_name in old_n_batch:
                    new_n_batch[param_name] = old_n_batch[param_name]
                continue
            
            # Reconstruct new parameter name (should be same structure)
            new_param_name = param_name  # Parameter name should be the same in expanded model
            
            # Check if this parameter exists in new model
            if new_param_name not in new_model_dict:
                LOGGER.warning(f"Parameter {new_param_name} not found in new model, skipping")
                continue
            
            new_param_shape = new_model_dict[new_param_name].shape
            old_param_shape = importance_val.shape
            
            # Expand importance tensor
            if is_detect_weight:
                # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                # We need to expand the first dimension (out_channels)
                new_importance_tensor = torch.zeros(new_param_shape, 
                                                    dtype=importance_val.dtype,
                                                    device=importance_val.device)
                
                # Copy importance from old channels to new channels according to mapping
                for old_idx, new_idx in channel_map.items():
                    if old_idx < old_param_shape[0] and new_idx < new_param_shape[0]:
                        new_importance_tensor[new_idx] = importance_val[old_idx].clone()
                
                # Initialize new channels
                if zero_importance_init:
                    # New channels already initialized to 0
                    pass
                elif copy_importance_init:
                    # Copy from nearest old channel (use last old channel's importance)
                    if old_num_classes > 0 and channel_map:
                        last_old_idx = old_num_classes - 1
                        # Find which new channel corresponds to last old channel
                        last_new_idx = None
                        if last_old_idx in channel_map.keys():
                            last_new_idx = channel_map[last_old_idx]
                        else:
                            # Use the maximum new index from channel_map
                            last_new_idx = max(channel_map.values())
                        
                        # Initialize new channels with the importance from last_new_idx
                        for new_idx in range(new_num_classes):
                            if new_idx not in channel_map.values():
                                new_importance_tensor[new_idx] = new_importance_tensor[last_new_idx].clone()
                
                new_importance[new_param_name] = new_importance_tensor
                if param_name in old_n_batch:
                    new_n_batch[new_param_name] = old_n_batch[param_name]
                    
            elif is_detect_bias:
                # Bias shape: [out_channels]
                # We need to expand the first dimension
                new_importance_tensor = torch.zeros(new_param_shape,
                                                    dtype=importance_val.dtype,
                                                    device=importance_val.device)
                
                # Copy importance from old channels to new channels according to mapping
                for old_idx, new_idx in channel_map.items():
                    if old_idx < old_param_shape[0] and new_idx < new_param_shape[0]:
                        new_importance_tensor[new_idx] = importance_val[old_idx].clone()
                
                # Initialize new channels
                if zero_importance_init:
                    # New channels already initialized to 0
                    pass
                elif copy_importance_init:
                    # Copy from nearest old channel
                    if old_num_classes > 0 and channel_map:
                        last_old_idx = old_num_classes - 1
                        # Find which new channel corresponds to last old channel
                        last_new_idx = None
                        if last_old_idx in channel_map.keys():
                            last_new_idx = channel_map[last_old_idx]
                        else:
                            # Use the maximum new index from channel_map
                            last_new_idx = max(channel_map.values())
                        
                        # Initialize new channels with the importance from last_new_idx
                        for new_idx in range(new_num_classes):
                            if new_idx not in channel_map.values():
                                new_importance_tensor[new_idx] = new_importance_tensor[last_new_idx].clone()
                
                new_importance[new_param_name] = new_importance_tensor
                if param_name in old_n_batch:
                    new_n_batch[new_param_name] = old_n_batch[param_name]
        else:
            # Not a detect layer parameter, copy as is
            new_importance[param_name] = importance_val.clone()
            if param_name in old_n_batch:
                new_n_batch[param_name] = old_n_batch[param_name]
    
    # Update modules list (should be the same, but update if needed)
    new_modules = old_modules.copy()
    
    # Save expanded importance
    new_state = {
        'running_importance': new_importance,
        'n_batch': new_n_batch,
        'modules': new_modules,
        'device': device,
    }
    
    LOGGER.info(f"Saving expanded importance to {save_path}...")
    import os
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with open(save_path, "wb") as f:
        torch.save(new_state, f)
    
    LOGGER.info(f"Expanded importance saved to {save_path}")
    LOGGER.info(f"Total parameters: {len(new_importance)}")
    total_params = sum(p.numel() for p in new_importance.values())
    LOGGER.info(f"Total parameter count: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand importance file to match expanded model head")
    parser.add_argument("--old_importance", type=str, required=True,
                       help="Path to the old importance file (.pth file)")
    parser.add_argument("--old_model", type=str, required=True,
                       help="Path to the old model checkpoint (.pt file)")
    parser.add_argument("--new_model", type=str, required=True,
                       help="Path to the expanded model checkpoint (.pt file)")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path where the expanded importance will be saved")
    parser.add_argument("--zero_importance_init", action="store_true",
                       help="Initialize importance of new channels to 0")
    parser.add_argument("--copy_importance_init", action="store_true",
                       help="Copy importance from old channels to new channels (uses channel mapping)")
    
    args = parser.parse_args()
    
    expand_importance(
        args.old_importance,
        args.old_model,
        args.new_model,
        args.save_path,
        args.zero_importance_init,
        args.copy_importance_init
    )

