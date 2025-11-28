#!/usr/bin/env python3
"""
Convert prototype class IDs from original model to expanded model.

This script converts prototypes generated for the original model (with fewer classes)
to match the class IDs in an expanded model (with more classes).

The conversion is based on class name mapping:
1. Load original model to get original class names and IDs
2. Load expanded model to get expanded class names and IDs
3. Create a mapping from original class IDs to expanded class IDs based on class names
4. Reorganize prototypes to match the expanded model's class structure

Usage:
    python tools/convert_prototype_classes.py \
        --prototypes <path_to_original_prototypes.pt> \
        --original_model <path_to_original_model.pt> \
        --expanded_model <path_to_expanded_model.pt> \
        --output <path_to_converted_prototypes.pt>

Arguments:
    --prototypes: Path to original prototypes file (.pt file) [required]
    --original_model: Path to original model (.pt file) [required]
    --expanded_model: Path to expanded model (.pt file) [required]
    --output: Path to save converted prototypes (.pt file) [required]

Example:
    python tools/convert_prototype_classes.py \
        --prototypes runs/task-1/prototypes.pt \
        --original_model runs/task-1/best.pt \
        --expanded_model runs/task-2/task-1-best-expanded.pt \
        --output runs/task-2/prototypes_converted.pt
"""

import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def get_class_names(model):
    """Get class names from a YOLO model."""
    if hasattr(model.model, 'names'):
        names = model.model.names
        if isinstance(names, dict):
            # Convert dict to list, sorted by key
            max_id = max(names.keys()) if names else -1
            class_names = [names.get(i, f"class_{i}") for i in range(max_id + 1)]
        else:
            class_names = list(names) if isinstance(names, (list, tuple)) else names
        return class_names
    return None


def create_class_mapping(original_names, expanded_names):
    """
    Create a mapping from original class IDs to expanded class IDs.
    
    Args:
        original_names: List of class names in original model
        expanded_names: List of class names in expanded model
    
    Returns:
        dict: Mapping from original class ID to expanded class ID
    """
    mapping = {}
    
    # Create a reverse mapping from name to ID in expanded model
    expanded_name_to_id = {name: idx for idx, name in enumerate(expanded_names)}
    
    # Map each original class to its position in expanded model
    for orig_id, orig_name in enumerate(original_names):
        if orig_name in expanded_name_to_id:
            expanded_id = expanded_name_to_id[orig_name]
            mapping[orig_id] = expanded_id
            LOGGER.info(f"Mapping class {orig_id} ({orig_name}) -> {expanded_id} ({expanded_names[expanded_id]})")
        else:
            LOGGER.warning(f"Class {orig_id} ({orig_name}) not found in expanded model!")
    
    return mapping


def convert_prototypes(prototypes, class_mapping, original_num_classes, expanded_num_classes, num_layers):
    """
    Convert prototypes from original class IDs to expanded class IDs.
    
    The prototypes are stored as a list of tensors (one per layer), where each tensor
    contains prototypes from all classes concatenated. The structure is:
    [num_prototypes_all_classes, feature_dim + reg_dim + cls_dim]
    
    where:
    - feature_dim = in_channels * 3 * 3 (flattened 3x3 feature patch)
    - reg_dim = reg_max * 4 (regression output channels)
    - cls_dim = num_classes (classification output channels)
    
    We need to:
    1. Split each prototype into feature+reg and cls parts
    2. Expand cls from original_num_classes to expanded_num_classes
    3. Map original class outputs to expanded class positions based on class name mapping
    4. Reconcatenate feature+reg and expanded cls
    
    Args:
        prototypes: List of tensors, one per layer, containing concatenated prototypes
        class_mapping: Dict mapping original class ID to expanded class ID
        original_num_classes: Number of classes in original model
        expanded_num_classes: Number of classes in expanded model
        num_layers: Number of detection layers
    
    Returns:
        List of tensors with expanded class dimensions
    """
    converted_prototypes = []
    
    for layer_idx in range(num_layers):
        if len(prototypes[layer_idx]) == 0:
            # Empty layer, create empty tensor with correct dimensions
            # We need to know the dimensions, but we can't infer from empty tensor
            # So we'll keep it empty and let the training code handle it
            converted_prototypes.append(torch.empty(0))
            continue
        
        layer_prototypes = prototypes[layer_idx]  # [num_prototypes, feature_dim + reg_dim + cls_dim]
        num_prototypes, total_dim = layer_prototypes.shape
        
        # Calculate dimensions
        # total_dim = feature_dim + reg_dim + original_cls_dim
        # We know original_cls_dim = original_num_classes
        # But we don't know feature_dim and reg_dim separately
        # However, we can infer: feature_dim + reg_dim = total_dim - original_num_classes
        
        feature_reg_dim = total_dim - original_num_classes
        
        # Split prototypes into feature+reg and cls parts
        feature_reg = layer_prototypes[:, :feature_reg_dim]  # [num_prototypes, feature_dim + reg_dim]
        cls_original = layer_prototypes[:, feature_reg_dim:]  # [num_prototypes, original_num_classes]
        
        # Expand cls from original_num_classes to expanded_num_classes
        # Initialize with zeros, then copy values based on class mapping
        cls_expanded = torch.zeros(
            num_prototypes, 
            expanded_num_classes, 
            device=cls_original.device, 
            dtype=cls_original.dtype
        )
        
        # Map original class outputs to expanded class outputs
        for orig_cls_id, expanded_cls_id in class_mapping.items():
            cls_expanded[:, expanded_cls_id] = cls_original[:, orig_cls_id]
        
        # Reconcatenate: feature_reg + cls_expanded
        converted_layer = torch.cat([feature_reg, cls_expanded], dim=1)
        converted_prototypes.append(converted_layer)
        
        LOGGER.info(
            f"Layer {layer_idx}: Converted {num_prototypes} prototypes from "
            f"{original_num_classes} classes to {expanded_num_classes} classes "
            f"(shape: {layer_prototypes.shape} -> {converted_layer.shape})"
        )
    
    return converted_prototypes


def convert_prototype_classes(
    prototypes_path: str,
    original_model_path: str,
    expanded_model_path: str,
    output_path: str
):
    """
    Convert prototype class IDs from original model to expanded model.
    
    Args:
        prototypes_path: Path to original prototypes file
        original_model_path: Path to original model file
        expanded_model_path: Path to expanded model file
        output_path: Path to save converted prototypes
    """
    LOGGER.info(f"Loading original prototypes from {prototypes_path}")
    prototypes = torch.load(prototypes_path, map_location='cpu')
    
    LOGGER.info(f"Loading original model from {original_model_path}")
    original_model = YOLO(original_model_path)
    original_names = get_class_names(original_model)
    original_num_classes = len(original_names) if original_names else original_model.model.model[-1].nc
    
    LOGGER.info(f"Loading expanded model from {expanded_model_path}")
    expanded_model = YOLO(expanded_model_path)
    expanded_names = get_class_names(expanded_model)
    expanded_num_classes = len(expanded_names) if expanded_names else expanded_model.model.model[-1].nc
    
    LOGGER.info(f"Original model: {original_num_classes} classes")
    LOGGER.info(f"Expanded model: {expanded_num_classes} classes")
    
    if original_names is None or expanded_names is None:
        raise ValueError("Could not extract class names from models")
    
    # Create class mapping
    class_mapping = create_class_mapping(original_names, expanded_names)
    
    if len(class_mapping) == 0:
        raise ValueError("No valid class mappings found! Check if class names match.")
    
    # Get number of layers from expanded model
    detect_head = expanded_model.model.model[-1]
    num_layers = detect_head.nl
    
    LOGGER.info(f"Number of detection layers: {num_layers}")
    
    # Convert prototypes
    converted_prototypes = convert_prototypes(
        prototypes,
        class_mapping,
        original_num_classes,
        expanded_num_classes,
        num_layers
    )
    
    # Save converted prototypes
    torch.save(converted_prototypes, output_path)
    LOGGER.info(f"Converted prototypes saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert prototype class IDs from original model to expanded model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--prototypes",
        type=str,
        required=True,
        help="Path to original prototypes file (.pt file)"
    )
    parser.add_argument(
        "--original_model",
        type=str,
        required=True,
        help="Path to original model (.pt file)"
    )
    parser.add_argument(
        "--expanded_model",
        type=str,
        required=True,
        help="Path to expanded model (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted prototypes (.pt file)"
    )
    args = parser.parse_args()
    
    convert_prototype_classes(
        prototypes_path=args.prototypes,
        original_model_path=args.original_model,
        expanded_model_path=args.expanded_model,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

