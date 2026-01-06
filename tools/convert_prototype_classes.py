#!/usr/bin/env python3
"""
Convert prototype class IDs from an original model to an expanded model schema.
Rearranges the classification segment of the prototype tensors.
"""

import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def get_names(model):
    """Safely extracts class name list from YOLO model."""
    names = model.model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)

def convert_prototypes(prototypes, mapping, dim_old_cls, dim_new_cls):
    """
    Rebuilds prototype tensors with expanded classification channels.
    Structure: [Features | Regression | Classification | ValidMask]
    """
    converted = []
    
    for layer_idx, tensor in enumerate(prototypes):
        if tensor is None:
            converted.append(None)
            continue
            
        N, total_dim = tensor.shape
        dim_mask = 25
        dim_feat_reg = total_dim - dim_old_cls - dim_mask
        
        # Split
        part_feat_reg = tensor[:, :dim_feat_reg]
        part_cls_old  = tensor[:, dim_feat_reg : dim_feat_reg + dim_old_cls]
        part_mask     = tensor[:, -dim_mask:]
        
        # Remap Classification
        part_cls_new = torch.zeros((N, dim_new_cls), dtype=tensor.dtype, device=tensor.device)
        for old_id, new_id in mapping.items():
            if old_id < dim_old_cls:
                part_cls_new[:, new_id] = part_cls_old[:, old_id]
        
        # Reassemble
        new_tensor = torch.cat([part_feat_reg, part_cls_new, part_mask], dim=1)
        converted.append(new_tensor)
        
        LOGGER.info(f"Layer {layer_idx}: {N} prototypes converted. Shape {tensor.shape} -> {new_tensor.shape}")
        
    return converted


def convert_neg_prototypes(prototypes_neg, mapping, dim_old_cls, dim_new_cls):
    """
    Rebuilds negative prototype tensors with expanded classification channels.
    Structure: [Features | cls_valid_mask | pad_mask]
    
    Args:
        prototypes_neg: List of negative prototype tensors
        mapping: Dictionary mapping old class IDs to new class IDs
        dim_old_cls: Number of classes in old model
        dim_new_cls: Number of classes in new model
    """
    converted = []
    
    for layer_idx, tensor in enumerate(prototypes_neg):
        if tensor is None:
            converted.append(None)
            continue
            
        N, total_dim = tensor.shape
        dim_mask = 25
        dim_cls_old = dim_old_cls
        
        # Calculate feat_dim for this layer: [feat(C*25) | cls_valid_mask(nc_old) | pad_mask(25)]
        feat_dim = total_dim - dim_cls_old - dim_mask
        
        # Split: [feat(C*25) | cls_valid_mask(nc_old) | pad_mask(25)]
        part_feat = tensor[:, :feat_dim]
        part_cls_valid_old = tensor[:, feat_dim : feat_dim + dim_cls_old]
        part_mask = tensor[:, -dim_mask:]
        
        # Remap cls_valid_mask
        part_cls_valid_new = torch.zeros((N, dim_new_cls), dtype=tensor.dtype, device=tensor.device)
        for old_id, new_id in mapping.items():
            if old_id < dim_cls_old:
                # Copy the cls_valid_mask value to the new position
                part_cls_valid_new[:, new_id] = part_cls_valid_old[:, old_id]
        
        # Reassemble: [feat(C*25) | cls_valid_mask(nc_new) | pad_mask(25)]
        new_tensor = torch.cat([part_feat, part_cls_valid_new, part_mask], dim=1)
        converted.append(new_tensor)
        
        LOGGER.info(f"Layer {layer_idx}: {N} negative prototypes converted. Shape {tensor.shape} -> {new_tensor.shape}")
        
    return converted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--original_model", required=True)
    parser.add_argument("--expanded_model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load resources
    data = torch.load(args.prototypes, map_location='cpu')
    model_src = YOLO(args.original_model)
    model_dst = YOLO(args.expanded_model)
    
    names_src = get_names(model_src)
    names_dst = get_names(model_dst)
    
    # Build Mapping
    mapping = {}
    name_to_id_dst = {n: i for i, n in enumerate(names_dst)}
    
    for i, name in enumerate(names_src):
        if name in name_to_id_dst:
            mapping[i] = name_to_id_dst[name]
        else:
            LOGGER.warning(f"Class '{name}' from original model not found in expanded model.")

    if not mapping:
        raise ValueError("No matching classes found between models.")

    # Convert
    nc_src = len(names_src)
    nc_dst = len(names_dst)
    
    new_protos = convert_prototypes(data['prototypes'], mapping, nc_src, nc_dst)
    
    # Prepare output dictionary
    output_data = {
        "prototypes": new_protos,
        "meta_info": data.get('meta_info', [])
    }
    
    # Convert negative prototypes if they exist
    if 'prototypes_neg' in data:
        new_protos_neg = convert_neg_prototypes(
            data['prototypes_neg'], mapping, nc_src, nc_dst
        )
        output_data['prototypes_neg'] = new_protos_neg
        output_data['meta_info_neg'] = data.get('meta_info_neg', [])
        LOGGER.info("Converted negative prototypes.")
    
    torch.save(output_data, args.output)
    
    LOGGER.info(f"Saved converted prototypes to {args.output}")

if __name__ == "__main__":
    main()