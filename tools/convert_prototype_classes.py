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
    
    torch.save({
        "prototypes": new_protos,
        "meta_info": data['meta_info']
    }, args.output)
    
    LOGGER.info(f"Saved converted prototypes to {args.output}")

if __name__ == "__main__":
    main()