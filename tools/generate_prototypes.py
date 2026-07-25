#!/usr/bin/env python3
"""
Generate prototypes for Prototype Replay (PRoRP) mechanism.

This script generates prototypes by:
1. Loading a trained YOLO model and dataset
2. For each training image, performing forward pass to get detection head features
3. For each ground truth bbox, collecting all detection vectors from all 3 output layers
   that have IOU > 0.5 and correct classification (max class matches GT class)
4. Extracting 5x5 feature patches and corresponding regression/classification outputs
5. Organizing prototypes by layer and ground truth class
6. Using K-means clustering to select representative prototypes (k_center=num_protos per class, default: 10)
7. Saving prototypes as a list of tensors (one per layer), where each tensor contains:
   [num_prototypes, in_channels*5*5 + reg_out_channels + cls_out_channels]

The output format is a list of tensors, one for each detection layer:
- Each tensor shape: [num_prototypes_all_classes, feature_dim + reg_dim + cls_dim]
- Prototypes from all classes are concatenated within each layer

Usage:
    python tools/generate_prototypes.py \
        --model <path_to_model.pt> \
        --data <path_to_dataset.yaml> \
        --output <path_to_output.pt> \
        [--device 0] \
        [--imgsz 640] \
        [--num_protos 10] \
        [--vis_dir <path_to_vis_dir>]

Arguments:
    --model: Path to trained YOLO model (.pt file) [required]
    --data: Path to dataset YAML configuration file [required]
    --output: Path to save generated prototypes (.pt file) [required]
    --device: Device to use (e.g., '0' for GPU 0, 'cpu' for CPU) [default: '0']
    --imgsz: Image size for inference [default: 640]
    --num_protos: Number of prototypes per class to select via K-means clustering [default: 10]
    --vis_dir: Optional directory to save visualization of prototypes [optional]

Example:
    python tools/generate_prototypes.py \
        --model runs/train/weights/best.pt \
        --data data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --output prototypes/task_1_prototypes.pt \
        --device 0 \
        --vis_dir prototypes/visualizations
"""

import argparse
import os
from pathlib import Path
import cv2

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM, YAML
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.metrics import bbox_iou, batch_probiou
from ultralytics.nn.modules.head import Detect, OBB


def _unpack_head_output(detect, captured_outputs):
    """Unpack hook output for Detect or OBB head.
    Returns (decoded_preds, raw_preds) for Detect; (decoded_preds_full, raw_preds) for OBB.
    OBB decoded_preds_full is (B, 4+nc+1, num_anchors) so bbox (first 4) and angle (last 1) can be used for rotated IoU.
    """
    if isinstance(detect, OBB):
        decoded_cat, (raw_preds, _) = captured_outputs  # decoded_cat (B, 4+nc+ne, sum(HW))
        return decoded_cat, raw_preds
    decoded, raw_preds = captured_outputs[0], captured_outputs[1]
    return decoded, raw_preds


def run_kmeans(features, k, max_iters=100):
    """
    Select k representative prototypes using K-Means clustering (Using consine distance).
    
    Args:
        features (torch.Tensor): Feature matrix [n, dim].
        k (int): Number of prototypes to select.
        max_iters (int): Maximum iterations. Default: 100.
    
    Returns:
        torch.Tensor: Indices of k prototypes closest to cluster centroids [k].
    """
    # Normalize for cosine distance-based clustering
    features = F.normalize(features, dim=1)

    n, dim = features.shape
    if n <= k:
        return torch.arange(n, device=features.device)

    # Random initialization
    centers = features[torch.randperm(n, device=features.device)[:k]]
    
    # K-Means iteration
    for _ in range(max_iters):
        # E-step: assign to nearest center
        dists = torch.cdist(features, centers)
        labels = dists.argmin(dim=1)
        
        # M-step: update centers
        new_centers = torch.stack([
            features[labels == i].mean(0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])
        
        # Check convergence
        if torch.norm(new_centers - centers) < 1e-4:
            break
        centers = new_centers

    # Return actual data points closest to centroids
    dists_to_centers = torch.cdist(features, centers)
    return dists_to_centers.argmin(dim=0)


def select_density_aware_groups(features, num_prototypes=10, radius=0.6):
    """Select one coarse group and density-aware fine-grained groups.

    Args:
        features (torch.Tensor): Feature matrix shaped (N, D).
        num_prototypes (int): Total prototypes, including one coarse prototype.
        radius (float): Cosine-similarity radius used to define hyperspheres.

    Returns:
        list[torch.Tensor]: Boolean membership masks shaped (N,).
    """
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty features shaped (N, D), got {tuple(features.shape)}")
    if num_prototypes < 1:
        raise ValueError(f"num_prototypes must be positive, got {num_prototypes}")
    if not -1.0 <= radius <= 1.0:
        raise ValueError(f"radius must be in [-1, 1], got {radius}")

    normalized = F.normalize(features.float(), dim=1)
    similarity = normalized @ normalized.T  # (N, N)
    neighborhoods = similarity >= radius
    densities, density_order = neighborhoods.sum(dim=1).sort(descending=True)

    groups = [torch.ones(features.shape[0], device=features.device, dtype=torch.bool)]
    low_density_index = max((features.shape[0] + 2) // 3, 1)
    density_threshold = densities[-low_density_index]
    excluded_centers = neighborhoods.sum(dim=1) <= density_threshold
    for center_idx in density_order:
        if excluded_centers[center_idx]:
            continue
        groups.append(neighborhoods[center_idx])
        excluded_centers.logical_or_(neighborhoods[center_idx])
        if len(groups) == num_prototypes:
            break
    return groups


def extract_pos_patches_from_layer(
    feat, bbox_map_px, cls_map, reg_map, gt_bbox_px, gt_cls,
    conf_thresh=0.25, iou_threshold=0.5, angle_map=None, gt_bbox_px_5=None
):
    """
    Extracts 5x5 feature patches from a single layer that match the GT.
    For OBB: pass angle_map (B, 1, H, W) and gt_bbox_px_5 (num_gt, 5) xywhr for rotated IoU.
    Returns list of: (y, x, patch, reg, cls, pad_mask, iou, gt_idx)
    """
    B, C, H, W = feat.shape
    feat_padded = F.pad(feat, (2, 2, 2, 2))
    patches = feat_padded.unfold(2, 5, 1).unfold(3, 5, 1)
    patches_flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(H * W, -1)

    unpadded_mask = F.pad(torch.ones([feat.shape[0], 1, *feat.shape[2:4]], dtype=torch.float32, device=feat_padded.device), (2, 2, 2, 2))
    unpadded_mask = unpadded_mask.unfold(2, 5, 1).unfold(3, 5, 1)
    unpadded_mask = unpadded_mask.permute(0, 2, 3, 1, 4, 5).reshape(H * W, -1)

    bbox_pred_px = bbox_map_px.permute(0, 2, 3, 1).reshape(H * W, 4)  # xywh [H*W, 4]
    cls_flat = cls_map.permute(0, 2, 3, 1).reshape(H * W, -1)
    reg_flat = reg_map.permute(0, 2, 3, 1).reshape(H * W, -1)

    valid_mask = (bbox_pred_px[:, 2] > 0) & (bbox_pred_px[:, 3] > 0)

    # IoU with GT: use rotated IoU (batch_probiou) when angle_map and gt_bbox_px_5 are provided (OBB)
    if angle_map is not None and gt_bbox_px_5 is not None:
        angle_flat = angle_map.permute(0, 2, 3, 1).reshape(H * W, 1)  # [H*W, 1]
        pred_xywhr = torch.cat([bbox_pred_px, angle_flat], dim=1)  # [H*W, 5]
        ious = batch_probiou(gt_bbox_px_5, pred_xywhr)  # [num_gt, H*W]
        ious = torch.nan_to_num(ious).clamp(0, 1)
    else:
        ious = bbox_iou(gt_bbox_px.unsqueeze(1), bbox_pred_px.unsqueeze(0), xywh=True)
        ious = torch.nan_to_num(ious).clamp(0, 1).squeeze(-1)  # [num_gt, H*W]
    
    # Filter 3: Correct Classification
    conf, cls_preds = cls_flat.sigmoid().max(dim=1)  # [H*W], [H*W]
    # gt_cls: [num_gt]
    correct_cls_mask = (cls_preds.unsqueeze(0) == gt_cls.unsqueeze(1)) & (conf.unsqueeze(0) > conf_thresh) & valid_mask.unsqueeze(0)

    # Select best candidate
    masked_ious = ious.clone()
    masked_ious[~correct_cls_mask] = -1.0
    
    best_idx = masked_ious.argmax(dim=1) # [num_gt]
    num_gt = best_idx.shape[0]
    row_indices = torch.arange(num_gt, device=best_idx.device)
    max_iou = masked_ious[row_indices, best_idx] # [num_gt]
    
    # Filter by iou_threshold: only keep matches with IOU >= threshold
    valid_iou_mask = max_iou >= iou_threshold
    
    # If no valid matches, return empty tensors
    if not valid_iou_mask.any():
        return (
            torch.tensor([], dtype=torch.long, device=feat.device),  # y
            torch.tensor([], dtype=torch.long, device=feat.device),  # x
            torch.empty((0, patches_flat.shape[1]), device=feat.device),  # patches
            torch.empty((0, reg_flat.shape[1]), device=feat.device),  # reg
            torch.empty((0, cls_flat.shape[1]), device=feat.device),  # cls
            torch.empty((0, unpadded_mask.shape[1]), device=feat.device),  # pad_mask
            torch.tensor([], dtype=torch.float32, device=feat.device),  # iou
            torch.tensor([], dtype=torch.long, device=feat.device)  # gt_idx
        )
    
    # Filter all outputs by valid_iou_mask
    y, x = best_idx // W, best_idx % W
    gt_indices = torch.arange(num_gt, device=feat.device)[valid_iou_mask]  # Original GT indices
    return (
        y[valid_iou_mask],  # [num_valid]
        x[valid_iou_mask],  # [num_valid]
        patches_flat[best_idx[valid_iou_mask]], # [num_valid, C*25]
        reg_flat[best_idx[valid_iou_mask]], # [num_valid, 4*reg_max]
        cls_flat[best_idx[valid_iou_mask]], # [num_valid, num_cls]
        unpadded_mask[best_idx[valid_iou_mask]], # [num_valid, 25]
        max_iou[valid_iou_mask], # [num_valid]
        gt_indices  # [num_valid] - indices to original GT
    )


def extract_neg_patches_from_layer(feat, cls_map, max_num, neg_conf_threshold=0.25):
    """
    Extracts 5x5 feature patches from a single layer where all class confidences are below threshold.
    Randomly selects patches that satisfy the condition.
    
    Args:
        feat: Feature map [B, C, H, W]
        cls_map: Classification logits [B, num_cls, H, W]
        max_num: Maximum number of patches to return
        neg_conf_threshold: Threshold for negative samples - all class confidences must be below this
    
    Returns:
        Tuple of (y, x, patch, pad_mask):
        - y: y coordinates [num_selected]
        - x: x coordinates [num_selected]
        - patch: Feature patches [num_selected, C*25]
        - pad_mask: Unpadded region mask [num_selected, 25]
    """
    B, C, H, W = feat.shape
    # Pad and unfold to get 5x5 patches: (1, C, H, W, 25) -> (H*W, C*25)
    feat_padded = F.pad(feat, (2, 2, 2, 2))
    patches = feat_padded.unfold(2, 5, 1).unfold(3, 5, 1)
    patches_flat = patches.permute(0, 2, 3, 1, 4, 5).reshape(H * W, -1) # [H*W, C*25]

    # Create unpadded region mask
    unpadded_mask = F.pad(torch.ones([feat.shape[0], 1, *feat.shape[2:4]], dtype=torch.float32, device=feat_padded.device), (2, 2, 2, 2))
    unpadded_mask = unpadded_mask.unfold(2, 5, 1).unfold(3, 5, 1)
    unpadded_mask = unpadded_mask.permute(0, 2, 3, 1, 4, 5).reshape(H * W, -1) # [H*W, 25]

    # Flatten classification map
    cls_flat = cls_map.permute(0, 2, 3, 1).reshape(H * W, -1) # [H*W, num_cls]
    
    # Filter: All class confidences must be below threshold
    cls_conf = cls_flat.sigmoid()  # [H*W, num_cls]
    # Check that all class confidences are below threshold
    valid_neg_mask = (cls_conf < neg_conf_threshold).all(dim=1)  # [H*W]
    
    # If no valid negative patches, return empty tensors
    if not valid_neg_mask.any():
        return (
            torch.tensor([], dtype=torch.long, device=feat.device),  # y
            torch.tensor([], dtype=torch.long, device=feat.device),  # x
            torch.empty((0, patches_flat.shape[1]), device=feat.device),  # patches
            torch.empty((0, unpadded_mask.shape[1]), device=feat.device)  # pad_mask
        )
    
    # Get indices of valid negative patches
    valid_indices = torch.where(valid_neg_mask)[0]  # [num_valid]
    
    # Randomly select patches (up to max_num if specified)
    num_valid = valid_indices.shape[0]
    if max_num is not None and num_valid > max_num:
        # Randomly sample max_num indices
        perm = torch.randperm(num_valid, device=feat.device)
        selected_indices = valid_indices[perm[:max_num]]
    else:
        selected_indices = valid_indices
    
    # Convert flat indices to (y, x) coordinates
    y = selected_indices // W
    x = selected_indices % W
    
    return (
        y,  # [num_selected]
        x,  # [num_selected]
        patches_flat[selected_indices],  # [num_selected, C*25]
        unpadded_mask[selected_indices]  # [num_selected, 25]
    )


def filter_old_neg_protos(p_old, meta_old, detect, layer_idx, neg_conf_threshold=0.25):
    """
    Filter old negative prototypes by checking if they are still negative samples
    under the new model parameters (with expanded class channels).
    
    Args:
        p_old: Old negative prototypes tensor [num_protos, C*25 + nc_old + 25]
               Format: [feat(C*25) | cls_valid_mask(nc_old) | pad_mask(25)]
        meta_old: List of meta information for old prototypes
        detect: Detection head module
        layer_idx: Layer index
        neg_conf_threshold: Confidence threshold for negative samples
    
    Returns:
        Tuple of (filtered_protos, filtered_meta):
        - filtered_protos: Filtered prototypes tensor [num_valid, C*25 + nc_new + 25]
        - filtered_meta: Filtered meta information list
    """
    if p_old is None or p_old.numel() == 0:
        return p_old, meta_old if meta_old else []
    
    device = detect.cv2[layer_idx][0].conv.weight.device
    p_old = p_old.to(device)
    
    num_protos = p_old.shape[0]
    in_channels = detect.cv2[layer_idx][0].conv.in_channels
    feat_dim = in_channels * 25
    nc_new = detect.nc
    
    # Parse old prototypes: [feat(C*25) | cls_valid_mask(nc_old) | pad_mask(25)]
    # We need to extract feat and pad_mask, ignore the old cls_valid_mask
    # The old format might have different nc, so we need to handle it
    # Assume the last 25 elements are pad_mask, and before that is cls_valid_mask
    pad_mask_flat = p_old[:, -25:]  # [num_protos, 25]
    # The feat is the first feat_dim elements
    feat_flat = p_old[:, :feat_dim]  # [num_protos, C*25]
    
    # Reshape features to [num_protos, C, 5, 5]
    feat_5x5 = feat_flat.reshape(num_protos, in_channels, 5, 5)
    
    # Reshape pad_mask to [num_protos, 5, 5]
    pad_mask_5x5 = pad_mask_flat.reshape(num_protos, 5, 5)
    
    # Restore prototypes from padded format to 5x5 feature maps
    restored_prototypes = torch.zeros([num_protos, in_channels, 5, 5], device=device)
    
    for k in range(num_protos):
        mask = pad_mask_5x5[k]  # [5, 5], 1=original, 0=padded
        proto = feat_5x5[k]  # [in_channels, 5, 5]
        
        # Find the valid region (original region) bounds in pad_mask
        valid_rows = torch.where(mask.sum(dim=1) > 0)[0]
        valid_cols = torch.where(mask.sum(dim=0) > 0)[0]
        
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            # Get valid region bounds in pad_mask coordinates
            mask_h_start, mask_h_end = valid_rows[0].item(), valid_rows[-1].item() + 1
            mask_w_start, mask_w_end = valid_cols[0].item(), valid_cols[-1].item() + 1
            
            # Extract valid region from prototype (this is the original feature map region)
            valid_proto = proto[:, mask_h_start:mask_h_end, mask_w_start:mask_w_end]  # [in_channels, H', W']
            
            # Calculate restore offsets
            offset_y = (0 - mask_h_start) + (5 - mask_h_end)
            offset_x = (0 - mask_w_start) + (5 - mask_w_end)
            
            # Place valid region at the offset position in restored feature map
            restored_h_start = mask_h_start + offset_y
            restored_w_start = mask_w_start + offset_x
            restored_h_end = mask_h_end + offset_y
            restored_w_end = mask_w_end + offset_x
            
            # Place the valid region at the corresponding position
            restored_prototypes[k, :, restored_h_start:restored_h_end, restored_w_start:restored_w_end] = valid_proto
    
    # Forward through classification head to get class predictions
    cls_output = detect.cv3[layer_idx](restored_prototypes)  # [num_protos, num_cls, 5, 5]
    
    # Average over spatial dimensions to get per-prototype class confidences
    cls_conf = cls_output.sigmoid().mean(dim=(2, 3))  # [num_protos, num_cls]
    
    # Check if all class confidences are below threshold
    all_below_threshold = (cls_conf < neg_conf_threshold).all(dim=1)  # [num_protos]
    
    # Filter prototypes that are still negative
    valid_indices = torch.where(all_below_threshold)[0]
    
    if len(valid_indices) == 0:
        # No valid negative prototypes, return empty tensor and empty meta
        return torch.empty((0, feat_dim + nc_new + 25), device=device).cpu(), []
    
    # Get filtered prototypes
    filtered_feat = feat_flat[valid_indices]  # [num_valid, C*25]
    filtered_pad_mask = pad_mask_flat[valid_indices]  # [num_valid, 25]
    
    # Create new cls_valid_mask (all ones for negative samples)
    cls_valid_mask = torch.ones([len(valid_indices), nc_new], device=device)
    
    # Combine: [feat(C*25) | cls_valid_mask(nc_new) | pad_mask(25)]
    filtered_protos = torch.cat([filtered_feat, cls_valid_mask, filtered_pad_mask], dim=1)
    
    # Filter meta information
    valid_indices_list = valid_indices.cpu().tolist()
    filtered_meta = [meta_old[i] for i in valid_indices_list] if meta_old else []
    
    return filtered_protos.cpu(), filtered_meta


def visualize_results(prototypes, meta_info, detect, vis_dir, class_names, imgsz=640):
    """Visualizes selected prototypes on original images."""
    if not prototypes or all(p is None for p in prototypes):
        LOGGER.warning("No prototypes to visualize.")
        return

    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Group annotations by image path to minimize I/O
    # Structure: img_path -> list of (bbox, label, color_index)
    draw_queue = {}
    nc = len(class_names)
    assert nc==detect.nc
    reg_dim = 4*detect.reg_max
    device = detect.cv2[0][0].conv.weight.device

    for layer_idx, (proto_tensor, metas) in enumerate(zip(prototypes, meta_info)):
        if proto_tensor is None:
            continue
        # Parse Tensor: [ feat(C*25) | reg(4*reg_max) | Cls(nc) | Mask(25) ]
        feat_dim = detect.cv2[layer_idx][0].conv.in_channels * 25
        cls_preds = proto_tensor[:, feat_dim+reg_dim:feat_dim+reg_dim+nc]
        reg_cls_preds = proto_tensor[:, feat_dim:feat_dim+reg_dim+nc]
        top_conf, top_cls = cls_preds.sigmoid().max(dim=1)
        
        for reg_cls, conf, cls, meta in zip(reg_cls_preds, top_conf, top_cls, metas):
            img_path = meta['img_path']
            if img_path not in draw_queue:
                draw_queue[img_path] = []
            
            # Retrieve BBox (in model input scale, usually 640px)
            # Ensure it's a flat list/array [x, y, w, h]
            HWs = meta['HWs']
            retrieved_pred_maps = [torch.zeros([1, reg_dim+nc, H, W], device=device) for H, W in HWs]
            y_idx, x_idx = meta["yx"]
            retrieved_pred_maps[layer_idx][0, :, y_idx, x_idx] = reg_cls.to(device)
            retrieved_decoded_bboxes = detect._inference(retrieved_pred_maps)
            retrieval_idx = sum([H*W for H, W in HWs[:layer_idx]]) + y_idx * HWs[layer_idx][1] + x_idx
            bbox = retrieved_decoded_bboxes[0,:4,retrieval_idx]

            # Convert xywh to xyxy for plotting
            if not detect.xyxy:
                x, y, w, h = bbox
                xyxy = torch.tensor([x - w/2, y - h/2, x + w/2, y + h/2])
            else:
                xyxy = bbox

            # Convert to unnormalized
            if xyxy.max() < 2:
                xyxy = xyxy * imgsz

            # Conver to np
            if isinstance(xyxy, torch.Tensor):
                xyxy = xyxy.cpu().numpy()
            
            cls = cls.item()
            conf = conf.item()
            name = class_names[cls]
            label = f"{name} {conf:.2f} L{layer_idx}"
            
            draw_queue[img_path].append((xyxy, label, cls))

    # 2. Draw and Save
    LOGGER.info(f"Visualizing prototypes on {len(draw_queue)} images...")
    for img_path, annotations in TQDM(draw_queue.items(), desc="Visualizing"):
        if not os.path.exists(img_path):
            LOGGER.warning(f"Image {img_path} doesn't exist, skipped for visualization")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            LOGGER.warning(f"Failed to read {img_path}, skipped for visualization")
            continue
        
        # Calculate scale factor (Original Image / Model Input)
        h, w = img.shape[:2]
        scale_x, scale_y = w / imgsz, h / imgsz
        
        annotator = Annotator(img, line_width=2, example=str(class_names))
        
        for xyxy, label, cls in annotations:
            # Scale coordinates back to original image size
            box = [
                xyxy[0] * scale_x, xyxy[1] * scale_y,
                xyxy[2] * scale_x, xyxy[3] * scale_y
            ]
            annotator.box_label(box, label, color=colors(cls, True))
            
        cv2.imwrite(str(os.path.join(vis_dir, os.path.basename(img_path))), annotator.result())


def generate_prototypes(args):
    # 1. Setup Model
    if str.isdigit(args.device):
        args.device = f"cuda:{args.device}"
    device = torch.device(args.device)
    model = YOLO(args.model).to(args.device)
    detect = model.model.model[-1]
    model.to(device).eval().fuse()
    is_xyxy = detect.xyxy
    names = model.model.names
    class_names_model = [names[i] for i in sorted(names.keys())]
    
    # 2. Hooks
    captured_inputs = []
    captured_outputs = []
    def input_hook(m, x): captured_inputs[:] = [t.detach() for t in x[0]]
    def output_hook(m, x, y): captured_outputs[:] = y
    detect.register_forward_pre_hook(input_hook)
    detect.register_forward_hook(output_hook)

    # 3. Setup Data
    data_cfg = YAML.load(args.data)
    class_names_dataset = data_cfg['names']
    class_id_map = {}
    for key, name in class_names_dataset.items():
        exist_flag = False
        for key_model, name_model in enumerate(class_names_model):
            if name==name_model:
                exist_flag = True
                class_id_map[key] = key_model
                break
        if not exist_flag:
            LOGGER.warning(f"Class {name} doesn't exist in model's class list, skipped")
    dataset_name = Path(data_cfg['path'] if 'path' in data_cfg else os.path.dirname(args.data)).stem
    img_dir = data_cfg['train'] if os.path.exists(data_cfg['train']) else \
        os.path.join(data_cfg['path'], data_cfg['train'])  if 'path' in data_cfg else \
            os.path.join(os.path.dirname(args.data), data_cfg["train"])
    lbl_dir = str(img_dir).replace('images', 'labels')    
    img_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
    img_extensions.extend([ext.upper() for ext in img_extensions])
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(tuple(img_extensions))])

    # 4. Iter through dataset to collect raw features
    collector_pos = [[[] for _ in range(detect.nc)] for _ in range(detect.nl)]
    collector_neg = [[] for _ in range(detect.nl)]
    pbar = TQDM(img_files, desc="Generating prototypes")
    for img_file in pbar:
        # Load Image
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, Path(img_file).stem + '.txt')
        if not os.path.exists(lbl_path):
            LOGGER.warning(f"Label file for image {img_file} not found, skipped")
            continue
        
        # Prepare Input
        img0 = cv2.imread(img_path)
        if img0 is None:
            LOGGER.warning(f"Failed to read image {img_path}, skipped")
            continue
        img = cv2.resize(img0, (args.imgsz, args.imgsz))
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

        # Inference (triggers hooks)
        model.model(img_tensor)
        
        # Parse Outputs (support both Detect and OBB head)
        decoded_preds, raw_preds = _unpack_head_output(detect, captured_outputs)
        is_obb = isinstance(detect, OBB)
        # Detect: decoded_preds (B, 4+nc, sum(HW)); OBB: (B, 4+nc+1, sum(HW)) with angle in last channel
        bbox_maps_px = []
        angle_maps = []  # OBB only: list of (B, 1, H, W) per layer
        start = 0
        HWs = []
        for lid, raw in enumerate(raw_preds):
            B, _, H, W = raw.shape
            bbox_layer = decoded_preds[:, :4, start:start + H * W].view(B, 4, H, W)
            if is_xyxy:
                bbox_layer_xyxy = bbox_layer.clone()
                bbox_layer[:, [0, 1], :] = (bbox_layer_xyxy[:, [0, 1], :] + bbox_layer_xyxy[:, [2, 3], :]) / 2
                bbox_layer[:, [2, 3], :] = (-bbox_layer_xyxy[:, [0, 1], :] + bbox_layer_xyxy[:, [2, 3], :]) / 2
            if bbox_layer.max() < 2:
                bbox_layer *= args.imgsz
            bbox_maps_px.append(bbox_layer)
            if is_obb:
                angle_layer = decoded_preds[:, -1:, start:start + H * W].view(B, 1, H, W)
                angle_maps.append(angle_layer)
            start += H * W
            HWs.append((H, W))

        # Parse GT
        with open(lbl_path) as f:
            gt_lines = [x.split() for x in f.readlines()]
        
        if not gt_lines:
            LOGGER.warning(f"No ground truth line in {lbl_path}, skipped")
            continue
        
        # Use the class map dict to map class id to model's output order
        # OBB labels: class_id + (cx,cy,w,h,angle) xywhr or class_id + 8 corners; detection: class_id + xywh
        gt_cls_batch = []
        gt_bbox_norm = []
        for x in gt_lines:
            if int(x[0]) not in class_id_map.keys():
                continue
            vals = [float(v) for v in x[1:]]
            if len(vals) >= 4:
                if len(vals) >= 8:
                    xs, ys = vals[0::2], vals[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    w, h = x_max - x_min, y_max - y_min
                    gt_bbox_norm.append([cx, cy, w, h])
                elif is_obb and len(vals) >= 5:
                    gt_bbox_norm.append(vals[:5])  # xywhr for OBB
                else:
                    gt_bbox_norm.append(vals[:4])
                gt_cls_batch.append(class_id_map[int(x[0])])
        if not gt_cls_batch:
            continue
        gt_cls_batch = torch.tensor(gt_cls_batch, device=device)
        gt_bbox_norm = torch.tensor(gt_bbox_norm, device=device)
        gt_bbox_px = gt_bbox_norm.clone()
        if is_obb and gt_bbox_px.shape[1] == 5:
            gt_bbox_px[:, :4] *= args.imgsz  # scale xywh only; angle (rad) unchanged
        else:
            gt_bbox_px *= args.imgsz

        # Collect Prototypes
        for layer_idx, (feat, raw_pred, bbox_map_px) in enumerate(zip(captured_inputs, raw_preds, bbox_maps_px)):
            reg_map = raw_pred[:, :detect.reg_max * 4]
            cls_map = raw_pred[:, detect.reg_max * 4:]
            angle_map = angle_maps[layer_idx] if is_obb else None
            gt_bbox_px_5 = gt_bbox_px if (is_obb and gt_bbox_px.shape[1] == 5) else None
            pos_result = extract_pos_patches_from_layer(
                feat, bbox_map_px, cls_map, reg_map,
                gt_bbox_px[:, :4] if gt_bbox_px_5 is not None else gt_bbox_px,
                gt_cls_batch,
                args.pos_conf_threshold, args.iou_threshold,
                angle_map=angle_map,
                gt_bbox_px_5=gt_bbox_px_5,
            )

            neg_result = extract_neg_patches_from_layer(
                feat, cls_map,
                max(args.num_protos//len(img_files), 1), args.neg_conf_threshold
            )

            for y, x, patch, reg, cls, pad_mask, iou, gt_idx in zip(*pos_result):
                meta = {
                    "img_path": img_path, "yx": (y, x), "HWs": HWs,
                    "bbox_gt_norm": gt_bbox_norm[gt_idx].cpu(),
                    "dataset": dataset_name
                }
                collector_pos[layer_idx][gt_cls_batch[gt_idx].item()].append(
                    (patch, reg, cls, pad_mask, meta)
                )
            
            for y, x, patch, pad_mask in zip(*neg_result):
                meta = {
                    "img_path": img_path, "yx": (y, x), "HWs": HWs,
                    "dataset": dataset_name
                }
                cls_valid_mask = torch.ones(detect.nc).to(device)
                collector_neg[layer_idx].append(
                    (patch, cls_valid_mask, pad_mask, meta)
                )


    # 5. Check prototypes for each class
    for cls_idx in range(detect.nc):
        num_cls_protos = 0
        for layer_idx in range(detect.nl):
            num_cls_protos += len(collector_pos[layer_idx][cls_idx])
        if num_cls_protos == 0 and cls_idx in class_id_map.values():
            LOGGER.warning(f"No prototypes of {class_names_model[cls_idx]} collected.")

    # 6. Clustering & Saving
    final_protos_pos = [None] * detect.nl
    final_metas_pos = [[] for _ in range(detect.nl)]
    final_protos_neg = [None] * detect.nl
    final_metas_neg = [[] for _ in range(detect.nl)]
    repre_levels = []
    
    LOGGER.info("Clustering prototypes...")
    for layer_idx in range(detect.nl):
        # Collect and clustering positive protos
        layer_tensors = []
        repre_features = []
        repre_masks = []
        repre_labels = []
        
        for cls_idx in range(detect.nc):
            items = collector_pos[layer_idx][cls_idx]
            if not items:
                continue
            
            # Unzip
            patches, regs, clss, masks, metas = zip(*items)
            patch_tensor = torch.stack(patches)
            reg_tensor = torch.stack(regs)
            cls_tensor = torch.stack(clss)
            mask_tensor = torch.stack(masks)

            if args.selection == "density":
                groups = select_density_aware_groups(
                    patch_tensor,
                    num_prototypes=args.num_protos,
                    radius=args.radius,
                )
                sel_patches = torch.stack([patch_tensor[group].mean(dim=0) for group in groups])
                sel_regs = torch.stack([reg_tensor[group].mean(dim=0) for group in groups])
                sel_clss = torch.stack([cls_tensor[group].mean(dim=0) for group in groups])
                sel_masks = torch.stack([mask_tensor[group].mean(dim=0) for group in groups])
                sel_metas = [metas[0]] * len(groups)
            else:
                indices = run_kmeans(patch_tensor, k=args.num_protos)
                sel_patches = patch_tensor[indices]
                sel_regs = reg_tensor[indices]
                sel_clss = cls_tensor[indices]
                sel_masks = mask_tensor[indices]
                sel_metas = [metas[i] for i in indices]
            
            # Combine: [Feat(25C) | Reg | Cls | Mask(25)]
            combined = torch.cat([sel_patches, sel_regs, sel_clss, sel_masks], dim=1)
            layer_tensors.append(combined)
            final_metas_pos[layer_idx].extend(sel_metas)
            repre_features.append(sel_patches)
            repre_masks.append(sel_masks)
            repre_labels.append(torch.full((sel_patches.shape[0],), cls_idx, device=sel_patches.device))

        if layer_tensors:
            final_protos_pos[layer_idx] = torch.cat(layer_tensors, dim=0).cpu()
            channels = captured_inputs[layer_idx].shape[1]
            repre_levels.append(
                {
                    "features": torch.cat(repre_features).reshape(-1, channels, 5, 5).cpu(),
                    "valid_masks": torch.cat(repre_masks).reshape(-1, 5, 5).cpu(),
                    "labels": torch.cat(repre_labels).long().cpu(),
                }
            )
        else:
            LOGGER.warning(f"No prototypes collected from layer {layer_idx}")
            channels = captured_inputs[layer_idx].shape[1]
            repre_levels.append(
                {
                    "features": torch.empty((0, channels, 5, 5)),
                    "valid_masks": torch.empty((0, 5, 5)),
                    "labels": torch.empty((0,), dtype=torch.long),
                }
            )

        # Collect and randomly sample negative protos
        num_neg_items = len(collector_neg[layer_idx])
        if num_neg_items > 0:
            # Randomly sample num_protos negative prototypes
            if num_neg_items > args.num_protos:
                # Randomly select indices
                selected_indices = torch.randperm(num_neg_items, device=torch.device('cpu'))[:args.num_protos].tolist()
            else:
                # Use all available negative prototypes
                selected_indices = list(range(num_neg_items))
            
            selected_items = [collector_neg[layer_idx][idx] for idx in selected_indices]
            final_protos_neg[layer_idx] = torch.stack([torch.cat(item[:3], dim=0) for item in selected_items]).cpu()
            final_metas_neg[layer_idx] = [item[-1] for item in selected_items]

    # 7. Merge History
    if args.load_hist:
        hist = torch.load(args.load_hist, map_location='cpu')
        if args.selection == "density" and "repre" not in hist:
            raise KeyError(f"Historical prototype artifact '{args.load_hist}' has no RePRE data")
        for lid in range(detect.nl):
            if args.selection != "density" and hist['prototypes'][lid] is not None:
                p_new = final_protos_pos[lid]
                p_old = hist['prototypes'][lid]
                final_protos_pos[lid] = torch.cat([p_old, p_new]) if p_new is not None else p_old
                final_metas_pos[lid] = hist['meta_info'][lid] + final_metas_pos[lid]
            if args.selection != "density" and hist['prototypes_neg'][lid] is not None:
                p_new = final_protos_neg[lid]
                p_old = hist['prototypes_neg'][lid]
                meta_old = hist['meta_info_neg'][lid]
                p_old, meta_old = filter_old_neg_protos(p_old, meta_old, detect, lid, args.neg_conf_threshold)
                final_protos_neg[lid] = torch.cat([p_old, p_new]) if p_new is not None else p_old
                final_metas_neg[lid] = meta_old + final_metas_neg[lid]
            if "repre" in hist:
                for key in ("features", "valid_masks", "labels"):
                    repre_levels[lid][key] = torch.cat((hist["repre"][lid][key], repre_levels[lid][key]))

    torch.save(
        {
            "prototypes": final_protos_pos,
            "meta_info": final_metas_pos,
            "prototypes_neg": final_protos_neg,
            "meta_info_neg": final_metas_neg,
            "repre": repre_levels,
        },
        args.output,
    )
    LOGGER.info(f"Saved to {args.output}")

    # 7. Visualize prototypes
    if args.vis_dir:
        visualize_results(
            final_protos_pos, 
            final_metas_pos,
            detect,
            args.vis_dir,
            class_names_model,
            args.imgsz
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num_protos", type=int, default=10)
    parser.add_argument("--selection", choices=["kmeans", "density"], default="kmeans")
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--pos_conf_threshold", type=float, default=0.1)
    parser.add_argument("--neg_conf_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--load_hist", default=None)
    args = parser.parse_args()
    
    generate_prototypes(args)