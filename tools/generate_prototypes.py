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
6. Using K-means clustering to select representative prototypes (k_center=10 per class)
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
        [--vis_dir <path_to_vis_dir>]

Arguments:
    --model: Path to trained YOLO model (.pt file) [required]
    --data: Path to dataset YAML configuration file [required]
    --output: Path to save generated prototypes (.pt file) [required]
    --device: Device to use (e.g., '0' for GPU 0, 'cpu' for CPU) [default: '0']
    --imgsz: Image size for inference [default: 640]
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

from numpy.core.defchararray import isdigit
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM, YAML
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.metrics import bbox_iou


def run_kmeans(features, k, max_iters=100):
    """Select k representative prototypes using K-Means clustering."""
    n, dim = features.shape
    if n <= k:
        return torch.arange(n, device=features.device)

    # Initialize
    centers = features[torch.randperm(n, device=features.device)[:k]]
    
    for _ in range(max_iters):
        # E-step: Assign points to nearest center
        dists = torch.cdist(features, centers)
        labels = dists.argmin(dim=1)
        
        # M-step: Update centers
        new_centers = torch.stack([
            features[labels == i].mean(0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])
        
        if torch.norm(new_centers - centers) < 1e-4:
            break
        centers = new_centers

    # Find actual data points closest to centroids
    dists_to_centers = torch.cdist(features, centers)
    return dists_to_centers.argmin(dim=0)


def extract_patches_from_layer(feat, bbox_map_px, cls_map, reg_map, gt_bbox_px, gt_cls):
    """
    Extracts 5x5 feature patches from a single layer that match the GT.
    Returns list of: (y, x, patch, reg, cls, pred_bbox, iou)
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

    # Flatten maps for vectorized operations
    bbox_pred_px = bbox_map_px.permute(0, 2, 3, 1).reshape(H * W, 4) # xywh, [H*W, 4]
    cls_flat = cls_map.permute(0, 2, 3, 1).reshape(H * W, -1) # [H*W, num_cls]
    reg_flat = reg_map.permute(0, 2, 3, 1).reshape(H * W, -1) # [H*W, 4*reg_max]
    
    # Filter 1: Valid dimensions
    valid_mask = (bbox_pred_px[:, 2] > 0) & (bbox_pred_px[:, 3] > 0) # [H*W]
    
    # Filter 2: IOU with GT
    # gt_bbox_px: [num_gt, 4]
    ious = bbox_iou(gt_bbox_px.unsqueeze(1), bbox_pred_px.unsqueeze(0), xywh=True)
    ious = torch.nan_to_num(ious).clamp(0, 1).squeeze(-1) # [num_gt, H*W]
    
    # Filter 3: Correct Classification
    cls_preds = cls_flat.argmax(dim=1) # [H*W]
    # gt_cls: [num_gt]
    correct_cls_mask = (cls_preds.unsqueeze(0) == gt_cls.unsqueeze(1)) & valid_mask.unsqueeze(0)

    # Select best candidate
    masked_ious = ious.clone()
    masked_ious[~correct_cls_mask] = -1.0
    
    best_idx = masked_ious.argmax(dim=1) # [num_gt]
    num_gt = best_idx.shape[0]
    row_indices = torch.arange(num_gt, device=best_idx.device)
    max_iou = masked_ious[row_indices, best_idx] # [num_gt]
    
    y, x = best_idx // W, best_idx % W
    return (
        y, x,
        patches_flat[best_idx], # [num_gt, C*25]
        reg_flat[best_idx], # [num_gt, 4*reg_max]
        cls_flat[best_idx], # [num_gt, num_cls]
        unpadded_mask[best_idx], # [num_gt, 25]
        max_iou # [num_gt]
    )


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
        top_conf, top_cls = cls_preds.softmax(dim=1).max(dim=1)
        
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
    
    # Hooks
    captured_inputs = []
    captured_outputs = []
    def input_hook(m, x): captured_inputs[:] = [t.detach() for t in x[0]]
    def output_hook(m, x, y): captured_outputs[:] = y
    detect.register_forward_pre_hook(input_hook)
    detect.register_forward_hook(output_hook)

    # 2. Setup Data
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
    img_dir = data_cfg['train'] if os.path.exists(data_cfg['train']) else \
        os.path.join(data_cfg['path'], data_cfg['train'])  if 'path' in data_cfg else \
            os.path.join(os.path.dirname(args.data), data_cfg["train"])
    lbl_dir = str(img_dir).replace('images', 'labels')    
    img_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
    img_extensions.extend([ext.upper() for ext in img_extensions])
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(tuple(img_extensions))])
    
    # Storage: layers -> classes -> list of prototypes
    # Use simple list for collection, process later
    collector = [[[] for _ in range(detect.nc)] for _ in range(detect.nl)]

    # 3. Processing each image
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
        
        # Parse Outputs
        # decoded_preds: (B, 4+nc, sum(HW))
        # raw_preds: list of (B, reg_max*4+nc, H, W)
        decoded_preds, raw_preds = captured_outputs
        
        # Split decoded bboxes back to layers
        bbox_maps_px = []
        start = 0
        HWs = []
        for i, raw in enumerate(raw_preds):
            B, _, H, W = raw.shape
            # decoded_preds contains xywh
            bbox_layer = decoded_preds[:, :4, start:start + H*W].view(B, 4, H, W)
            if is_xyxy: # All bbox should be converted to xywh format
                bbox_layer_xyxy = bbox_layer.clone()
                bbox_layer[:,[0,1],:] = ( bbox_layer_xyxy[:,[0,1],:]+bbox_layer_xyxy[:,[2,3],:]) / 2
                bbox_layer[:,[2,3],:] = (-bbox_layer_xyxy[:,[0,1],:]+bbox_layer_xyxy[:,[2,3],:]) / 2
            if bbox_layer.max() < 2: # All bbox should be unnormalized
                bbox_layer *= args.imgsz
            bbox_maps_px.append(bbox_layer)
            start += H * W
            HWs.append((H, W))

        # Parse GT
        with open(lbl_path) as f:
            gt_lines = [x.split() for x in f.readlines()]
        
        if not gt_lines:
            LOGGER.warning(f"No ground truth line in {lbl_path}, skipped")
            continue
        
        # Use the class map dict to map class id to model's output order
        gt_cls_batch = []
        gt_bbox_norm = []
        for x in gt_lines:
            if int(x[0]) in class_id_map.keys():
                gt_cls_batch.append(class_id_map[int(x[0])])
                gt_bbox_norm.append([float(v) for v in x[1:]])
        gt_cls_batch = torch.tensor(gt_cls_batch, device=device)
        gt_bbox_norm = torch.tensor(gt_bbox_norm, device=device)
        gt_bbox_px = gt_bbox_norm.clone()
        gt_bbox_px *= args.imgsz

        # Collect Prototypes
        for layer_idx, (feat, raw_pred, bbox_map_px) in enumerate(zip(captured_inputs, raw_preds, bbox_maps_px)):
            reg_map = raw_pred[:, :detect.reg_max * 4]
            cls_map = raw_pred[:, detect.reg_max * 4:]
            
            result = extract_patches_from_layer(
                feat, bbox_map_px, cls_map, reg_map, 
                gt_bbox_px, gt_cls_batch
            )

            for idx, (y, x, patch, reg, cls, pad_mask, iou) in enumerate(zip(*result)):
                if iou > args.iou_threshold:
                    meta = {
                        "img_path": img_path, "yx": (y, x), "HWs": HWs,
                        "bbox_gt_norm": gt_bbox_norm[idx].cpu()
                    }
                    collector[layer_idx][gt_cls_batch[idx].item()].append(
                        (patch, reg, cls, pad_mask, meta)
                    )

    # 4. Check prototypes for each class
    for cls_idx in range(detect.nc):
        num_cls_protos = 0
        for layer_idx in range(detect.nl):
            num_cls_protos += len(collector[layer_idx][cls_idx])
        if num_cls_protos == 0 and cls_idx in class_id_map.values():
            LOGGER.warning(f"No prototypes of {class_names_model[cls_idx]} collected.")

    # 5. Clustering & Saving
    final_protos = [None] * detect.nl
    final_metas = [[] for _ in range(detect.nl)]
    
    LOGGER.info("Clustering prototypes...")
    for layer_idx in range(detect.nl):
        layer_tensors = []
        
        for cls_idx in range(detect.nc):
            items = collector[layer_idx][cls_idx]
            if not items:
                continue
            
            # Unzip
            patches, regs, clss, masks, metas = zip(*items)
            patch_tensor = torch.stack(patches)
            
            # K-Means
            indices = run_kmeans(patch_tensor, k=10)
            
            # Select
            sel_patches = patch_tensor[indices]
            sel_regs = torch.stack(regs)[indices]
            sel_clss = torch.stack(clss)[indices]
            sel_masks = torch.stack(masks)[indices]
            sel_metas = [metas[i] for i in indices]
            
            # Combine: [Feat(25C) | Reg | Cls | Mask(25)]
            combined = torch.cat([sel_patches, sel_regs, sel_clss, sel_masks], dim=1)
            layer_tensors.append(combined)
            final_metas[layer_idx].extend(sel_metas)

        if layer_tensors:
            final_protos[layer_idx] = torch.cat(layer_tensors, dim=0).cpu()
        else:
            LOGGER.warning(f"No prototypes collected from layer {layer_idx}")

    # 6. Merge History
    if args.load_hist:
        hist = torch.load(args.load_hist, map_location='cpu')
        for i in range(detect.nl):
            if hist['prototypes'][i] is not None:
                p_new = final_protos[i]
                p_old = hist['prototypes'][i]
                final_protos[i] = torch.cat([p_old, p_new]) if p_new is not None else p_old
                final_metas[i] = hist['meta_info'][i] + final_metas[i]

    torch.save({"prototypes": final_protos, "meta_info": final_metas}, args.output)
    LOGGER.info(f"Saved to {args.output}")

    # 7. Visualize prototypes
    if args.vis_dir:
        visualize_results(
            final_protos, 
            final_metas,
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
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--load_hist", default=None)
    args = parser.parse_args()
    
    generate_prototypes(args)