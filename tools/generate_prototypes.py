#!/usr/bin/env python3
"""
Generate prototypes for Prototype Replay (PRoRP) mechanism.

This script generates prototypes by:
1. Loading a trained YOLO model and dataset
2. For each training image, performing forward pass to get detection head features
3. Selecting prototypes with maximum IOU for each ground truth bbox across all layers
4. Extracting 3x3 feature patches and corresponding regression/classification outputs
5. Organizing prototypes by layer and ground truth class
6. Using K-means clustering to select representative prototypes (k_center=10 per class)
7. Saving prototypes as a list of tensors (one per layer), where each tensor contains:
   [num_prototypes, in_channels*3*3 + reg_out_channels + cls_out_channels]

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
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM, YAML
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.plotting import Annotator, colors

from utils import RealTimeMemoryMonitor


def k_means(prototypes, k_center):
    """ Find cluster centers in prototypes using K-means clustering

    Args:
        prototypes: torch.Tensor, shape=[n_prototypes, feature_dim] - feature vectors to cluster
        k_center: int - number of cluster centers to find

    Returns:
        - center_idx: torch.Tensor, shape=[k_center], dtype=torch.long - indices of cluster centers
    """
    n_prototypes, feature_dim = prototypes.shape
    
    # If we have fewer prototypes than requested centers, return all indices
    if n_prototypes <= k_center:
        return torch.arange(n_prototypes, dtype=torch.long, device=prototypes.device)
    
    # Initialize cluster centers randomly
    indices = torch.randperm(n_prototypes, device=prototypes.device)[:k_center]
    centers = prototypes[indices].clone()
    
    max_iters = 100
    tol = 1e-4
    
    for iteration in range(max_iters):
        # Assign each point to nearest center
        distances = torch.cdist(prototypes, centers)  # (n_prototypes, k_center)
        labels = distances.argmin(dim=1)  # (n_prototypes,)
        
        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(k_center):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = prototypes[mask].mean(dim=0)
            else:
                # If cluster is empty, keep old center or reinitialize
                new_centers[k] = centers[k]
        
        # Check convergence
        center_shift = (new_centers - centers).norm(dim=1).max()
        if center_shift < tol:
            break
        
        centers = new_centers
    
    # Find the prototypes closest to each cluster center
    distances_to_centers = torch.cdist(prototypes, centers)  # (n_prototypes, k_center)
    center_idx = distances_to_centers.argmin(dim=0)  # (k_center,) - indices of prototypes closest to centers
    
    return center_idx


def extract_feature_patch(
    feature_map: torch.Tensor, 
    center_y: int, 
    center_x: int, 
    patch_size: int = 3
) -> torch.Tensor:
    """
    Extract a patch from feature map centered at (center_y, center_x).
    
    Args:
        feature_map: Feature map tensor of shape (C, H, W)
        center_y: Center y coordinate
        center_x: Center x coordinate
        patch_size: Size of the patch (default: 3 for 3x3)
    
    Returns:
        Flattened patch tensor of shape (C * patch_size * patch_size)
    """
    C, H, W = feature_map.shape
    half = patch_size // 2
    
    # Ensure coordinates are within bounds
    center_y = max(half, min(center_y, H - 1 - half))
    center_x = max(half, min(center_x, W - 1 - half))
    
    # Calculate patch boundaries
    y_start = center_y - half
    y_end = center_y + half + 1
    x_start = center_x - half
    x_end = center_x + half + 1
    
    # Extract patch - should always be (C, patch_size, patch_size) if coordinates are valid
    patch = feature_map[:, y_start:y_end, x_start:x_end]
    
    # Verify patch shape
    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
        # If patch is not the right size, pad it
        pad_y_before = max(0, half - center_y)
        pad_y_after = max(0, (center_y + half + 1) - H)
        pad_x_before = max(0, half - center_x)
        pad_x_after = max(0, (center_x + half + 1) - W)
        
        patch = nn.functional.pad(
            patch, 
            (pad_x_before, pad_x_after, pad_y_before, pad_y_after),
            mode='constant',
            value=0.0
        )
        
        # Ensure final shape is correct
        if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
            # If still wrong, create a zero patch of correct size
            patch = torch.zeros((C, patch_size, patch_size), device=feature_map.device, dtype=feature_map.dtype)
            # Copy available data
            actual_h = min(patch_size, patch.shape[1])
            actual_w = min(patch_size, patch.shape[2])
            patch[:, :actual_h, :actual_w] = feature_map[:, y_start:y_start+actual_h, x_start:x_start+actual_w]
    
    # Flatten: (C, patch_size, patch_size) -> (C * patch_size * patch_size)
    return patch.flatten()

def visualize_prototypes(
    prototypes,
    class_names,
    vis_dir
):
    """
    Visualize representative prototypes by drawing bboxes and class labels on images.
    
    For each prototype, the function:
    1. Loads the source image from metadata
    2. Extracts bbox coordinates (in xyxy format)
    3. Gets predicted class and confidence from classification output
    4. Draws bounding box and label on the image
    5. Saves the visualized image to vis_dir
    
    Args:
        prototypes: List[List[Tuple or None]], shape=[num_layers][num_classes]
                   Each element is a tuple of (selected_prototypes, selected_regs, 
                   selected_cls, selected_bboxes, selected_meta_info) or None.
                   Each component is a list of tensors/values for multiple prototypes.
        class_names: List[str] or None - class names for visualization labels.
                     If None, uses "class_{idx}" format.
        vis_dir: str - output directory path for saving visualization images.
                 Images are saved as "{image_name}_prototypes.jpg"
    
    Returns:
        None. Saves visualization images to vis_dir.
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Visualizing prototypes to {vis_dir}")
    
    num_layers = len(prototypes)
    if num_layers == 0:
        LOGGER.warning("No prototypes to visualize")
        return
    
    num_classes = len(prototypes[0]) if prototypes[0] else 0
    
    # Group prototypes by image path for efficient visualization
    img_prototypes = {}  # {img_path: [(layer_idx, cls_idx, bbox, cls, class_name), ...]}
    
    for layer_idx in range(num_layers):
        for cls_idx in range(num_classes):
            if prototypes[layer_idx][cls_idx] is not None:
                proto_data = prototypes[layer_idx][cls_idx]
                selected_prototypes, selected_regs, selected_cls, selected_bboxes, selected_meta_info = proto_data
                
                # Iterate through all selected prototypes (always stored as lists now)
                for prototype, reg, cls, bbox, meta_info in zip(
                    selected_prototypes, selected_regs, selected_cls, selected_bboxes, selected_meta_info
                ):
                    # Get image path from metadata
                    img_path = meta_info.get('image_path', None) if isinstance(meta_info, dict) else None
                    if img_path is None:
                        continue
                    
                    # Get predicted class and confidence
                    if isinstance(cls, torch.Tensor):
                        pred_class = cls.argmax().item() if cls.numel() > 0 else cls_idx
                        pred_conf = cls.softmax(0)[pred_class].item() if cls.numel() > 0 else 0.0
                    else:
                        pred_class = cls_idx
                        pred_conf = 0.0
                    
                    # Get class name
                    if class_names and pred_class < len(class_names):
                        class_name = class_names[pred_class]
                    else:
                        class_name = f"class_{pred_class}"
                    
                    # Convert decoded bbox (predicted bbox) from xywh to xyxy format (pixel coordinates)
                    # bbox is in xywh format (x_center, y_center, width, height) in pixel coordinates
                    decoded_bbox_xyxy = None
                    if isinstance(bbox, torch.Tensor):
                        bbox_np = bbox.cpu().numpy() if bbox.requires_grad else bbox.detach().cpu().numpy()
                    else:
                        bbox_np = np.array(bbox)
                    # Convert xywh to xyxy
                    if len(bbox_np) >= 4:
                        x_center, y_center, w, h = bbox_np[:4]
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        decoded_bbox_xyxy = np.array([x1, y1, x2, y2])
                    
                    if img_path not in img_prototypes:
                        img_prototypes[img_path] = []
                    img_prototypes[img_path].append((layer_idx, cls_idx, decoded_bbox_xyxy, pred_class, pred_conf, class_name))
    
    LOGGER.info(f"Visualizing {len(img_prototypes)} unique images")
    
    # Process each image
    for img_path, proto_list in img_prototypes.items():
        # Load image
        if not os.path.exists(img_path):
            LOGGER.warning(f"Image not found: {img_path}, skipping")
            continue
        
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 640))
        if img is None:
            LOGGER.warning(f"Failed to load image: {img_path}, skipping")
            continue
        
        img_h, img_w = 640, 640
        
        # Create annotator
        annotator = Annotator(img, line_width=2, font_size=1)
        
        # Process each prototype for this image
        for layer_idx, cls_idx, decoded_bbox_xyxy, pred_class, pred_conf, class_name in proto_list:
            # decoded_bbox_xyxy is already in xyxy format (4 values)
            if decoded_bbox_xyxy is not None and len(decoded_bbox_xyxy) >= 4:
                x1, y1, x2, y2 = decoded_bbox_xyxy[:4]
                
                # Clip to image boundaries
                x1 = max(0, min(float(x1), img_w))
                y1 = max(0, min(float(y1), img_h))
                x2 = max(0, min(float(x2), img_w))
                y2 = max(0, min(float(y2), img_h))
                
                # Draw bbox and label
                label = f"{class_name} {pred_conf:.2f} (L{layer_idx})"
                annotator.box_label([x1, y1, x2, y2], label, color=colors(pred_class, True))
        
        # Save visualized image
        img_name = Path(img_path).stem
        output_path = vis_dir / f"{img_name}_prototypes.jpg"
        cv2.imwrite(str(output_path), annotator.result())
    
    LOGGER.info(f"Visualization complete. Results saved to {vis_dir}")


def map_bbox_to_prototypes(detec_input, pred_output):
    """
    Map decoded bbox predictions back to each layer's spatial dimensions.
    
    The function splits the concatenated pred_output (which contains bboxes from all layers)
    back into per-layer bbox maps that match each layer's spatial dimensions.
    
    Args:
        detec_input: List[torch.Tensor], input feature maps to detection head.
                     Each element has shape [1, c_in_i, H_i, W_i] for layer i.
        pred_output: torch.Tensor, decoded predictions from detection head.
                     Shape: [1, 4+nc, H*W_all] where H*W_all is the concatenation of 
                     all layers' spatial dimensions (H*W). First 4 channels are decoded 
                     bboxes (treated as xywh format: x, y, w, h).

    Returns:
        bbox_map: List[torch.Tensor], one tensor per layer.
                  Each element has shape [1, 4, H_i, W_i] containing decoded bboxes 
                  in xywh format (x, y, w, h), reshaped to match the layer's spatial dimensions.
                  Note: The shape is [1, 4, H_i, W_i] (not [1, reg_max*4, H_i, W_i]) 
                  because pred_output contains already-decoded bboxes.
                  The bboxes are treated as xywh format (x, y, w, h) for IOU computation.
    """
    # pred_output shape: (B, 4+nc, H*W_all) where H*W_all is concatenated across all layers
    # We need to split it back to each layer's spatial dimensions
    
    bbox_map = []
    start_idx = 0
    
    for i, feat in enumerate(detec_input):
        # Get spatial dimensions of this layer
        B, C, H, W = feat.shape
        H_W = H * W
        
        # Extract bbox predictions for this layer (first 4 channels are decoded bbox, treated as xywh format)
        bbox_layer = pred_output[:, :4, start_idx:start_idx + H_W]  # (B, 4, H*W)
        bbox_layer = bbox_layer.view(B, 4, H, W)  # (B, 4, H, W) - decoded bbox in xywh format (x, y, w, h)
        
        bbox_map.append(bbox_layer)
        start_idx += H_W
    
    return bbox_map


def prototypes_with_max_iou_all_layers(
    detect_input_list, reg_output_list, cls_output_list, bbox_map_list, 
    gt_bboxes, gt_classes, imgsz=640
):
    """
    Select prototypes with maximum IOU across all layers for each ground truth bbox.
    
    This function merges prototypes from all layers and finds the location with maximum IOU
    for each GT bbox, then returns the layer index and location for each selected prototype.
    
    Args:
        detect_input_list: List[torch.Tensor], input feature maps for all layers.
                           Each element has shape [1, c_in_i, H_i, W_i].
        reg_output_list: List[torch.Tensor], regression outputs for all layers.
                         Each element has shape [1, reg_max*4, H_i, W_i].
        cls_output_list: List[torch.Tensor], classification outputs for all layers.
                         Each element has shape [1, nc, H_i, W_i].
        bbox_map_list: List[torch.Tensor], decoded bounding boxes for all layers (xywh format).
                       Each element has shape [1, 4, H_i, W_i].
        gt_bboxes: torch.Tensor, ground truth bounding boxes.
                   Shape: [n_gt, 4] in normalized xywh format (x, y, w, h).
        gt_classes: torch.Tensor, ground truth class labels.
                    Shape: [n_gt] with class indices.
        imgsz: int - image size (assumed square). Default: 640.

    Returns:
        List of tuples, one per GT bbox:
        Each tuple contains: (layer_idx, y, x, prototype, reg, cls, bbox, gt_class)
        - layer_idx: int - which layer the prototype comes from
        - y, x: int - spatial coordinates in that layer
        - prototype: torch.Tensor - feature patch (c_in*3*3,)
        - reg: torch.Tensor - regression output (reg_max*4,)
        - cls: torch.Tensor - classification output (nc,)
        - bbox: torch.Tensor - decoded bbox (4,) in xywh format
        - gt_class: int - ground truth class index
    """
    num_layers = len(detect_input_list)
    device = gt_bboxes.device
    n_gt = gt_bboxes.shape[0]
    
    if n_gt == 0:
        return []
    
    # Convert GT bboxes from normalized xywh to pixel coordinates (xywh format)
    gt_bboxes_px = gt_bboxes.clone()
    gt_bboxes_px[:, [0, 2]] = gt_bboxes_px[:, [0, 2]] * imgsz  # x, w
    gt_bboxes_px[:, [1, 3]] = gt_bboxes_px[:, [1, 3]] * imgsz  # y, h
    
    # Collect all bboxes from all layers and create index mapping
    all_bboxes_list = []  # List of bbox tensors
    layer_indices = []  # List of layer indices for each bbox
    spatial_indices = []  # List of (y, x) tuples for each bbox
    
    for layer_idx in range(num_layers):
        bbox_map = bbox_map_list[layer_idx]
        B, C, H, W = bbox_map.shape
        # Flatten spatial dimensions: (1, 4, H, W) -> (H*W, 4)
        bbox_flat = bbox_map[0].permute(1, 2, 0).reshape(-1, 4)  # (H*W, 4)
        all_bboxes_list.append(bbox_flat)
        
        # Create indices for this layer
        for y in range(H):
            for x in range(W):
                layer_indices.append(layer_idx)
                spatial_indices.append((y, x))
    
    if len(all_bboxes_list) == 0:
        return []
    
    # Concatenate all bboxes: (n_all, 4)
    all_bboxes_tensor = torch.cat(all_bboxes_list, dim=0)  # (n_all, 4)
    
    # Compute IOU between each GT bbox and all predicted bboxes across all layers
    iou_matrix = []
    for i in range(n_gt):
        gt_bbox_single = gt_bboxes_px[i:i+1]  # (1, 4) in xywh format
        # Compute IOU: (1, 4) vs (n_all, 4) -> (1, n_all)
        iou_row = bbox_iou(gt_bbox_single, all_bboxes_tensor, xywh=True)  # (1, n_all)
        iou_matrix.append(iou_row.squeeze(0))  # (n_all,)
    
    # Stack to get (n_gt, n_all)
    iou_matrix = torch.stack(iou_matrix, dim=0)  # (n_gt, n_all)
    
    # Find the location with maximum IOU for each GT bbox
    _, max_iou_indices = iou_matrix.max(dim=1)  # (n_gt,)
    
    # Extract prototypes for max IOU locations
    results = []
    for i in range(n_gt):
        best_idx = max_iou_indices[i].item()
        layer_idx = layer_indices[best_idx]
        y, x = spatial_indices[best_idx]
        
        # Extract prototype, reg, cls, bbox from the corresponding layer
        prototypes = detect_input_list[layer_idx]
        reg_output = reg_output_list[layer_idx]
        cls_output = cls_output_list[layer_idx]
        bbox_map = bbox_map_list[layer_idx]
        
        # Extract 3x3 feature patch
        feature_patch = extract_feature_patch(prototypes[0], y, x, patch_size=3)  # (c_in*3*3,)
        
        # Extract reg, cls, bbox at (y, x)
        reg_at_loc = reg_output[0, :, y, x]  # (reg_max*4,)
        cls_at_loc = cls_output[0, :, y, x]  # (nc,)
        bbox_at_loc = bbox_map[0, :, y, x]  # (4,)
        
        gt_class = gt_classes[i].item()
        
        results.append((layer_idx, y, x, feature_patch, reg_at_loc, cls_at_loc, bbox_at_loc, gt_class))
    
    return results


def generate_prototypes(
    model_path: str,
    data_yaml: str,
    output_path: str,
    device: str = "0",
    imgsz: int = 640,
    vis_dir: Optional[str] = None,
    load_hist: Optional[str] = None
):
    """
    Generate prototypes from training dataset.
    
    The function processes training images to extract representative prototypes:
    1. For each image, performs forward pass to get detection head features
    2. Selects prototypes with maximum IOU for each ground truth bbox across all layers
    3. Organizes prototypes by layer and ground truth class
    4. Uses K-means clustering (k_center=10) to select representative prototypes per class
    5. Saves prototypes as a list of tensors (one per layer)
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        data_yaml: Path to dataset YAML configuration file
        output_path: Path to save generated prototypes (.pt file)
        device: Device to use (e.g., "0" for GPU 0, "cpu" for CPU). Default: "0"
        imgsz: Image size for inference. Default: 640
        vis_dir: Optional directory to save visualization of prototypes. If None, no visualization is performed.
        load_hist: Optional path to existing prototypes file to load and merge with new prototypes. Default: None
    
    Returns:
        None. Saves prototypes to output_path and optionally generates visualizations.
    
    Output format:
        List of tensors, one per detection layer. Each tensor has shape:
        [num_prototypes_all_classes, feature_dim + reg_dim + cls_dim]
        where:
        - feature_dim = in_channels * 3 * 3 (3x3 feature patch)
        - reg_dim = reg_max * 4 (regression output)
        - cls_dim = num_classes (classification output)
    """
    # Load model
    model = YOLO(model_path)
    model.model.eval()
    model.model.to(device)
    
    # Get detection head
    detect_head = model.model.model[-1]
    
    # Get detection head settings
    num_layers = detect_head.nl
    reg_max = detect_head.reg_max
    num_classes = detect_head.nc
    
    # Extract sample and label paths
    data_dict = YAML.load(data_yaml)
    image_dir = data_dict["train"]
    if not os.path.exists(image_dir):
        image_dir = os.path.join(data_dict["path"], image_dir) if "path" in data_dict.keys() \
            else os.path.join(os.path.dirname(data_yaml), image_dir)
    label_dir = str(image_dir).replace("images", "labels")
    # get image and label file name list, ensure they match
    image_paths_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_paths_list = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    # Match image files with label files (remove extension and match)
    matched_pairs = []
    image_stems = {Path(f).stem: f for f in image_paths_list}
    label_stems = {Path(f).stem: f for f in label_paths_list}
    for stem in image_stems:
        if stem in label_stems:
            matched_pairs.append((image_stems[stem], label_stems[stem]))
        else:
            LOGGER.warning(f"Image {stem} matches no label, skipped.")
    
    # Store prototypes by layer and class
    prototypes_all = [[[] for j in range(num_classes)] for i in range(num_layers)]

    detect_input = None
    detect_output = None
    def input_hook(module, input):
        """Capture input features to detection head before processing (pre-forward hook)"""
        nonlocal detect_input
        x = input[0]
        detect_input = [feat.clone().detach() for feat in x]
    
    def output_hook(module, input, output):
        """Capture raw outputs from detection head"""
        nonlocal detect_output
        detect_output = output
    
    # Setup memory monitoring
    memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)
    pbar = TQDM(matched_pairs, desc="Generating prototypes")
    memory_monitor.set_progress_bar(pbar)
    memory_monitor.start_monitoring()
    
    with torch.no_grad():
        for image_file, label_file in pbar:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                LOGGER.warning(f"Failed to load image: {image_path}")
                continue
            image = cv2.resize(image, (640, 640))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0  # 0-1 normalization
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)  # HWC to CHW, move to device
            
            # Forward pass to get feature maps and outputs from detection head using hooks
            # Use pre-forward hook to capture input before module modifies it inplace
            input_hook_handle = detect_head.register_forward_pre_hook(input_hook)
            output_hook_handle = detect_head.register_forward_hook(output_hook)
            _ = model.model(image)
            input_hook_handle.remove()
            output_hook_handle.remove()
            
            # raw_outputs is a list of tensors, each of shape (B, reg_max*4+nc, H, W)
            # Split into regression and classification parts
            pred_output, raw_output = detect_output
            
            # Split each layer's output into regression and classification
            reg_output = []
            cls_output = []
            for output in raw_output:
                reg_part = output[:, :reg_max * 4, :, :]  # Regression: (B, reg_max*4, H, W)
                cls_part = output[:, reg_max * 4:, :, :]  # Classification: (B, nc, H, W)
                reg_output.append(reg_part)
                cls_output.append(cls_part)
            
            # Read instance-level ground truth
            label_path = os.path.join(label_dir, label_file)
            gt_cls = []
            gt_bboxes = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                    for _line in lines:
                        _line = _line.strip().split()
                        if len(_line) >= 5:
                            gt_cls.append(int(_line[0]))
                            gt_bboxes.append([float(x) for x in _line[1:5]])  # x, y, w, h normalized
            
            # Skip if no ground truth annotations
            if len(gt_bboxes) == 0:
                continue
            
            gt_cls = torch.tensor(gt_cls, device=device, dtype=torch.long)
            gt_bboxes = torch.tensor(gt_bboxes, device=device)

            # Map decoded bbox to the prototypes
            # Note: pred_output from detect_head._inference() uses decode_bboxes() which decodes
            # regression outputs to bboxes in xywh format: (x_center, y_center, width, height)
            bbox_map = map_bbox_to_prototypes(detect_input, pred_output)

            # Select prototypes with maximum IOU across all layers for each ground truth bbox
            # This merges all layers and finds the best matching prototype for each GT bbox
            selected_prototypes = prototypes_with_max_iou_all_layers(
                detect_input_list=detect_input,
                reg_output_list=reg_output,
                cls_output_list=cls_output,
                bbox_map_list=bbox_map,
                gt_bboxes=gt_bboxes,
                gt_classes=gt_cls,
                imgsz=imgsz
            )
            
            # Store prototypes using ground truth class labels, maintaining layer and class structure
            for layer_idx, y, x, prototype, reg, cls, bbox, gt_cls_idx in selected_prototypes:
                cls_idx = gt_cls_idx
                meta_info = {
                    "image_path": image_path
                }
                prototypes_all[layer_idx][cls_idx].append((prototype.cpu(), reg.cpu(), cls.cpu(), bbox.cpu(), meta_info))
    memory_monitor.stop_monitoring()
    
    # Check if any prototypes were generated
    total_prototypes = sum(
        sum(len(protos) for protos in layer_list)
        for layer_list in prototypes_all
    )
    if total_prototypes == 0:
        raise RuntimeError("No prototypes generated! Check dataset and model compatibility.")
    
    # Perform clustering algorithm to select representative prototypes
    prototypes = [[None for j in range(num_classes)] for i in range(num_layers)]
    k_center = 10  # Fixed number of clusters per class
    for layer_idx in range(num_layers):
        for cls_idx in range(num_classes):
            if len(prototypes_all[layer_idx][cls_idx]) > 0:
                # Collect all prototypes for this layer and class
                prototype_list = []
                reg_list = []
                cls_list = []
                bbox_list = []
                meta_info_list = []
                
                for proto_data in prototypes_all[layer_idx][cls_idx]:
                    prototype, reg, cls, bbox, meta_info = proto_data
                    prototype_list.append(prototype)
                    reg_list.append(reg)
                    cls_list.append(cls)
                    bbox_list.append(bbox)
                    meta_info_list.append(meta_info)
                
                # Stack prototypes into a tensor for clustering
                prototypes_tensor = torch.stack(prototype_list, dim=0)  # (n_prototypes, feature_dim)
                
                # Perform clustering to find representative prototypes
                center_idx = k_means(prototypes_tensor.to(device), k_center).cpu()
                
                # Select representative prototypes based on cluster centers
                selected_prototypes = []
                selected_regs = []
                selected_cls = []
                selected_bboxes = []
                selected_meta_info = []
                
                for idx in center_idx:
                    idx_item = idx.item()
                    selected_prototypes.append(prototype_list[idx_item])
                    selected_regs.append(reg_list[idx_item])
                    selected_cls.append(cls_list[idx_item])
                    selected_bboxes.append(bbox_list[idx_item])
                    selected_meta_info.append(meta_info_list[idx_item])
                
                prototypes[layer_idx][cls_idx] = (
                    selected_prototypes, selected_regs, selected_cls, 
                    selected_bboxes, selected_meta_info
                )
            else:
                LOGGER.warning(f"Prototypes for layer {layer_idx} and class {cls_idx} are not generated")
    
    # Concatenate prototypes and save
    # Output format: List of tensors, one per detection layer
    # Each tensor contains prototypes from all classes concatenated along the first dimension
    # Shape: [num_prototypes_all_classes, feature_dim + reg_dim + cls_dim]
    # where:
    #   - feature_dim = in_channels * 3 * 3 (flattened 3x3 feature patch)
    #   - reg_dim = reg_max * 4 (regression output channels)
    #   - cls_dim = num_classes (classification output channels)
    prototypes_save = [None for i in range(num_layers)]
    for layer_idx in range(num_layers):
        for cls_idx in range(num_classes):
            if prototypes[layer_idx][cls_idx] is not None:
                selected_prototypes, selected_regs, selected_cls, selected_bboxes, selected_meta_info = prototypes[layer_idx][cls_idx]
                
                # Skip if no prototypes selected
                if len(selected_prototypes) == 0:
                    continue
                
                # Stack lists of tensors into tensors
                # selected_prototypes is a list of tensors, each of shape (feature_dim,)
                prototypes_tensor = torch.stack(selected_prototypes, dim=0)  # (n, feature_dim)
                reg_tensor = torch.stack(selected_regs, dim=0)  # (n, reg_dim)
                cls_tensor = torch.stack(selected_cls, dim=0)  # (n, cls_dim)
                
                # Concatenate prototype, reg, and cls along feature dimension: (n, feature_dim + reg_dim + cls_dim)
                combined = torch.cat([prototypes_tensor, reg_tensor, cls_tensor], dim=1)
                
                # Concatenate prototypes from different classes within the same layer
                if prototypes_save[layer_idx] is None:
                    prototypes_save[layer_idx] = combined
                else:
                    prototypes_save[layer_idx] = torch.cat([prototypes_save[layer_idx], combined], dim=0)
    
    # Load historical prototypes if specified and merge with new prototypes
    if load_hist is not None:
        LOGGER.info(f"Loading historical prototypes from {load_hist}")
        hist_prototypes = torch.load(load_hist, map_location='cpu')
        if isinstance(hist_prototypes, list) and len(hist_prototypes) == num_layers:
            # Merge historical prototypes with new prototypes
            for layer_idx in range(num_layers):
                if hist_prototypes[layer_idx] is not None:
                    if prototypes_save[layer_idx] is not None:
                        # Both have prototypes, concatenate them
                        prototypes_save[layer_idx] = torch.cat(
                            [hist_prototypes[layer_idx], prototypes_save[layer_idx]], dim=0
                        )
                    else:
                        # Only historical prototypes exist, use them
                        prototypes_save[layer_idx] = hist_prototypes[layer_idx]
                else:
                    continue # If historical prototypes do not exist, use only the new prototypes
        else:
            LOGGER.warning(
                f"Historical prototypes file has unexpected format (expected list of {num_layers} tensors). "
                f"Saving only new prototypes."
            )
    
    torch.save(prototypes_save, output_path)
    LOGGER.info(f"Prototypes saved to {output_path}")
    
    # Visualize prototypes if vis_dir is specified
    if vis_dir:
        # Get model class names
        class_names = None
        if hasattr(model.model, 'names'):
            class_names = list(model.model.names.values()) if isinstance(model.model.names, dict) else model.model.names
        
        visualize_prototypes(
            prototypes,
            class_names,
            vis_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate prototypes for Prototype Replay mechanism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save generated prototypes (.pt file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (e.g., '0' for GPU 0, 'cpu' for CPU). Default: '0'"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference. Default: 640"
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Optional directory to save visualization of representative prototypes. If not specified, no visualization is performed."
    )
    parser.add_argument(
        "--load_hist",
        type=str,
        default=None,
        help="Optional path to existing prototypes file to load and merge with new prototypes. Default: None"
    )
    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    generate_prototypes(
        model_path=args.model,
        data_yaml=args.data,
        output_path=args.output,
        device=args.device,
        imgsz=args.imgsz,
        vis_dir=args.vis_dir,
        load_hist=args.load_hist
    )


if __name__ == "__main__":
    main()

