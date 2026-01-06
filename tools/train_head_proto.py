#!/usr/bin/env python3
"""
Train YOLO detection head using prototypes as training samples.
Freezes all layers except the detection head.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM


def restore_prototypes(prototypes, pad_mask, device):
    """
    Restore padded prototypes back to 5x5 feature maps and compute offsets.

    Args:
        prototypes (Tensor): Tensor of shape (N, C, 5, 5) containing prototype features.
        pad_mask (Tensor): Tensor of shape (N, 5, 5) indicating valid (1) and padded (0) regions.
        device: Device to place tensors on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Restored prototypes (N, C, 5, 5), offset_y (N,), offset_x (N,)
    """
    num_prototypes = prototypes.shape[0]
    in_channels = prototypes.shape[1]
    restored_prototypes = torch.zeros([num_prototypes, in_channels, 5, 5], device=device)
    offset_y_batch = torch.zeros([num_prototypes], device=device, dtype=torch.long)
    offset_x_batch = torch.zeros([num_prototypes], device=device, dtype=torch.long)

    for k in range(num_prototypes):
        mask = pad_mask[k]  # [5, 5], 1=original, 0=padded
        proto = prototypes[k]  # [in_channels, 5, 5]

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
            offset_y_batch[k] = offset_y
            offset_x_batch[k] = offset_x

            # Place valid region at the offset position in restored feature map
            restored_h_start = mask_h_start + offset_y
            restored_w_start = mask_w_start + offset_x
            restored_h_end = mask_h_end + offset_y
            restored_w_end = mask_w_end + offset_x

            # Place the valid region at the corresponding position
            restored_prototypes[k, :, restored_h_start:restored_h_end, restored_w_start:restored_w_end] = valid_proto

    return restored_prototypes, offset_y_batch, offset_x_batch


def place_patch(feature_map, patch_flat, valid_mask_flat, y, x):
    """
    Places a flattened 5x5 patch back into the feature map at (y,x),
    respecting image boundaries defined by valid_mask.
    """
    C, H, W = feature_map.shape
    patch = patch_flat.view(C, 5, 5)
    mask = valid_mask_flat.view(5, 5)

    # Locate valid region in mask
    valid_rows = torch.where(mask.sum(dim=1) > 0)[0]
    valid_cols = torch.where(mask.sum(dim=0) > 0)[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return

    patch_h_start, patch_h_end = valid_rows[0].item(), valid_rows[-1].item() + 1
    patch_w_start, patch_w_end = valid_cols[0].item(), valid_cols[-1].item() + 1

    # Map patch coords to feature map coords (centered at y, x)
    tgt_h_start = y - 2 + patch_h_start
    tgt_h_end = y - 2 + patch_h_end
    tgt_w_start = x - 2 + patch_w_start
    tgt_w_end = x - 2 + patch_w_end

    # Clip to feature map bounds
    tgt_h_start_clip = max(0, tgt_h_start)
    tgt_w_start_clip = max(0, tgt_w_start)
    tgt_h_end_clip = min(H, tgt_h_end)
    tgt_w_end_clip = min(W, tgt_w_end)

    if tgt_h_start_clip >= tgt_h_end_clip or tgt_w_start_clip >= tgt_w_end_clip:
        return

    # Corresponding patch indices after clipping
    src_h_start = patch_h_start + (tgt_h_start_clip - tgt_h_start)
    src_w_start = patch_w_start + (tgt_w_start_clip - tgt_w_start)
    src_h_end = src_h_start + (tgt_h_end_clip - tgt_h_start_clip)
    src_w_end = src_w_start + (tgt_w_end_clip - tgt_w_start_clip)

    feature_map[:, tgt_h_start_clip:tgt_h_end_clip, tgt_w_start_clip:tgt_w_end_clip] = \
        patch[:, src_h_start:src_h_end, src_w_start:src_w_end]


def evaluate_prototypes(model, prototypes_list, meta_info, device, max_protos=None):
    """
    Evaluate detection head performance on prototypes.
    
    Args:
        model: YOLO model
        prototypes_list: List of prototype tensors for each detection layer
        meta_info: List of metadata for each layer's prototypes
        device: Device to run computation on
        max_protos: Maximum number of prototypes to evaluate per layer (None for all)
    
    Returns:
        dict: Evaluation metrics including location and classification accuracy
    """
    detect = model.model.model[-1]
    detect.eval()
    
    # Identify channels dynamically
    in_channels = [m[0].conv.in_channels if hasattr(m[0], 'conv') else m[0].in_channels 
                   for m in detect.cv3]
    
    losses = {'cls': 0.0, 'reg': 0.0, 'count': 0, 
              'correct_both': 0, 'correct_loc': 0, 'correct_cls': 0}
    
    with torch.no_grad():
        for l_idx, proto_tensor in enumerate(prototypes_list):
            if proto_tensor is None:
                continue
            
            # Get metadata for this layer
            if meta_info is None or l_idx >= len(meta_info):
                continue
            metas = meta_info[l_idx]
            if not metas or len(metas) == 0:
                continue
            
            # Slicing Dimensions
            nc = detect.nc
            reg_out = detect.reg_max * 4
            feat_dim = in_channels[l_idx] * 25
            
            # Slice Tensor: [Feat | Reg | Cls | Mask]
            feats = proto_tensor[:, :feat_dim]
            regs_sup = proto_tensor[:, feat_dim : feat_dim + reg_out]
            clss_sup = proto_tensor[:, feat_dim + reg_out : feat_dim + reg_out + nc]
            masks = proto_tensor[:, -25:]

            limit = min(len(feats), max_protos) if max_protos is not None else len(feats)
            if limit == 0:
                continue

            for i in range(limit):
                if i >= len(metas):
                    break
                    
                meta = metas[i]
                y, x = meta['yx']
                H, W = meta['HWs'][l_idx]
                
                # 1. Reconstruct Input
                # Create empty inputs for all layers (Detect expects list)
                inputs = [torch.zeros(1, ch, *meta['HWs'][l_idx], device=device) for ch in in_channels]
                # Specifically fill current layer
                place_patch(inputs[l_idx][0], feats[i].to(device), masks[i].to(device), y, x)
                
                # 2. Forward
                # Detect returns [decoded, raw_list]
                decoded, raw_list = detect(inputs)
                
                # 3. Extract Predictions at (y,x)
                pred_reg_map = raw_list[l_idx][:, :reg_out]
                pred_cls_map = raw_list[l_idx][:, reg_out:]
                
                pred_reg = pred_reg_map[0, :, y, x]
                pred_cls = pred_cls_map[0, :, y, x]
                
                # Decoding BBox
                offset = sum(inputs[k].shape[2]*inputs[k].shape[3] for k in range(l_idx))
                idx_linear = offset + y * W + x
                pred_bbox_xywh = decoded[0, :4, idx_linear]
                
                # 4. Losses & Metrics
                loss_cls = F.binary_cross_entropy_with_logits(pred_cls, clss_sup[i].to(device).sigmoid()) \
                    -F.binary_cross_entropy_with_logits(clss_sup[i].to(device), clss_sup[i].to(device).sigmoid())

                reg_supervision_softmax = F.softmax(regs_sup[i].to(device).reshape(-1, detect.reg_max), dim=1)  # [4, reg_max]
                loss_reg = F.cross_entropy(pred_reg.reshape(-1, detect.reg_max), reg_supervision_softmax)
                
                losses['cls'] += loss_cls.item()
                losses['reg'] += loss_reg.item()
                losses['count'] += 1
                
                # 5. IoU Check
                pred_bbox_xyxy = torch.cat([
                    pred_bbox_xywh[:2] - pred_bbox_xywh[2:]/2,
                    pred_bbox_xywh[:2] + pred_bbox_xywh[2:]/2
                ])
                gt_xywh = meta['bbox_gt_norm'].to(device)
                gt_bbox_xyxy = torch.stack([
                    gt_xywh[0] - gt_xywh[2] / 2,
                    gt_xywh[1] - gt_xywh[3] / 2,
                    gt_xywh[0] + gt_xywh[2] / 2,
                    gt_xywh[1] + gt_xywh[3] / 2,
                ]) * 640

                # Compute IoU between pred and gt
                inter_lt = torch.max(pred_bbox_xyxy[:2], gt_bbox_xyxy[:2])
                inter_rb = torch.min(pred_bbox_xyxy[2:], gt_bbox_xyxy[2:])
                inter_wh = (inter_rb - inter_lt).clamp(min=0)
                inter_area = inter_wh[0] * inter_wh[1]
                pred_area = (pred_bbox_xyxy[2] - pred_bbox_xyxy[0]) * (pred_bbox_xyxy[3] - pred_bbox_xyxy[1])
                gt_area = (gt_bbox_xyxy[2] - gt_bbox_xyxy[0]) * (gt_bbox_xyxy[3] - gt_bbox_xyxy[1])
                union_area = pred_area + gt_area - inter_area + 1e-6
                iou = inter_area / union_area
                iou_ok = iou > 0.5

                # Classification correctness
                pred_cls_id = pred_cls.argmax().item()
                sup_cls_id = clss_sup[i].argmax().item()
                cls_correct = pred_cls_id == sup_cls_id

                # Both IoU and classification correct
                both_correct = bool(iou_ok and cls_correct)
                if iou_ok:
                    losses['correct_loc'] += 1
                if cls_correct:
                    losses['correct_cls'] += 1
                if both_correct:
                    losses['correct_both'] += 1

    # Average
    if losses['count']:
        losses['cls'] /= losses['count']
        losses['reg'] /= losses['count']
    
    losses['acc_loc'] = losses['correct_loc'] / losses['count'] if losses['count'] else 0.0
    losses['acc_cls'] = losses['correct_cls'] / losses['count'] if losses['count'] else 0.0
    losses['acc'] = losses['correct_both'] / losses['count'] if losses['count'] else 0.0
    
    detect.train()
    return losses


def compute_proto_replay_loss(model, prototypes_list, batch_size, device, lid, batch_idx=0, distill_model=None):
    """
    Compute prototype replay classification and regression losses for a specific layer and batch.
    
    Args:
        model: YOLO model
        prototypes_list: List of prototype tensors for each detection layer
        batch_size: Batch size for processing prototypes
        device: Device to run computation on
        lid: Layer index to process
        batch_idx: Batch index for processing prototypes in batches
        distill_model: Optional distillation model to use for supervision signals
    
    Returns:
        Tuple[Tensor, Tensor]: (cls_loss_proto, reg_loss_proto)
    """
    detect = model.model.model[-1]
    # Set detection head to eval mode for computing loss
    detect.eval()
    cls_loss_proto = 0.0
    reg_loss_proto = 0.0
    reg = detect.cv2
    cls = detect.cv3
    reg_max = detect.reg_max
    
    if prototypes_list[lid] is None or torch.numel(prototypes_list[lid]) == 0:
        detect.train()
        return cls_loss_proto, reg_loss_proto

    in_channels = cls[lid][0].conv.in_channels
    reg_out_channels = reg[lid][-1].out_channels
    cls_out_channels = cls[lid][-1].out_channels

    pad_mask = prototypes_list[lid][:, in_channels * 5 * 5 + reg_out_channels + cls_out_channels :].reshape(-1, 5, 5)
    prototypes = prototypes_list[lid][:, : in_channels * 5 * 5].reshape(-1, in_channels, 5, 5)
    num_prototypes_all = prototypes.shape[0]

    # Process prototypes in batches
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_prototypes_all)
    pad_mask = pad_mask[start_idx:end_idx]
    prototypes = prototypes[start_idx:end_idx]
    num_prototypes = prototypes.shape[0]
    
    if num_prototypes == 0:
        detect.train()
        return cls_loss_proto, reg_loss_proto

    prototypes, offset_y_batch, offset_x_batch = restore_prototypes(prototypes, pad_mask, device)

    reg_out = reg[lid](prototypes)
    cls_out = cls[lid](prototypes)

    y_positions = offset_y_batch + 2  # [num_prototypes]
    x_positions = offset_x_batch + 2  # [num_prototypes]

    reg_out_list = []
    cls_out_list = []
    for i in range(num_prototypes):
        reg_out_list.append(reg_out[i, :, y_positions[i], x_positions[i]])
        cls_out_list.append(cls_out[i, :, y_positions[i], x_positions[i]])
    reg_out = torch.stack(reg_out_list)  # (num_prototypes, out_channels)
    cls_out = torch.stack(cls_out_list)  # (num_prototypes, out_channels)

    # Use distillation model or prototype's built-in supervision
    if distill_model is not None:
        with torch.no_grad():
            distill_detect = distill_model.model[-1]
            distill_reg = distill_detect.cv2
            distill_cls = distill_detect.cv3
            distill_reg_out = distill_reg[lid](prototypes)
            distill_cls_out = distill_cls[lid](prototypes)
            reg_supervision_list = []
            cls_supervision_list = []
            for i in range(num_prototypes):
                reg_supervision_list.append(distill_reg_out[i, :, y_positions[i], x_positions[i]])
                cls_supervision_list.append(distill_cls_out[i, :, y_positions[i], x_positions[i]])
            reg_supervision = torch.stack(reg_supervision_list)  # (num_prototypes, reg_out_channels)
            cls_supervision = torch.stack(cls_supervision_list)  # (num_prototypes, cls_out_channels)
    else:
        # Use prototype's built-in supervision
        reg_supervision = prototypes_list[lid][:, in_channels * 5 * 5 : in_channels * 5 * 5 + reg_out_channels]
        cls_supervision = prototypes_list[lid][:, in_channels * 5 * 5 + reg_out_channels : in_channels * 5 * 5 + reg_out_channels + cls_out_channels]
        reg_supervision = reg_supervision[start_idx:end_idx]
        cls_supervision = cls_supervision[start_idx:end_idx]

    cls_loss_proto = (F.binary_cross_entropy_with_logits(cls_out, cls_supervision.sigmoid())
        - F.binary_cross_entropy_with_logits(cls_supervision, cls_supervision.sigmoid()))  # min value of cls_loss_proto

    reg_supervision = F.softmax(reg_supervision.reshape(-1, reg_max), dim=1)  # [num_prototypes*4, reg_max]
    reg_loss_proto = F.cross_entropy(reg_out.reshape(-1, reg_max), reg_supervision)

    # Set detection head back to train mode
    detect.train()
    
    return cls_loss_proto, reg_loss_proto


def train_detection_head(model_path, prototypes_path, output_path, epochs=10, batch_size=32, lr=0.001, device='cuda', distill_model_path=None):
    """
    Train YOLO detection head using prototypes.
    
    Args:
        model_path: Path to YOLO model checkpoint
        prototypes_path: Path to prototypes file
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for processing prototypes
        lr: Learning rate
        device: Device to use ('cuda' or 'cpu')
        distill_model_path: Optional path to distillation model for supervision signals
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Load model
    LOGGER.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    model = model.to(device)
    
    # Load distillation model if provided
    distill_model = None
    if distill_model_path is not None:
        LOGGER.info(f"Loading distillation model from {distill_model_path}")
        distill_model = YOLO(distill_model_path)
        distill_model = distill_model.to(device)
        distill_model.eval()
        for param in distill_model.parameters():
            param.requires_grad = False
        LOGGER.info("Distillation model loaded and frozen")
    
    # Freeze all layers except detection head (model[-1])
    LOGGER.info("Freezing all layers except detection head...")
    num_layers = len(model.model.model)
    detection_head_prefix = f'model.{num_layers - 1}.'
    
    for name, param in model.model.named_parameters():
        if name.startswith(detection_head_prefix):
            param.requires_grad = True
            LOGGER.info(f"Training parameter: {name}")
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.model.parameters())
    LOGGER.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load prototypes
    LOGGER.info(f"Loading prototypes from {prototypes_path}")
    prototypes_data = torch.load(prototypes_path, map_location=device)
    prototypes_list = prototypes_data["prototypes"]  # List[torch.Tensor]
    meta_info = prototypes_data.get("meta_info", None)  # List of metadata for each layer
    
    if meta_info is None:
        LOGGER.warning("No meta_info found in prototypes file. Evaluation will be skipped.")
    
    # Move prototypes to device and set requires_grad=False
    for lid, x in enumerate(prototypes_list):
        if x is not None:
            prototypes_list[lid] = x.to(device)
            prototypes_list[lid].requires_grad_(False)
    
    # Calculate total number of prototypes and batches per layer
    detect = model.model.model[-1]
    total_prototypes_per_layer = []
    batches_per_layer = []
    for lid in range(detect.nl):
        if prototypes_list[lid] is not None and torch.numel(prototypes_list[lid]) > 0:
            num_prototypes = prototypes_list[lid].shape[0]
            total_prototypes_per_layer.append(num_prototypes)
            batches_per_layer.append((num_prototypes + batch_size - 1) // batch_size)
        else:
            total_prototypes_per_layer.append(0)
            batches_per_layer.append(0)
    
    total_prototypes = sum(total_prototypes_per_layer)
    total_batches = sum(batches_per_layer)
    LOGGER.info(f"Total prototypes: {total_prototypes}")
    LOGGER.info(f"Batch size: {batch_size}, Total batches per epoch: {total_batches}")
    for lid, (num_proto, num_batches) in enumerate(zip(total_prototypes_per_layer, batches_per_layer)):
        if num_proto > 0:
            LOGGER.info(f"  Layer {lid}: {num_proto} prototypes, {num_batches} batches")
    
    # Setup optimizer (only for detection head parameters)
    optimizer = Adam([p for p in detect.parameters() if p.requires_grad], lr=lr)
    scheduler = StepLR(optimizer, step_size=epochs // 2, gamma=0.1)
    
    # Training loop
    LOGGER.info(f"Starting training for {epochs} epochs...")
    model.model.train()
    
    # Evaluate before train
    if meta_info is not None:
        LOGGER.info(f"Evaluating prototypes before train...")
        eval_metrics = evaluate_prototypes(model, prototypes_list, meta_info, device)
        LOGGER.info(f"Initial Evaluation - "
                    f"Loc Acc (IoU>0.5): {eval_metrics['acc_loc']:.4f}, "
                    f"Cls Acc: {eval_metrics['acc_cls']:.4f}, "
                    f"Both Acc: {eval_metrics['acc']:.4f}, "
                    f"Cls Loss: {eval_metrics['cls']:.4f}, "
                    f"Reg Loss: {eval_metrics['reg']:.4f}")

    for epoch in range(epochs):
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        
        # Process prototypes in batches for each layer
        for lid in range(detect.nl):
            if prototypes_list[lid] is None or torch.numel(prototypes_list[lid]) == 0:
                continue
            
            num_prototypes = prototypes_list[lid].shape[0]
            num_batches_layer = (num_prototypes + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches_layer):
                cls_loss, reg_loss = compute_proto_replay_loss(
                    model, prototypes_list, batch_size, device, 
                    lid=lid, batch_idx=batch_idx, distill_model=distill_model
                )
                
                # Only accumulate if there's actual loss (non-zero batch)
                if cls_loss.item() != 0.0 or reg_loss.item() != 0.0:
                    total_loss = cls_loss + reg_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_cls_loss += cls_loss.item()
                    epoch_reg_loss += reg_loss.item()
                    num_batches += 1
        
        scheduler.step()
        
        if num_batches > 0:
            avg_cls_loss = epoch_cls_loss / num_batches
            avg_reg_loss = epoch_reg_loss / num_batches
            avg_total_loss = avg_cls_loss + avg_reg_loss
            LOGGER.info(f"Epoch [{epoch+1}/{epochs}] - CLS Loss: {avg_cls_loss:.6f}, REG Loss: {avg_reg_loss:.6f}, Total Loss: {avg_total_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}, Batches: {num_batches}")
        else:
            LOGGER.warning(f"Epoch [{epoch+1}/{epochs}] - No batches processed")
        
        # Evaluate after each epoch
        if meta_info is not None and (epoch+1)%10==0:
            LOGGER.info(f"Evaluating prototypes after epoch {epoch+1}...")
            eval_metrics = evaluate_prototypes(model, prototypes_list, meta_info, device)
            LOGGER.info(f"Epoch [{epoch+1}/{epochs}] Evaluation - "
                       f"Loc Acc (IoU>0.5): {eval_metrics['acc_loc']:.4f}, "
                       f"Cls Acc: {eval_metrics['acc_cls']:.4f}, "
                       f"Both Acc: {eval_metrics['acc']:.4f}, "
                       f"Cls Loss: {eval_metrics['cls']:.4f}, "
                       f"Reg Loss: {eval_metrics['reg']:.4f}")
    
    # Save model
    LOGGER.info(f"Saving trained model to {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    model.save(output_path)
    LOGGER.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train YOLO detection head using prototypes')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model checkpoint')
    parser.add_argument('--prototypes', type=str, required=True,
                        help='Path to prototypes file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing prototypes (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--distill-model', type=str, default=None,
                        help='Optional path to distillation model for supervision signals (default: None, uses prototype built-in supervision)')
    
    args = parser.parse_args()
    
    train_detection_head(
        model_path=args.model,
        prototypes_path=args.prototypes,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        distill_model_path=args.distill_model
    )


if __name__ == '__main__':
    main()

