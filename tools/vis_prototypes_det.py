#!/usr/bin/env python3
"""
Visualize and evaluate prototypes by replaying them through the YOLO detection head.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.plotting import colors


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


def draw_labeled_box(img, bbox_px, label_lines, color, dashed=False):
    """Draws a box with text labels below the image."""
    x1, y1, x2, y2 = map(int, bbox_px)
    
    # Draw Box
    if dashed:
        gap = 10
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)] # Top, Right, Bot, Left
        for i in range(4):
            p1, p2 = pts[i], pts[i+1]
            if p1[0] == p2[0]: # Vertical
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]), gap*2):
                    cv2.line(img, (p1[0], y), (p1[0], min(y+gap, max(p1[1], p2[1]))), color, 2)
            else: # Horizontal
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]), gap*2):
                    cv2.line(img, (x, p1[1]), (min(x+gap, max(p1[0], p2[0])), p1[1]), color, 2)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return label_lines # Return text to be appended to footer


def run_evaluation(model, prototypes, meta_info, device, max_protos=torch.inf):
    class_names = model.names
    detect = model.model.model[-1]
    
    # Identify channels dynamically
    in_channels = [m[0].conv.in_channels if hasattr(m[0], 'conv') else m[0].in_channels 
                   for m in detect.cv3]
    
    results = []
    losses = {'cls': 0.0, 'reg': 0.0, 'count': 0, 'correct_both': 0}
    
    with torch.no_grad():
        for l_idx, (proto_tensor, metas) in enumerate(zip(prototypes, meta_info)):
            if proto_tensor is None: continue
            
            # Slicing Dimensions
            nc = detect.nc
            reg_out = detect.reg_max * 4
            feat_dim = in_channels[l_idx] * 25
            
            # Slice Tensor: [Feat | Reg | Cls | Mask]
            feats = proto_tensor[:, :feat_dim]
            regs_sup = proto_tensor[:, feat_dim : feat_dim + reg_out]
            clss_sup = proto_tensor[:, feat_dim + reg_out : feat_dim + reg_out + nc]
            masks = proto_tensor[:, -25:]

            limit = min(len(feats), max_protos)
            LOGGER.info(f"Layer {l_idx}: Eval {limit}/{len(feats)} prototypes")

            for i in range(limit):
                meta = metas[i]
                y, x = meta['yx']
                H, W = meta['HWs'][l_idx] # Note: meta stores HW for specific layer if generated correctly
                
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
                
                # Decoding BBox (simplified: assume decoded output is aligned linearly)
                # Need to find linear index of (y,x) in the concatenated decoded output
                # This depends on stride order. 
                # Alternative: Use meta['bbox_pred_px'] stored during generation? 
                # The script asks to *replay*. Let's extract from `decoded` based on shape.
                offset = sum(inputs[k].shape[2]*inputs[k].shape[3] for k in range(l_idx))
                idx_linear = offset + y * W + x
                pred_bbox_xywh = decoded[0, :4, idx_linear] 
                
                # 4. Losses & Metrics
                cls_out_log_softmax = F.log_softmax(pred_cls, dim=0)
                cls_supervision_softmax = F.softmax(clss_sup[i].to(device), dim=0)
                loss_cls = F.kl_div(cls_out_log_softmax, cls_supervision_softmax, reduction="batchmean")
                # loss_cls = F.binary_cross_entropy_with_logits(pred_cls, torch.sigmoid(clss_sup[i].to(device))).item()

                reg_out_log_softmax = F.log_softmax(pred_reg.reshape(-1, detect.reg_max), dim=1)  # [num_prototypes*4, reg_max]
                reg_supervision_softmax = F.softmax(regs_sup[i].to(device).reshape(-1, detect.reg_max), dim=1)  # [num_prototypes*4, reg_max]
                loss_reg = F.kl_div(reg_out_log_softmax, reg_supervision_softmax, reduction='batchmean')
                # loss_reg = F.mse_loss(pred_reg, regs_sup[i].to(device)).item()
                
                losses['cls'] += loss_cls
                losses['reg'] += loss_reg
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
                if both_correct:
                    losses['correct_both'] += 1
                
                # Store
                results.append({
                    'meta': meta,
                    'layer': l_idx,
                    'pred': {'cls': class_names[pred_cls.argmax().item()],
                             'cls_id': pred_cls.argmax().item(),
                             'conf': pred_cls.softmax(0).max().item(),
                             'bbox': pred_bbox_xyxy.cpu()},
                    'sup': {'cls': class_names[clss_sup[i].argmax().item()],
                            'cls_id': clss_sup[i].argmax().item(),
                            'conf': clss_sup[i].softmax(0).max().item()},
                    'loss': {'cls': loss_cls, 'reg': loss_reg},
                    'metrics': {
                        'iou': iou.item(),
                        'iou_ok': bool(iou_ok),
                        'cls_correct': bool(cls_correct),
                        'both_ok': both_correct,
                    }
                })

    # Average
    if losses['count']:
        losses['cls'] /= losses['count']
        losses['reg'] /= losses['count']
    
    acc = losses['correct_both'] / losses['count'] if losses['count'] else 0.0
    losses['acc'] = acc
    return losses, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    
    if str.isdigit(args.device):
        args.device = f"cuda:{args.device}"
    device = torch.device(args.device)
    model = YOLO(args.model).to(device).eval()
    model.fuse()
    data = torch.load(args.prototypes, map_location='cpu')
    for i in range(len(data['prototypes'])):
        data['prototypes'][i] = data['prototypes'][i].to(device)
    
    losses, results = run_evaluation(model, data['prototypes'], data['meta_info'], device)
    
    # Save Report
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'report.txt', 'w') as f:
        f.write(f"Cls Loss: {losses['cls']:.4f}\nReg Loss: {losses['reg']:.4f}\n")
        f.write(f"Acc (IoU>0.5 & cls correct): {losses['acc']:.4f}\n")
    
    # Visualize
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    img_groups = {}
    for r in results:
        img_groups.setdefault(r['meta']['img_path'], []).append(r)
        
    for path, res_list in TQDM(img_groups.items(), desc="Visualizing"):
        if not os.path.exists(path):
            LOGGER.warning(f"Image file {path} not found, skipped")
            continue
        img = cv2.imread(path)
        h, w = img.shape[:2]
        
        footer_texts = []
        for i, res in enumerate(res_list):
            color = [int(c) for c in colors(i)]
            
            # Ground Truth / Supervision (Dashed)
            # Use bbox from meta (gt_norm) converted to px
            gt = res['meta']['bbox_gt_norm'].numpy()
            gt_px = gt * [w, h, w, h]
            gt_px_xyxy = [gt_px[0]-gt_px[2]/2, gt_px[1]-gt_px[3]/2, gt_px[0]+gt_px[2]/2, gt_px[1]+gt_px[3]/2]
            draw_labeled_box(img, gt_px_xyxy, [], color, dashed=True)
            
            # Prediction (Solid)
            pred_px = res['pred']['bbox'].numpy()
            # Note: pred_bbox in result is likely relative to 640 or whatever input size. 
            # Scale to img size
            imgsz = args.imgsz
            scale = [w/imgsz, h/imgsz, w/imgsz, h/imgsz] # Simplified assumption
            pred_px = pred_px * scale
            
            txt = f"L{res['layer']} P:{res['pred']['cls']}({res['pred']['cls_id']}) ({res['pred']['conf']:.2f}) | GT:{res['sup']['cls']}({res['sup']['cls_id']})"
            footer_texts.append(txt)
            draw_labeled_box(img, pred_px, [], color, dashed=False)
            
        # Draw footer
        pad = 20
        font_scale = 0.5  # Font size (adjust as needed: 0.4=small, 0.6=default, 0.8=medium, 1.0=large, 1.5=very large)
        line_height = int(30 * font_scale / 0.6)  # Adjust line height proportionally with font size
        footer = np.zeros((len(footer_texts)*line_height + pad, w, 3), dtype=np.uint8)
        for i, txt in enumerate(footer_texts):
            cv2.putText(footer, txt, (10, line_height*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
        
        final = np.vstack([img, footer])
        cv2.imwrite(str(vis_dir / Path(path).name), final)


if __name__ == "__main__":
    main()