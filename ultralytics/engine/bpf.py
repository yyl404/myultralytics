"""Bridge Past and Future utilities for incremental object detection."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywhn


@dataclass(frozen=True)
class BPFLoss:
    """DwF loss terms."""

    cls: torch.Tensor
    box: torch.Tensor


def normalize_head_output(head: Detect, raw_output, name: str = "output") -> dict[str, torch.Tensor]:
    """Return the Detect head's raw prediction dict from a train- or eval-mode model output.

    Train mode returns the dict directly; eval mode returns a (decoded, preds) tuple. For end2end
    heads the one-to-many branch is used.
    """
    if not isinstance(head, Detect):
        raise TypeError(f"BPF supports Detect heads only, got {type(head)} for {name}")
    preds = raw_output[1] if isinstance(raw_output, tuple) else raw_output
    if not isinstance(preds, dict):
        raise TypeError(f"{name} must be a Detect prediction dict or (decoded, dict) tuple, got {type(preds)}")
    if head.end2end:
        preds = preds["one2many"]
    if not {"boxes", "scores", "feats"} <= preds.keys():
        raise ValueError(f"{name} prediction dict must contain 'boxes', 'scores' and 'feats' keys")
    return preds


def level_tensors_from_preds(head: Detect, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Rebuild per-level (B, no, H, W) head outputs from the concatenated prediction dict.

    The head flattens each level row-major before concatenation, so slicing the anchor axis and
    reshaping back to (H, W) is the exact inverse.
    """
    levels = []
    offset = 0
    batch_size = preds["boxes"].shape[0]
    for feature in preds["feats"]:
        height, width = feature.shape[-2:]
        num_locations = height * width
        boxes = preds["boxes"][:, :, offset : offset + num_locations].reshape(
            batch_size, head.reg_max * 4, height, width
        )
        scores = preds["scores"][:, :, offset : offset + num_locations].reshape(
            batch_size, head.nc, height, width
        )
        levels.append(torch.cat((boxes, scores), dim=1))
        offset += num_locations
    return levels


def bpf_attention_map(feature: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    """Return BPF spatial attention maps with shape (B, H, W) from (B, C, H, W)."""
    if feature.ndim != 4:
        raise ValueError(f"feature must have shape (B, C, H, W), got {tuple(feature.shape)}")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    batch_size, _, height, width = feature.shape
    activation = feature.abs().pow(power).mean(dim=1)
    return (height * width * F.softmax(activation.reshape(batch_size, -1), dim=1)).reshape(
        batch_size, height, width
    )


def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU with shape (N, M) for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)
    return bbox_iou(boxes1[:, None, :], boxes2[None, :, :], xywh=False).squeeze(-1)


def merge_bpf_pseudo_labels(
    batch: dict[str, torch.Tensor],
    detections: list[torch.Tensor],
    iou_low: float = 0.4,
    iou_high: float = 0.7,
    low_weight: float = 1.0,
    high_weight: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Merge old-class detections into a YOLO batch using the official two-IoU-band weighting."""
    if not 0 <= iou_low < iou_high <= 1:
        raise ValueError(f"Expected 0 <= iou_low < iou_high <= 1, got {iou_low}, {iou_high}")
    if low_weight < 0 or high_weight < 0:
        raise ValueError(f"Pseudo-label weights must be non-negative, got {low_weight}, {high_weight}")
    batch_size = batch["img"].shape[0]
    if len(detections) != batch_size:
        raise ValueError(f"Expected {batch_size} detection tensors, got {len(detections)}")

    gt_bboxes = batch["bboxes"]
    gt_cls = batch["cls"]
    gt_batch_idx = batch["batch_idx"].long()
    dtype = gt_bboxes.dtype
    device = gt_bboxes.device
    merged_boxes, merged_cls, merged_batch_idx, merged_weights = [], [], [], []

    for image_index in range(batch_size):
        gt_mask = gt_batch_idx == image_index
        gt_boxes_image = gt_bboxes[gt_mask]
        gt_cls_image = gt_cls[gt_mask]
        pseudo = detections[image_index]
        if pseudo.ndim != 2 or pseudo.shape[1] != 6:
            raise ValueError(f"detections[{image_index}] must have shape (N, 6), got {tuple(pseudo.shape)}")

        pseudo_boxes = pseudo[:, :4].to(device=device, dtype=dtype)
        pseudo_cls = pseudo[:, 5:6].to(device=device, dtype=gt_cls.dtype)
        iou = _pairwise_iou_xyxy(xywh2xyxy(pseudo_boxes), xywh2xyxy(gt_boxes_image))
        low_mask = (iou <= iou_low).all(dim=1)
        middle_mask = ((iou <= iou_high) & (iou > iou_low)).all(dim=1)
        # Preserve the source ordering and its empty-GT behavior: middle-band rows are appended before low-band rows.
        middle_boxes, low_boxes = pseudo_boxes[middle_mask], pseudo_boxes[low_mask]
        middle_cls, low_cls = pseudo_cls[middle_mask], pseudo_cls[low_mask]
        pseudo_boxes = torch.cat((middle_boxes, low_boxes), dim=0)
        pseudo_cls = torch.cat((middle_cls, low_cls), dim=0)
        pseudo_weights = torch.cat(
            (
                torch.full_like(middle_cls.squeeze(1), high_weight, dtype=dtype),
                torch.full_like(low_cls.squeeze(1), low_weight, dtype=dtype),
            ),
            dim=0,
        )

        image_boxes = torch.cat((gt_boxes_image, pseudo_boxes), dim=0)
        image_cls = torch.cat((gt_cls_image, pseudo_cls), dim=0)
        image_weights = torch.cat(
            (torch.ones(len(gt_boxes_image), device=device, dtype=dtype), pseudo_weights), dim=0
        )
        merged_boxes.append(image_boxes)
        merged_cls.append(image_cls)
        merged_weights.append(image_weights)
        merged_batch_idx.append(
            torch.full((len(image_boxes),), image_index, device=device, dtype=gt_batch_idx.dtype)
        )

    batch["bboxes"] = torch.cat(merged_boxes, dim=0)
    batch["cls"] = torch.cat(merged_cls, dim=0)
    batch["batch_idx"] = torch.cat(merged_batch_idx, dim=0)
    batch["bpf_weights"] = torch.cat(merged_weights, dim=0)
    return batch


def bpf_pseudo_detections(
    head: Detect,
    raw_output,
    image_size: tuple[int, int],
    score_threshold: float = 0.75,
    nms_threshold: float = 0.3,
) -> list[torch.Tensor]:
    """Decode old-model outputs into normalized (xywh, confidence, class) pseudo labels."""
    preds = normalize_head_output(head, raw_output, "source output")
    decoded = head._inference(preds)
    predictions = non_max_suppression(
        decoded,
        conf_thres=score_threshold,
        iou_thres=nms_threshold,
        max_det=head.max_det,
        nc=head.nc,
    )
    image_height, image_width = image_size
    output = []
    for prediction in predictions:
        if len(prediction):
            boxes = xyxy2xywhn(prediction[:, :4], w=image_width, h=image_height)
            output.append(torch.cat((boxes, prediction[:, 4:6]), dim=1))
        else:
            output.append(torch.empty((0, 6), device=decoded.device, dtype=decoded.dtype))
    return output


def select_future_ignore_mask(
    head: Detect,
    raw_output,
    head_features: list[torch.Tensor],
    batch: dict[str, torch.Tensor],
    object_topk: float = 0.1,
    attention_topk: float = 0.1,
    iou_threshold: float = 0.1,
    attention_power: float = 2.0,
) -> torch.Tensor:
    """Select high-objectness/high-attention unlabeled locations as future-class ignores."""
    preds = normalize_head_output(head, raw_output, "student output")
    raw_levels = level_tensors_from_preds(head, preds)
    if len(head_features) != head.nl:
        raise ValueError(f"Expected {head.nl} Detect input features, got {len(head_features)}")
    if not 0 < object_topk <= 1 or not 0 < attention_topk <= 1:
        raise ValueError(f"Top-k ratios must be in (0, 1], got {object_topk}, {attention_topk}")

    batch_size = preds["scores"].shape[0]
    decoded_boxes = head._inference(preds)[:, :4, :].permute(0, 2, 1).contiguous()
    decoded_boxes = xywh2xyxy(decoded_boxes)
    level_masks = []
    anchor_offset = 0
    gt_boxes = batch["bboxes"]
    gt_batch_idx = batch["batch_idx"].long()
    image_height, image_width = batch["img"].shape[-2:]
    scale = gt_boxes.new_tensor((image_width, image_height, image_width, image_height))

    for level, (feature, prediction) in enumerate(zip(head_features, raw_levels)):
        if feature.shape[0] != batch_size or feature.shape[-2:] != prediction.shape[-2:]:
            raise ValueError(
                f"Feature/output shape mismatch at level {level}: {tuple(feature.shape)} vs {tuple(prediction.shape)}"
            )
        num_locations = prediction.shape[2] * prediction.shape[3]
        cls_logits = prediction[:, head.reg_max * 4 :, :, :].reshape(batch_size, head.nc, -1)
        objectness = cls_logits.sigmoid().amax(dim=1)
        attention = bpf_attention_map(feature, power=attention_power).reshape(batch_size, -1)
        object_count = max(1, int(num_locations * object_topk))
        attention_count = max(1, int(num_locations * attention_topk))
        object_indices = objectness.topk(object_count, dim=1).indices
        attention_indices = attention.topk(attention_count, dim=1).indices
        candidate = torch.zeros((batch_size, num_locations), device=prediction.device, dtype=torch.bool)
        candidate.scatter_(1, object_indices, True)
        attention_mask = torch.zeros_like(candidate)
        attention_mask.scatter_(1, attention_indices, True)
        candidate &= attention_mask

        level_boxes = decoded_boxes[:, anchor_offset : anchor_offset + num_locations]
        for image_index in range(batch_size):
            gt_image = gt_boxes[gt_batch_idx == image_index]
            if len(gt_image):
                gt_xyxy = xywh2xyxy(gt_image) * scale
                max_iou = _pairwise_iou_xyxy(level_boxes[image_index], gt_xyxy).max(dim=1).values
                candidate[image_index] &= max_iou < iou_threshold
        level_masks.append(candidate)
        anchor_offset += num_locations
    return torch.cat(level_masks, dim=1)


def _categorical_probabilities(head: Detect, raw_output) -> torch.Tensor:
    """Return categorical [background, classes] probabilities with shape (B, A, C+1)."""
    preds = normalize_head_output(head, raw_output, "DwF output")
    logits = preds["scores"].permute(0, 2, 1)  # (B, A, C) raw cls logits, anchors in level order
    background = torch.zeros((*logits.shape[:2], 1), device=logits.device, dtype=logits.dtype)
    return torch.cat((background, logits), dim=-1).float().softmax(dim=-1)


def build_dwf_targets(
    source_probs: torch.Tensor, interim_probs: torch.Tensor, old_region_mask: torch.Tensor
) -> torch.Tensor:
    """Reconstruct the official DwF categorical target distribution."""
    if source_probs.ndim != 3 or interim_probs.ndim != 3:
        raise ValueError("source_probs and interim_probs must have shape (B, A, C+1)")
    if source_probs.shape[:2] != interim_probs.shape[:2] or old_region_mask.shape != source_probs.shape[:2]:
        raise ValueError(
            f"DwF batch/anchor shapes disagree: {source_probs.shape}, {interim_probs.shape}, "
            f"{old_region_mask.shape}"
        )
    eps = torch.finfo(torch.float32).eps
    source = source_probs.float()
    interim = interim_probs.float()
    scaled_interim = interim * (
        source[..., :1] / interim.sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    old_target = torch.cat((scaled_interim[..., :1], source[..., 1:], scaled_interim[..., 1:]), dim=-1)
    scaled_source = source * (
        interim[..., :1] / source.sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    new_target = torch.cat((scaled_source, interim[..., 1:]), dim=-1)
    return torch.where(old_region_mask.unsqueeze(-1), old_target, new_target)


def _decoded_boxes(head: Detect, raw_output) -> torch.Tensor:
    """Return decoded xyxy boxes with shape (B, A, 4)."""
    preds = normalize_head_output(head, raw_output, "box output")
    return xywh2xyxy(head._inference(preds)[:, :4, :].permute(0, 2, 1).contiguous())


def compute_dwf_loss(
    student_head: Detect,
    student_output,
    source_head: Detect,
    source_output,
    interim_head: Detect,
    interim_output,
    batch: dict[str, torch.Tensor],
    proposal_topk: int = 128,
    proposal_samples: int = 64,
    split_iou: float = 0.5,
) -> BPFLoss:
    """Compute Distillation with Future using official proposal sampling and old/new branches."""
    if student_head.nc != source_head.nc + interim_head.nc:
        raise ValueError(
            f"Student classes must equal old+new classes, got {student_head.nc}, "
            f"{source_head.nc}, {interim_head.nc}"
        )
    source_probs = _categorical_probabilities(source_head, source_output)
    interim_probs = _categorical_probabilities(interim_head, interim_output)
    student_probs = _categorical_probabilities(student_head, student_output)
    if source_probs.shape[:2] != interim_probs.shape[:2] or source_probs.shape[:2] != student_probs.shape[:2]:
        raise ValueError("DwF requires source, interim, and student to share dense prediction locations")
    num_anchors = source_probs.shape[1]
    if proposal_topk <= 0 or proposal_samples <= 0 or proposal_samples > proposal_topk:
        raise ValueError(
            f"DwF requires 0 < proposal_samples <= proposal_topk, got "
            f"{proposal_samples}, {proposal_topk}"
        )

    source_boxes = _decoded_boxes(source_head, source_output)
    interim_boxes = _decoded_boxes(interim_head, interim_output)
    student_boxes = _decoded_boxes(student_head, student_output)
    source_objectness = source_probs[..., 1:].amax(dim=-1)
    top_count = min(num_anchors, proposal_topk)
    sample_count = min(num_anchors, proposal_samples)
    top_indices = source_objectness.topk(top_count, dim=1).indices
    sampled_positions = torch.tensor(
        [random.sample(range(top_count), sample_count) for _ in range(source_probs.shape[0])],
        device=source_probs.device,
        dtype=torch.long,
    )
    selected = top_indices.gather(1, sampled_positions)
    gather_box = selected.unsqueeze(-1).expand(-1, -1, 4)
    selected_source_boxes = source_boxes.gather(1, gather_box)

    batch_size = source_probs.shape[0]
    old_region = torch.ones((batch_size, sample_count), device=selected.device, dtype=torch.bool)
    gt_boxes = batch["bboxes"]
    gt_batch_idx = batch["batch_idx"].long()
    image_height, image_width = batch["img"].shape[-2:]
    scale = gt_boxes.new_tensor((image_width, image_height, image_width, image_height))
    for image_index in range(batch_size):
        gt_image = gt_boxes[gt_batch_idx == image_index]
        if len(gt_image):
            gt_xyxy = xywh2xyxy(gt_image) * scale
            max_iou = _pairwise_iou_xyxy(selected_source_boxes[image_index], gt_xyxy).max(dim=1).values
            old_region[image_index] = max_iou <= split_iou

    source_selected = source_probs.gather(
        1, selected.unsqueeze(-1).expand(-1, -1, source_probs.shape[-1])
    )
    interim_selected = interim_probs.gather(
        1, selected.unsqueeze(-1).expand(-1, -1, interim_probs.shape[-1])
    )
    student_selected = student_probs.gather(
        1, selected.unsqueeze(-1).expand(-1, -1, student_probs.shape[-1])
    )
    targets = build_dwf_targets(source_selected, interim_selected, old_region)
    cls_loss = -(targets * student_selected.float().clamp_min(torch.finfo(torch.float32).eps).log()).mean()

    source_selected_boxes = source_boxes.gather(1, gather_box) / scale
    interim_selected_boxes = interim_boxes.gather(1, gather_box) / scale
    student_selected_boxes = student_boxes.gather(1, gather_box) / scale
    teacher_boxes = torch.where(old_region.unsqueeze(-1), source_selected_boxes, interim_selected_boxes)
    box_loss = F.mse_loss(
        student_selected_boxes.float(), teacher_boxes.float(), reduction="none"
    ).sum(dim=-1).mean()
    return BPFLoss(cls=cls_loss.to(student_probs.dtype), box=box_loss.to(student_probs.dtype))
