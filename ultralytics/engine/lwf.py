"""YOLO LwF self-distillation for class-incremental object detection.

Two self-distillation variants share this module:

- ``yolo_lwf``: the YOLO LwF loss. The regression output is a categorical DFL
  distribution per anchor side, so it is distilled with a temperature-scaled
  cross entropy instead of a logit L2, and each anchor cross entropy is weighted
  by the teacher's objectness (``wce``). The classification output is distilled
  with a binary cross entropy against the teacher probabilities, weighted per
  anchor by the agreement of the two decoded boxes (``cls_iou``). Both terms
  only cover the historical class channels, so the freshly initialized
  current-task channels never drag the student back to its random initialization.
- ``lwf``: the plain LwF baseline. It matches the Detect-head input features of
  the last backbone/neck layers with an L2 loss, truncated to the first
  ``4 * reg_max + history_nc`` channels, which is what the baseline in
  "Teach YOLO to Remember" uses for comparison.

All loss terms are means over the batch (matching the reference implementation),
so a caller that scales the detection criterion by the batch size must scale
these terms the same way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ("YOLOLwFLoss", "detect_branch_output", "feature_l2_loss")

_EPS = 1e-6


def detect_branch_output(output, branch="one2many"):
    """Return the raw Detect-head output dict of the branch the student trains on.

    Args:
        output (dict | tuple): Train-mode prediction dict, which may hold the
            ``one2many``/``one2one`` branches of an end-to-end head, or an
            eval-mode ``(decoded, preds)`` tuple.
        branch (str): Branch key extracted from an end-to-end prediction dict.

    Returns:
        (dict[str, torch.Tensor]): ``boxes`` (B, 4 * reg_max, A), ``scores``
            (B, nc, A) and ``feats`` (list of (B, C, H, W) head inputs).
    """
    if isinstance(output, tuple):
        if len(output) != 2 or not isinstance(output[1], dict):
            raise TypeError(f"Expected an eval-mode (decoded, preds) tuple, got a {len(output)}-element tuple")
        output = output[1]
    if not isinstance(output, dict):
        raise TypeError(f"Expected a Detect prediction dict, got {type(output)}")
    if branch in output and isinstance(output[branch], dict):
        output = output[branch]
    missing = {"boxes", "scores", "feats"} - set(output)
    if missing:
        raise KeyError(f"Detect prediction dict is missing {sorted(missing)}")
    return output


def _decode_boxes(boxes, reg_max, anchors):
    """Decode raw box outputs into xyxy boxes in stride units.

    Args:
        boxes (torch.Tensor): Box head output with shape (B, 4 * reg_max, A).
        reg_max (int): DFL bins per side; ``1`` means the head predicts distances directly.
        anchors (torch.Tensor): Anchor points with shape (A, 2).

    Returns:
        (torch.Tensor): Boxes with shape (B, A, 4) in xyxy stride units.
    """
    batch, _, num_anchors = boxes.shape
    if reg_max > 1:
        proj = torch.arange(reg_max, device=boxes.device, dtype=torch.float32)
        dist = boxes.view(batch, 4, reg_max, num_anchors).float().softmax(2).mul(proj.view(1, 1, -1, 1)).sum(2)
    else:
        dist = boxes.float()
    return dist2bbox(dist.permute(0, 2, 1), anchors, xywh=False, dim=-1)


class YOLOLwFLoss:
    """YOLO LwF distillation terms for one teacher/student output pair.

    Args:
        reg_max (int): DFL bins per box side of the student head.
        history_nc (int): Number of class channels owned by the previous tasks.
        temperature (float): Regression cross-entropy temperature.
        wce (bool): Weight each regression cross entropy by the teacher objectness.
        cls_iou (bool): Weight each classification loss by the box agreement of teacher and student.
    """

    def __init__(self, reg_max: int, history_nc: int, temperature: float = 2.0, wce: bool = True, cls_iou: bool = True):
        """Validate the distillation settings and bind them."""
        if reg_max < 1:
            raise ValueError(f"reg_max must be >= 1, got {reg_max}")
        if history_nc < 1:
            raise ValueError(f"history_nc must be >= 1, got {history_nc}")
        if not 0.0 < float(temperature):
            raise ValueError(f"lwf_temperature must be positive, got {temperature}")
        self.reg_max = reg_max
        self.history_nc = history_nc
        self.temperature = float(temperature)
        self.wce = wce
        self.cls_iou = cls_iou

    def __call__(self, student, teacher, stride):
        """Compute the classification and regression distillation terms.

        Args:
            student (dict): Student Detect output, ``boxes`` (B, 4 * reg_max, A) and
                ``scores`` (B, nc, A) plus ``feats`` for the anchor grid.
            teacher (dict): Teacher Detect output with the same box/feature layout and
                exactly ``history_nc`` class channels.
            stride (torch.Tensor): Per-level strides of the head, shape (3,).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Classification and regression terms, both means.
        """
        student_boxes = student["boxes"]
        teacher_boxes = teacher["boxes"]
        student_scores = student["scores"]
        teacher_scores = teacher["scores"]
        if student_boxes.shape != teacher_boxes.shape:
            raise RuntimeError(
                f"LwF box output mismatch: student {tuple(student_boxes.shape)}, "
                f"teacher {tuple(teacher_boxes.shape)} (reg_max/anchors differ)"
            )
        if teacher_scores.shape[1] != self.history_nc:
            raise RuntimeError(
                f"LwF teacher must expose {self.history_nc} history class channels, got {teacher_scores.shape[1]}"
            )
        if student_scores.shape[1] < self.history_nc:
            raise RuntimeError(
                f"LwF student has {student_scores.shape[1]} class channels, fewer than the "
                f"{self.history_nc} history channels"
            )
        if student_boxes.shape[-1] == 0:
            device = student_boxes.device
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return zero, zero

        anchors, _ = make_anchors(student["feats"], stride, 0.5)  # (A, 2)
        teacher_scores = teacher_scores.float().detach()
        teacher_prob = teacher_scores.sigmoid()  # (B, history_nc, A)

        # Cls-IoU weight: (DIoU + 1) / 2 per anchor, detached from both sides.
        if self.cls_iou:
            with torch.no_grad():
                student_xyxy = _decode_boxes(student_boxes.detach(), self.reg_max, anchors)
                teacher_xyxy = _decode_boxes(teacher_boxes.detach(), self.reg_max, anchors)
                overlap = (bbox_iou(student_xyxy, teacher_xyxy, xywh=False, DIoU=True) + 1.0) / 2.0
                # Degenerate (inverted) boxes can push DIoU below -1; a negative weight
                # would reward disagreement, so keep the weight in [0, 1].
                overlap = overlap.clamp(0.0, 1.0)
            cls_weight = overlap.reshape(overlap.shape[0], 1, -1)  # (B, 1, A)
        else:
            cls_weight = 1.0

        student_hist = student_scores[:, : self.history_nc].float()
        bce = F.binary_cross_entropy_with_logits(student_hist, teacher_prob, reduction="none")  # (B, history_nc, A)
        cls_loss = (bce * cls_weight).mean()

        # Prediction-wise weight: the teacher's objectness (max historical class score).
        if self.wce:
            anchor_weight = teacher_prob.amax(dim=1)  # (B, A)
        else:
            anchor_weight = 1.0

        if self.reg_max > 1:
            temperature = self.temperature
            batch, _, num_anchors = student_boxes.shape
            teacher_dist = teacher_boxes.float().detach().view(batch, 4, self.reg_max, num_anchors)
            student_dist = student_boxes.float().view(batch, 4, self.reg_max, num_anchors)
            target = (teacher_dist / temperature).softmax(dim=2)
            log_prob = (student_dist / temperature).log_softmax(dim=2)
            ce = -(target * log_prob).sum(dim=2)  # (B, 4, A)
            weight = anchor_weight.unsqueeze(1) if self.wce else 1.0
            reg_loss = (ce * weight).mean()
        else:
            # No DFL distribution to match: distill the four predicted side distances.
            distance = (student_boxes.float() - teacher_boxes.float().detach()).abs()  # (B, 4, A)
            weight = anchor_weight.unsqueeze(1) if self.wce else 1.0
            reg_loss = (distance * weight).mean()
        return cls_loss, reg_loss


def feature_l2_loss(student_feats, teacher_feats, keep_channels):
    """L2 distillation of the Detect-head input features (plain LwF baseline).

    Args:
        student_feats (list[torch.Tensor]): Student head inputs, one (B, C, H, W) tensor per level.
        teacher_feats (list[torch.Tensor]): Teacher head inputs with the same layout.
        keep_channels (int): Number of leading channels compared; clamped to the level width.

    Returns:
        (torch.Tensor): Mean squared error averaged over the levels.
    """
    if len(student_feats) != len(teacher_feats):
        raise RuntimeError(f"LwF feature level mismatch: student {len(student_feats)}, teacher {len(teacher_feats)}")
    if keep_channels < 1:
        raise ValueError(f"keep_channels must be >= 1, got {keep_channels}")
    losses = []
    for level, (student, teacher) in enumerate(zip(student_feats, teacher_feats)):
        if student.shape != teacher.shape:
            raise RuntimeError(
                f"LwF feature shape mismatch at level {level}: "
                f"student {tuple(student.shape)}, teacher {tuple(teacher.shape)}"
            )
        width = min(keep_channels, student.shape[1])
        losses.append(F.mse_loss(student[:, :width].float(), teacher[:, :width].float().detach()))
    if not losses:
        raise RuntimeError("LwF feature distillation received no feature levels")
    return torch.stack(losses).mean()
