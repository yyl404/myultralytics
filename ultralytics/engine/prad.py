"""Prototype-reliability-aware YOLO LwF distillation on real incremental images.

The teacher's old-class output is split into reliable foreground, reliable background
and an uncertain band.  Foreground reliability multiplies the teacher confidence by a
prototype-match factor computed from the detection-head input patch, so a teacher
response that fires on a new-class object (or on new-domain background) is down-weighted
instead of being copied.  Background responses are distilled from teacher confidence
only, and regression is distilled on reliable foreground only.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from ultralytics.engine.lwf import _decode_boxes
from ultralytics.engine.patch_proto import (
    EPS,
    anchor_levels,
    extract_patches,
    level_offsets,
    level_shapes,
    valid_patch_positions,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


def binary_kl(student_logits, teacher_probs):
    """Return the per-element Bernoulli KL ``KL(p || q)`` with ``q = sigmoid(student_logits)``.

    Computed in float32 from the student logits (``logsigmoid`` keeps both tails exact),
    so the reported value matches the method definition instead of dropping the
    teacher entropy term.
    """
    probs = teacher_probs.to(torch.float32).clamp(EPS, 1.0 - EPS)
    log_q = F.logsigmoid(student_logits.to(torch.float32))
    log_one_minus_q = F.logsigmoid(-student_logits.to(torch.float32))
    return probs * (probs.log() - log_q) + (1.0 - probs) * ((1.0 - probs).log() - log_one_minus_q)


class PradLossTerms(NamedTuple):
    """Terms of one prototype-reliability-aware distillation step."""

    cls_fg: torch.Tensor
    cls_bg: torch.Tensor
    cls: torch.Tensor
    reg: torch.Tensor


class PrototypeReliabilityLwFLoss:
    """Prototype-reliability-aware old-class distillation for one student/teacher pair.

    Args:
        subspace (HeadPatchSubspace): Fixed shared patch codec.
        bank (MultiCenterBank): Old-class prototype bank in that codec.
        history_nc (int): Number of old-class channels of the teacher head.
        fg_threshold (float): ``tau_F`` on the teacher's maximum old-class probability.
        bg_threshold (float): ``tau_B`` on the same quantity, ``tau_B < tau_F``.
        conf_power (float): ``gamma`` on the teacher confidence weight.
        proto_threshold (float): Center of the prototype-reliability sigmoid.
        proto_temperature (float): Width of that sigmoid.
        bg_weight (float): ``eta_B`` on the background term.
        split_normalize (bool): Normalize foreground and background separately.
        max_fg (int): Safety cap on foreground candidates per step (highest confidence kept).
    """

    def __init__(
        self,
        subspace,
        bank,
        history_nc,
        fg_threshold=0.5,
        bg_threshold=0.1,
        conf_power=1.0,
        proto_threshold=0.5,
        proto_temperature=0.1,
        bg_weight=1.0,
        split_normalize=True,
        max_fg=512,
    ):
        """Validate the distillation settings and bind the prototype reader."""
        if history_nc < 1:
            raise ValueError(f"history_nc must be >= 1, got {history_nc}")
        if not bg_threshold < fg_threshold:
            raise ValueError(f"bg_threshold ({bg_threshold}) must be below fg_threshold ({fg_threshold})")
        if proto_temperature <= 0.0:
            raise ValueError(f"proto_temperature must be positive, got {proto_temperature}")
        if conf_power < 0.0:
            raise ValueError(f"conf_power must be non-negative, got {conf_power}")
        if max_fg < 1:
            raise ValueError(f"max_fg must be >= 1, got {max_fg}")
        if bank.num_levels != subspace.num_levels:
            raise ValueError(f"Bank has {bank.num_levels} levels but subspace has {subspace.num_levels}")
        self.subspace = subspace
        self.bank = bank
        self.history_nc = history_nc
        self.fg_threshold = float(fg_threshold)
        self.bg_threshold = float(bg_threshold)
        self.conf_power = float(conf_power)
        self.proto_threshold = float(proto_threshold)
        self.proto_temperature = float(proto_temperature)
        self.bg_weight = float(bg_weight)
        self.split_normalize = bool(split_normalize)
        self.max_fg = int(max_fg)
        self._missing_warned = set()
        self._level_cache = None

    def _prototype_reliability(self, teacher_feats, candidate_levels, images, rows, cols, classes, shapes):
        """Return the prototype-match reliability ``g_k`` for the candidate anchors."""
        reliability = torch.zeros(
            candidate_levels.shape[0], dtype=torch.float32, device=candidate_levels.device
        )
        for level in torch.unique(candidate_levels).tolist():
            select = candidate_levels == level
            height, width = shapes[level]
            level_rows = rows[select]
            level_cols = cols[select]
            inside = valid_patch_positions(level_rows, level_cols, height, width, self.subspace.radius)
            if not bool(inside.any()):
                continue
            patches = extract_patches(
                teacher_feats,
                level,
                images[select][inside],
                level_rows[inside],
                level_cols[inside],
                self.subspace.radius,
            )
            codes = self.subspace.encode(level, patches)
            similarity, has_prototype = self.bank.max_cosine(level, classes[select][inside], codes)
            for class_id in torch.unique(classes[select][inside][~has_prototype]).tolist():
                key = (int(class_id), int(level))
                if key not in self._missing_warned:
                    self._missing_warned.add(key)
                    LOGGER.warning(
                        f"No patch prototype for class {class_id} at level {level}; "
                        f"its foreground distillation weight is zero"
                    )
            gate = torch.sigmoid((similarity - self.proto_threshold) / self.proto_temperature)
            gate = gate * has_prototype.to(gate.dtype)
            level_positions = select.nonzero().flatten()[inside]
            reliability[level_positions] = gate.to(reliability.dtype)
        return reliability

    def __call__(self, student, teacher, new_class_mask, stride):
        """Return the foreground/background classification and regression terms.

        Args:
            student (dict): Student Detect output with ``boxes``/``scores``/``feats``.
            teacher (dict): Teacher Detect output with the same layout but only old classes.
            new_class_mask (torch.Tensor): ``(B, A)`` bool, anchors owned by a current-task GT.
            stride (torch.Tensor): Per-level strides of the head.

        Returns:
            (PradLossTerms): ``cls_fg``, ``cls_bg``, the composed ``cls`` and ``reg`` terms.
        """
        student_boxes, student_scores = student["boxes"], student["scores"]
        teacher_boxes, teacher_scores, teacher_feats = teacher["boxes"], teacher["scores"], teacher["feats"]
        if student_boxes.shape != teacher_boxes.shape:
            raise RuntimeError(
                f"Distillation box mismatch: student {tuple(student_boxes.shape)}, "
                f"teacher {tuple(teacher_boxes.shape)}"
            )
        if teacher_scores.shape[1] != self.history_nc:
            raise RuntimeError(
                f"Teacher must expose {self.history_nc} old-class channels, got {teacher_scores.shape[1]}"
            )
        expected_mask = (teacher_scores.shape[0], teacher_scores.shape[2])
        if tuple(new_class_mask.shape) != expected_mask:
            raise RuntimeError(
                f"new_class_mask must be shaped {expected_mask} like the anchor axis, "
                f"got {tuple(new_class_mask.shape)}"
            )
        shapes = level_shapes(student["feats"])
        if level_shapes(teacher_feats) != shapes:
            raise RuntimeError("Student and teacher head inputs disagree on the level shapes")
        anchors, _ = make_anchors(student["feats"], stride, 0.5)

        teacher_prob = teacher_scores.to(torch.float32).sigmoid()  # (B, nc_old, A)
        confidence, predicted = teacher_prob.amax(dim=1), teacher_prob.argmax(dim=1)  # (B, A)
        foreground = (confidence >= self.fg_threshold) & ~new_class_mask
        background = (confidence <= self.bg_threshold) & ~new_class_mask

        # Regression-agreement weight A_k, detached from both sides.
        with torch.no_grad():
            student_xyxy = _decode_boxes(student_boxes.detach(), 1, anchors)
            teacher_xyxy = _decode_boxes(teacher_boxes.detach(), 1, anchors)
            agreement = bbox_iou(student_xyxy, teacher_xyxy, xywh=False).clamp(0.0, 1.0).reshape(confidence.shape)

        # Prototype reliability g_k on the foreground candidates only.
        candidates = foreground.nonzero(as_tuple=False)
        if candidates.shape[0] > self.max_fg:
            keep = confidence[candidates[:, 0], candidates[:, 1]].topk(self.max_fg).indices
            candidates = candidates[keep]
        reliability = confidence.new_zeros(confidence.shape)
        if candidates.shape[0] > 0:
            images, flat_anchors = candidates[:, 0], candidates[:, 1]
            if self._level_cache is None or self._level_cache[0] != shapes:
                self._level_cache = (shapes, anchor_levels(shapes, confidence.device))
            levels = self._level_cache[1][flat_anchors]
            offsets = torch.tensor(level_offsets(shapes), device=confidence.device)
            width_of_level = torch.tensor([width for _, width in shapes], device=confidence.device)
            local = flat_anchors - offsets[levels]
            rows, cols = local // width_of_level[levels], local % width_of_level[levels]
            gate = self._prototype_reliability(
                teacher_feats,
                levels,
                images,
                rows,
                cols,
                predicted[images, flat_anchors],
                shapes,
            )
            reliability[images, flat_anchors] = gate.to(reliability.dtype)

        weight_fg = foreground.to(torch.float32) * confidence.pow(self.conf_power) * reliability * agreement
        weight_bg = background.to(torch.float32) * (1.0 - confidence).pow(self.conf_power)

        kl = binary_kl(student_scores[:, : self.history_nc], teacher_prob)  # (B, nc_old, A)
        fg_numerator = (kl * weight_fg.unsqueeze(1)).sum()
        fg_denominator = self.history_nc * weight_fg.sum()
        bg_numerator = (kl * weight_bg.unsqueeze(1)).sum()
        bg_denominator = self.history_nc * weight_bg.sum()

        if self.split_normalize:
            cls_fg = fg_numerator / (fg_denominator + EPS)
            cls_bg = bg_numerator / (bg_denominator + EPS)
            cls = cls_fg + self.bg_weight * cls_bg
        else:
            cls = (fg_numerator + bg_numerator) / (fg_denominator + bg_denominator + EPS)
            cls_fg, cls_bg = cls, cls.new_zeros(())

        distance = F.smooth_l1_loss(
            student_boxes.to(torch.float32), teacher_boxes.to(torch.float32), reduction="none"
        ).mean(dim=1)  # (B, A), mean over the four box sides
        reg = (distance * weight_fg).sum() / (weight_fg.sum() + EPS)
        return PradLossTerms(cls_fg=cls_fg, cls_bg=cls_bg, cls=cls, reg=reg)
