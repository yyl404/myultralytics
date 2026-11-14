"""All-class patch replay distillation from historical prototypes.

Reconstructed head-input patches are pushed through the teacher and the student
detection heads.  The teacher supplies soft old-class responses; the new-class channels
are set to probability zero so the teacher target covers all classes.  The student's
all-class classification and regression outputs are then distilled on the same patch.

The replay patches are synthetic, so the head branches run with their BatchNorm running
statistics (evaluation mode) and their training flags are restored afterwards; otherwise
a synthetic patch batch would overwrite the head statistics that the deployed model uses.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ultralytics.engine.patch_proto import EPS
from ultralytics.engine.prad import binary_kl
from ultralytics.utils import LOGGER


@contextmanager
def _evaluation_mode(modules):
    """Temporarily evaluate `modules` and restore every training flag afterwards."""
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, state in zip(modules, states):
            module.train(state)


class PrdLossTerms(NamedTuple):
    """Terms of one patch replay distillation step."""

    cls: torch.Tensor
    reg: torch.Tensor
    num_patches: int


class PatchReplayDistillation:
    """Distill the student's all-class head output on historical prototype patches.

    Args:
        student_head (Detect): Student detection head.
        teacher_head (Detect): Frozen previous-task detection head with old classes only.
        subspace (HeadPatchSubspace): Fixed shared patch codec of the prototype bank.
        bank (MultiCenterBank): Old-class prototype bank in that codec.
        samples (int): Patch draws per level per iteration.
        std_scale (float): Multiplier on the local diagonal standard deviation.
        smooth_l1_beta (float): Transition point of the regression SmoothL1.
        seed (int): Generator seed; ranks stay independent because it is built per rank.
    """

    def __init__(self, student_head, teacher_head, subspace, bank, samples=64, std_scale=1.0, smooth_l1_beta=1.0, seed=0):
        """Validate the head/artifact pairing and build the per-rank sampler."""
        if student_head.reg_max != 1 or teacher_head.reg_max != 1:
            raise ValueError(
                f"Patch replay distillation targets direct-regression heads (reg_max=1), got student "
                f"{student_head.reg_max} and teacher {teacher_head.reg_max}"
            )
        if student_head.nl != teacher_head.nl:
            raise ValueError(f"Head level mismatch: student {student_head.nl}, teacher {teacher_head.nl}")
        if bank.num_levels != student_head.nl or subspace.num_levels != student_head.nl:
            raise ValueError(
                f"Artifact has {bank.num_levels} bank levels and {subspace.num_levels} codec levels, "
                f"but the head has {student_head.nl}"
            )
        if teacher_head.nc >= student_head.nc:
            raise ValueError(
                f"Teacher must hold fewer class channels than the student, got {teacher_head.nc} vs {student_head.nc}"
            )
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        self.student_head = student_head
        self.teacher_head = teacher_head
        self.subspace = subspace
        self.bank = bank
        self.samples = int(samples)
        self.std_scale = float(std_scale)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.history_nc = int(teacher_head.nc)
        self.num_classes = int(student_head.nc)
        self.center = subspace.radius
        self._generator = torch.Generator().manual_seed(int(seed))
        self._empty_warned = False

    def _head_outputs(self, level, patches):
        """Return teacher old-class logits/boxes and student all-class logits/boxes on one patch batch."""
        student_cls_branch = self.student_head.cv3[level]
        student_box_branch = self.student_head.cv2[level]
        with torch.no_grad():
            teacher_logits = self.teacher_head.cv3[level](patches)[:, :, self.center, self.center]
            teacher_boxes = self.teacher_head.cv2[level](patches)[:, :, self.center, self.center]
        with _evaluation_mode([student_cls_branch, student_box_branch]):
            student_logits = student_cls_branch(patches)[:, :, self.center, self.center]
            student_boxes = student_box_branch(patches)[:, :, self.center, self.center]
        return teacher_logits, teacher_boxes, student_logits, student_boxes

    def compute_loss(self):
        """Return the all-class classification and regression replay terms of this step."""
        class_numerator = None
        regression_numerator = None
        class_denominator = 0
        regression_denominator = 0
        num_patches = 0
        for level in range(self.student_head.nl):
            codes = self.bank.sample(level, self.samples, self._generator, self.std_scale)
            if codes is None:
                continue
            patches = self.subspace.decode(level, codes)
            teacher_logits, teacher_boxes, student_logits, student_boxes = self._head_outputs(level, patches)
            # All-class teacher target: soft old-class probabilities, probability zero for new classes.
            target = torch.cat(
                [
                    teacher_logits.to(torch.float32).sigmoid().clamp(EPS, 1.0 - EPS),
                    torch.zeros(
                        teacher_logits.shape[0],
                        self.num_classes - self.history_nc,
                        device=teacher_logits.device,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
            kl = binary_kl(student_logits, target)
            class_numerator = kl.sum() if class_numerator is None else class_numerator + kl.sum()
            class_denominator += int(kl.numel())
            distance = F.smooth_l1_loss(
                student_boxes.to(torch.float32),
                teacher_boxes.to(torch.float32),
                reduction="sum",
                beta=self.smooth_l1_beta,
            )
            regression_numerator = distance if regression_numerator is None else regression_numerator + distance
            regression_denominator += int(student_boxes.numel())
            num_patches += int(codes.shape[0])
        if num_patches == 0:
            if not self._empty_warned:
                LOGGER.warning("Patch replay distillation found no prototyped class; its loss stays zero")
                self._empty_warned = True
            zero = torch.zeros((), device=self.student_head.cv3[0][0].conv.weight.device)
            return PrdLossTerms(cls=zero, reg=zero, num_patches=0)
        return PrdLossTerms(
            cls=class_numerator / float(class_denominator),
            reg=regression_numerator / float(regression_denominator),
            num_patches=num_patches,
        )
