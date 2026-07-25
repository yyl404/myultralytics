"""Tests for Bridge Past and Future primitives."""

import random

import torch

from ultralytics.engine.bpf import (
    bpf_attention_map,
    build_dwf_targets,
    compute_dwf_loss,
    merge_bpf_pseudo_labels,
    select_future_ignore_mask,
)
from ultralytics.nn.modules.head import Detect


def _raw_output(head: Detect, height: int, width: int, requires_grad: bool = False) -> list[torch.Tensor]:
    output = torch.zeros((1, head.no, height, width), requires_grad=requires_grad)
    output.data[:, : head.reg_max * 4] = 1.0
    return [output]


def _head(num_classes: int) -> Detect:
    head = Detect(nc=num_classes, ch=(8,))
    head.stride = torch.tensor([8.0])
    return head


def test_bpf_attention_map_matches_official_formula():
    feature = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    attention = bpf_attention_map(feature, power=2.0)
    activation = feature.abs().pow(2).mean(dim=1)
    expected = 2 * torch.softmax(activation.reshape(1, -1), dim=1).reshape(1, 1, 2)
    torch.testing.assert_close(attention, expected)
    torch.testing.assert_close(attention.sum(), torch.tensor(2.0))


def test_merge_bpf_pseudo_labels_preserves_two_iou_bands():
    batch = {
        "img": torch.zeros((1, 3, 32, 32)),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "cls": torch.tensor([[2.0]]),
        "batch_idx": torch.tensor([0]),
    }
    detections = [
        torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1, 0.9, 0.0],  # IoU 0: full weight
                [0.5, 0.5, 0.3, 0.3, 0.9, 1.0],  # IoU 4/9: reduced weight
                [0.5, 0.5, 0.2, 0.2, 0.9, 1.0],  # IoU 1: rejected
            ]
        )
    ]
    merged = merge_bpf_pseudo_labels(batch, detections)
    assert merged["bboxes"].shape == (3, 4)
    torch.testing.assert_close(merged["bpf_weights"], torch.tensor([1.0, 0.3, 1.0]))
    torch.testing.assert_close(merged["cls"].squeeze(1), torch.tensor([2.0, 1.0, 0.0]))


def test_merge_bpf_pseudo_labels_preserves_source_empty_gt_duplication():
    batch = {
        "img": torch.zeros((1, 3, 32, 32)),
        "bboxes": torch.empty((0, 4)),
        "cls": torch.empty((0, 1)),
        "batch_idx": torch.empty((0,), dtype=torch.long),
    }
    detections = [torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 1.0]])]
    merged = merge_bpf_pseudo_labels(batch, detections)
    assert merged["bboxes"].shape == (2, 4)
    torch.testing.assert_close(merged["bpf_weights"], torch.tensor([0.3, 1.0]))


def test_build_dwf_targets_uses_old_and_new_background_scaling():
    source = torch.tensor([[[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]]])
    interim = torch.tensor([[[0.25, 0.75], [0.6, 0.4]]])
    mask = torch.tensor([[True, False]])
    target = build_dwf_targets(source, interim, mask)
    expected_old = torch.tensor([0.1, 0.3, 0.3, 0.3])
    expected_new = torch.tensor([0.12, 0.3, 0.18, 0.4])
    torch.testing.assert_close(target[0, 0], expected_old)
    torch.testing.assert_close(target[0, 1], expected_new)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((1, 2)))


def test_select_future_ignore_mask_uses_topk_intersection():
    head = _head(2)
    raw = _raw_output(head, 2, 2)
    cls_start = head.reg_max * 4
    raw[0][0, cls_start, 0, 1] = 10.0
    feature = torch.zeros((1, 8, 2, 2))
    feature[0, :, 0, 1] = 10.0
    batch = {
        "img": torch.zeros((1, 3, 16, 16)),
        "bboxes": torch.empty((0, 4)),
        "cls": torch.empty((0, 1)),
        "batch_idx": torch.empty((0,), dtype=torch.long),
    }
    mask = select_future_ignore_mask(
        head=head,
        raw_output=raw,
        head_features=[feature],
        batch=batch,
        object_topk=0.25,
        attention_topk=0.25,
    )
    assert mask.shape == (1, 4)
    assert mask.sum().item() == 1
    assert mask[0, 1]


def test_compute_dwf_loss_samples_official_64_of_top_128():
    random.seed(42)
    source_head = _head(2)
    interim_head = _head(1)
    student_head = _head(3)
    source = _raw_output(source_head, 8, 16)
    interim = _raw_output(interim_head, 8, 16)
    student = _raw_output(student_head, 8, 16, requires_grad=True)
    batch = {
        "img": torch.zeros((1, 3, 64, 128)),
        "bboxes": torch.empty((0, 4)),
        "cls": torch.empty((0, 1)),
        "batch_idx": torch.empty((0,), dtype=torch.long),
    }
    losses = compute_dwf_loss(
        student_head=student_head,
        student_output=student,
        source_head=source_head,
        source_output=source,
        interim_head=interim_head,
        interim_output=interim,
        batch=batch,
    )
    assert torch.isfinite(losses.cls)
    assert torch.isfinite(losses.box)
    (losses.cls + losses.box).backward()
    assert student[0].grad is not None
