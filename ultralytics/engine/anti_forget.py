import numpy as np
import warnings
from copy import deepcopy
from pathlib import Path
import math
import time
import joblib

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.nn import functional as F

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    colorstr,
)
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    unwrap_model,
    unset_deterministic,
)

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywhn, xywh2xyxy
from ultralytics.utils.metrics import bbox_iou, batch_probiou
from ultralytics.engine.espreg import ESPRegLoss
from ultralytics.engine.bpf import (
    bpf_pseudo_detections,
    compute_dwf_loss,
    merge_bpf_pseudo_labels,
    select_future_ignore_mask,
)
from ultralytics.engine.ewc import EWCLoss, load_ewc_state
from ultralytics.engine.l2 import L2Loss
from ultralytics.engine.lwf import YOLOLwFLoss, detect_branch_output, feature_l2_loss
from ultralytics.engine.nsgp import NSGP
from ultralytics.engine.patch_proto import (
    RecordingDetectionLoss,
    get_assignment,
    head_input_channels,
    head_patch_radius,
    load_patch_artifact,
    new_class_mask,
)
from ultralytics.engine.prad import PrototypeReliabilityLwFLoss
from ultralytics.engine.prd import PatchReplayDistillation
from ultralytics.engine.repre import RegionalPrototypeReplay
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.nn.modules.head import Detect, OBB


def get_model_raw_output(batch, model, device):
    """Run model forward on the current batch images and return raw head output.

    This helper keeps prediction retrieval explicit in training code and returns the
    un-decoded model output as-is (before NMS/post-processing).
    """
    model_pred = model(batch['img'].to(device))
    return model_pred


def get_model_raw_output_and_features(batch, model, device):
    """Return raw output and the Detect input features for Bridge Future."""
    head = _get_detect_head(model)
    captured = {}

    def _capture_head_input(_module, inputs):
        features = inputs[0]
        if not isinstance(features, list):
            raise TypeError(f"Detect input must be a feature list, got {type(features)}")
        captured["features"] = list(features)

    handle = head.register_forward_pre_hook(_capture_head_input)
    try:
        model_pred = model(batch["img"].to(device))
    finally:
        handle.remove()
    if "features" not in captured:
        raise RuntimeError("Detect input hook did not capture Bridge Future features")
    return model_pred, captured["features"]


def _raw_detect_levels(output):
    """Return the raw Detect prediction dict from train- or eval-mode model output."""
    if isinstance(output, dict):
        return output
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        return output[1]
    raise TypeError(f"Expected Detect prediction dict or (decoded, dict), got {type(output)}")


def _get_detect_head(model):
    """Return the detection head module (Detect or OBB)."""
    model = model.module if hasattr(model, "module") else model  # unwrap DDP before attribute access
    head = model.model[-1]
    if isinstance(head, (Detect, OBB)):
        return head
    raise TypeError(f"Unsupported head type: {type(head)}")


def _apply_train_head_args(model, args):
    """Copy train-time decode settings onto a (possibly frozen) detection model.

    `Detect._end2end` and `agnostic_nms` are Python attributes, not state_dict
    entries. A yaml-rebuilt YOLO26 head defaults to end2end=True, so a teacher
    loaded from an expanded checkpoint would otherwise decode the untrained
    one2one branch while the student trains one2many+NMS — which silently
    destroys pseudo-label quality and anti-forgetting.
    """
    if not hasattr(model, "end2end"):
        return
    if getattr(args, "end2end", None) is not None:
        model.end2end = args.end2end
    head_kwargs = {}
    if getattr(args, "agnostic_nms", None) is not None:
        head_kwargs["agnostic_nms"] = args.agnostic_nms
    if getattr(model, "end2end", False):
        head_kwargs["max_det"] = args.max_det
    if head_kwargs:
        model.set_head_attr(**head_kwargs)


def _check_teacher_decode_alignment(teacher, student, teacher_name):
    """Fail fast if a frozen teacher's decode mode differs from the student's.

    `Detect._end2end` is a Python attribute, not a state_dict entry, so a teacher
    checkpoint can silently disagree with the current training configuration (e.g.
    an expanded head rebuilt from a yaml whose default is end2end=True) and decode
    the untrained one2one branch for pseudo labels instead of one2many+NMS.
    """
    expected = getattr(student, "end2end", None)
    actual = getattr(teacher, "end2end", None)
    if expected is not None and actual is not None and expected != actual:
        raise ValueError(
            f"{teacher_name} decode mismatch: expected end2end={expected} (current training model), "
            f"got end2end={actual}. Pass a matching --end2end or re-generate the checkpoint with "
            f"tools/expand_model_head.py, which copies the source head's decode attributes."
        )


def _get_nsgp_backbone_names(model, module_names):
    """Return PCA module names that belong to the configured YOLO backbone."""
    backbone_config = getattr(model, "yaml", {}).get("backbone")
    if not isinstance(backbone_config, list):
        raise ValueError("Cannot identify YOLO backbone layers for normalized NSGP projection")
    backbone_layer_count = len(backbone_config)
    backbone_names = []
    for name in module_names:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            if int(parts[1]) < backbone_layer_count:
                backbone_names.append(name)
    return backbone_names


def _extract_pca_spectra(pca_cache):
    """Collect per-group PCA components and explained variances from a loaded PCA cache.

    Cache entries are plain dicts serialized by tools/pca.py
    ({"class": ..., "state": ...}), so loading never requires the PCA operator
    classes to be importable in this process (DDP workers run outside tools/).

    Args:
        pca_cache (dict): module name -> list of serialized per-group operator entries.

    Returns:
        tuple[dict, dict, dict]: module name -> (groups, n_components) components
            tensor, module name -> (groups, n_components) explained-variance tensor,
            and module name -> governed parameter name (``weight`` by default).
    """
    components, eigen_values, param_names = {}, {}, {}
    for name, entries in pca_cache.items():
        group_components, group_eigen_values = [], []
        param_names[name] = entries[0].get("param", "weight") if entries else "weight"
        for ig, entry in enumerate(entries):
            if not isinstance(entry, dict) or "state" not in entry:
                raise TypeError(
                    f"PCA cache entry for module '{name}' group {ig} has an unexpected format "
                    f"(expected a dict with a 'state' key, got {type(entry)}). "
                    f"Regenerate the cache with tools/pca.py."
                )
            group_components.append(entry["state"]["components_"])
            group_eigen_values.append(entry["state"]["explained_variance_"])
        components[name] = torch.stack(group_components)
        eigen_values[name] = torch.stack(group_eigen_values)
    return components, eigen_values, param_names


def _head_raw_output_to_list(head, raw_output, img_h, img_w, conf_threshold, agnostic_nms=False):
    """Convert head raw output to list of detections per image for merging.
    Detect: list of (num_boxes, 6) [xywhn, conf, cls].
    OBB: list of (num_boxes, 7) [xywhn, conf, cls, angle] with rotated NMS and angle preserved.
    """
    if isinstance(head, OBB):
        # OBB: pass full (4+nc+1) for rotated NMS; keep angle in output
        decoded_cat = raw_output[0]  # (bs, 4+nc+ne, num_anchors)
        prediction = decoded_cat if decoded_cat.dim() == 3 else decoded_cat.unsqueeze(0)
        pred = non_max_suppression(
            prediction=prediction,
            conf_thres=conf_threshold,
            iou_thres=0.45,
            max_det=head.max_det,
            nc=head.nc,
            agnostic=agnostic_nms,
            rotated=True,
        )
        # pred: list of (num_boxes, 7) [xywh_px, conf, cls, angle_rad]; convert xywh to xywhn
        out = []
        for p in pred:
            if len(p) > 0:
                xywh_px = p[:, :4].clone()
                xywh_px[:, 0] /= img_w
                xywh_px[:, 1] /= img_h
                xywh_px[:, 2] /= img_w
                xywh_px[:, 3] /= img_h
                pred_7 = torch.cat([xywh_px, p[:, 4:]], dim=1)  # [xywhn, conf, cls, angle]
                out.append(pred_7)
            else:
                out.append(torch.empty((0, 7), device=decoded_cat.device, dtype=decoded_cat.dtype))
        return out
    # Detect: (decoded, raw); NMS returns xyxy then we convert to xywhn
    decoded, _ = raw_output
    prediction = decoded if decoded.dim() == 3 else decoded[0]
    pred = non_max_suppression(
        prediction=prediction,
        conf_thres=conf_threshold,
        iou_thres=0.45,
        max_det=head.max_det,
        nc=head.nc,
        agnostic=agnostic_nms,
    )
    base_model_pred_xywh = []
    for p in pred:
        if len(p) > 0:
            boxes_xyxy = p[:, :4]
            boxes_xywhn = xyxy2xywhn(boxes_xyxy, w=img_w, h=img_h)
            pred_xywh = torch.cat([boxes_xywhn, p[:, 4:]], dim=1)
            base_model_pred_xywh.append(pred_xywh)
        else:
            base_model_pred_xywh.append(p)
    return base_model_pred_xywh


def merge_pseudo_labels_with_gt(
    batch, base_model, conf_threshold, filter_iou_threshold, device, base_model_pred=None, agnostic_nms=False
):
    """
    Generate pseudo labels from base model and merge with GT labels, filtering by IoU.
    Supports both Detect (axis-aligned) and OBB (oriented) heads.
    
    Args:
        batch: Dict with keys 'img' (B, C, H, W), 'bboxes' (N, 4), 'cls' (N, 1), 'batch_idx' (N,)
        base_model: Base model for pseudo label generation
        conf_threshold: Minimum teacher confidence retained by NMS
        filter_iou_threshold: IoU threshold for filtering pseudo labels
        device: Device to run inference on
        base_model_pred (Optional): The prediction of base model
        agnostic_nms (bool): Class-agnostic NMS, matching train/eval `agnostic_nms`.
    Returns:
        Modified batch dict with merged labels
    """
    if base_model_pred is None:
        base_model_pred = base_model(batch['img'].to(device))
    head = _get_detect_head(base_model)
    img_h, img_w = batch['img'].shape[-2:]
    base_model_pred_list = _head_raw_output_to_list(
        head, base_model_pred, img_h, img_w, conf_threshold, agnostic_nms=agnostic_nms
    )
    is_obb = isinstance(head, OBB)
    # Detect: list of (num_boxes, 6) [xywhn, conf, cls]. OBB: list of (num_boxes, 7) [xywhn, conf, cls, angle].

    batch_size = batch['img'].shape[0]
    gt_bboxes = batch['bboxes']  # (N, 4) or (N, 5) for OBB xywhr
    gt_cls = batch['cls']  # (N, 1)
    gt_batch_idx = batch['batch_idx']  # (N,)
    n_bbox_cols = gt_bboxes.shape[1]  # 4 or 5
    merged_cols = 4 + 1 + 1 if not is_obb else 5 + 1 + 1  # bbox_cols + conf + cls

    merged_labels = []
    for img_idx in range(batch_size):
        gt_mask = (gt_batch_idx == img_idx)
        if gt_mask.any():
            gt_boxes_img = gt_bboxes[gt_mask]  # (num_gt, 4) or (num_gt, 5)
            gt_cls_img = gt_cls[gt_mask]  # (num_gt, 1)
        else:
            gt_boxes_img = torch.empty((0, n_bbox_cols), device=gt_bboxes.device)
            gt_cls_img = torch.empty((0, 1), device=gt_cls.device, dtype=torch.long)

        pseudo_labels_img = base_model_pred_list[img_idx]  # (num_pseudo, 6) or (num_pseudo, 7)

        if len(pseudo_labels_img) > 0 and len(gt_boxes_img) > 0:
            if is_obb:
                # OBB: use rotated IoU (batch_probiou) with xywhr format
                pseudo_xywhr = pseudo_labels_img[:, [0, 1, 2, 3, 6]]  # (num_pseudo, 5)
                iou_matrix = batch_probiou(gt_boxes_img, pseudo_xywhr)  # (num_gt, num_pseudo)
                max_iou_per_pseudo = iou_matrix.max(dim=0)[0]  # (num_pseudo,)
            else:
                pseudo_boxes_xywh = pseudo_labels_img[:, :4]
                pseudo_boxes_xyxy = xywh2xyxy(pseudo_boxes_xywh)
                gt_boxes_xyxy = xywh2xyxy(gt_boxes_img)
                iou_matrix = bbox_iou(pseudo_boxes_xyxy.unsqueeze(1), gt_boxes_xyxy.unsqueeze(0), xywh=False).squeeze(-1)
                max_iou_per_pseudo = iou_matrix.max(dim=1)[0]
            keep_mask = max_iou_per_pseudo < filter_iou_threshold
            filtered_pseudo = pseudo_labels_img[keep_mask]
        else:
            filtered_pseudo = pseudo_labels_img

        if len(gt_boxes_img) > 0:
            gt_labels = torch.cat([
                gt_boxes_img,
                torch.ones((len(gt_boxes_img), 1), device=gt_boxes_img.device),
                gt_cls_img
            ], dim=1)  # (num_gt, 6) or (num_gt, 7)
        else:
            gt_labels = torch.empty((0, merged_cols), device=filtered_pseudo.device if len(filtered_pseudo) > 0 else batch['img'].device)

        if len(filtered_pseudo) > 0:
            merged = torch.cat([gt_labels, filtered_pseudo], dim=0) if len(gt_labels) > 0 else filtered_pseudo
        else:
            merged = gt_labels
        merged_labels.append(merged)

    all_bboxes = []
    all_cls = []
    all_batch_idx = []
    for img_idx, merged in enumerate(merged_labels):
        if len(merged) > 0:
            all_bboxes.append(merged[:, :n_bbox_cols])
            all_cls.append(merged[:, n_bbox_cols + 1].long().unsqueeze(-1))  # cls column
            all_batch_idx.append(torch.full((len(merged),), img_idx, device=merged.device))

    if len(all_bboxes) > 0:
        batch['bboxes'] = torch.cat(all_bboxes, dim=0)
        batch['cls'] = torch.cat(all_cls, dim=0)
        batch['batch_idx'] = torch.cat(all_batch_idx, dim=0)
    else:
        batch['bboxes'] = torch.empty((0, n_bbox_cols), device=batch['img'].device)
        batch['cls'] = torch.empty((0, 1), device=batch['img'].device, dtype=torch.long)
        batch['batch_idx'] = torch.empty((0), device=batch['img'].device, dtype=torch.long)
    
    return batch


def _decode_detect_output(head, raw_output):
    """Decode a Detect head output to the (B, 4 + nc, num_anchors) prediction layout.

    Accepts train-mode prediction dicts and eval-mode (decoded, preds) tuples, with or
    without an end2end {"one2many", "one2one"} split. End2end outputs are decoded from
    the one2many branch; the one2one branch is detached from the backbone during training.

    Args:
        head: The Detect head that produced the output (its own anchors/strides are used).
        raw_output: Train-mode dict, end2end train-mode dict, or eval-mode (decoded, preds) tuple.

    Returns:
        Decoded predictions, (B, 4 + nc, num_anchors), with sigmoid-activated cls channels.
    """
    if isinstance(raw_output, tuple):  # eval mode: (decoded, preds)
        decoded, preds = raw_output
        if not (isinstance(preds, dict) and "one2many" in preds):
            return decoded  # already (B, 4 + nc, A)
        raw_output = preds["one2many"]
    elif isinstance(raw_output, dict) and "one2many" in raw_output:  # end2end train mode
        raw_output = raw_output["one2many"]
    return head._inference(raw_output)


def get_dist_loss(model_pred, base_model_pred, model, base_model, dist_topk=1):
    """KL distillation from teacher to student on the teacher's top-k class channels.

    For each anchor, take the k teacher class channels (historical classes only) with the
    highest confidence, compute a binary KL(teacher || student) on each selected channel
    independently, weight each channel's loss by its teacher confidence, and normalize by
    the sum of all selected teacher confidences over all anchors and channels.

    Args:
        model_pred: Student raw output (pre-NMS), train-mode dict or end2end train-mode dict.
        base_model_pred: Teacher raw output, eval-mode (decoded, preds) tuple.
        model: Student detection model (used to locate the student Detect head).
        base_model: Teacher detection model (used to locate the teacher Detect head).
        dist_topk (int): Number of teacher channels to distill per anchor; -1 means all
            historical class channels. Must be -1 or >= 1.

    Returns:
        Weighted KL distillation loss (scalar tensor).
    """
    eps = 1e-6
    student_head = _get_detect_head(model)
    teacher_head = _get_detect_head(base_model)
    nc = student_head.nc
    model_pred = _decode_detect_output(student_head, model_pred)  # (B, 4 + nc, A)
    base_model_pred = _decode_detect_output(teacher_head, base_model_pred)  # (B, 4 + nc_teacher, A)
    cls_start, cls_end = 4, 4 + nc
    student_cls = model_pred[:, cls_start:cls_end, :]  # (B, C, A)
    teacher_cls = base_model_pred[:, cls_start:cls_end, :]  # (B, C_t, A), historical classes only

    n_teacher_ch = teacher_cls.shape[1]
    k = n_teacher_ch if dist_topk == -1 else min(dist_topk, n_teacher_ch)
    # Per-anchor top-k teacher channels by confidence.
    teacher_topk, topk_idx = teacher_cls.topk(k, dim=1)  # (B, K, A), (B, K, A)
    student_topk = torch.gather(student_cls, 1, topk_idx)  # (B, K, A)

    # Binary KL(teacher || student) per selected channel:
    # p,q in (0,1) from sigmoid cls; complementary mass is 1-p / 1-q.
    dtype = model_pred.dtype
    p = teacher_topk.to(torch.float32).clamp(eps, 1.0 - eps)
    q = student_topk.to(torch.float32).clamp(eps, 1.0 - eps)
    teacher_bin = torch.stack((p, 1 - p), dim=2)  # (B, K, 2, A)
    student_bin = torch.stack((q, 1 - q), dim=2)  # (B, K, 2, A)
    kl_per_channel = F.kl_div(student_bin.log(), teacher_bin, reduction="none").sum(dim=2).to(dtype)  # (B, K, A)

    # Weighted sum over anchors and channels, normalized by total top-k confidence.
    return (kl_per_channel * teacher_topk).sum() / teacher_topk.sum().clamp_min(eps)


def _cycle_batches(loader):
    """Yield batches from a dataloader indefinitely, re-entering it after each full pass.

    YOLO loaders are InfiniteDataLoader whose ``__iter__`` yields exactly ``len(loader)``
    batches per pass over a persistent worker pool, so re-entering is cheap.
    """
    while True:
        yield from loader


def _mean_lwf_terms(first, second):
    """Average two ``(classification, regression)`` LwF term pairs.

    ``None`` marks a term the configured LwF mode does not produce (the plain LwF
    baseline has no classification term), and averaging it in keeps the mean over
    the current-task and replay images identical to a single concatenated batch.
    """
    if first is None:
        return second
    if second is None:
        return first
    return tuple(None if a is None else 0.5 * (a + b) for a, b in zip(first, second))


class AntiForgetTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the AntiForgetTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)

    def _anti_forget_loss_names(self):
        """Return training loss names extended with the enabled anti-forgetting loss terms.

        The order matches the order in which the extra loss items are appended in `_do_train`.
        The third criterion term is dfl_loss when reg_max > 1, l1_loss otherwise (e.g. yolo26).
        """
        model_to_use = self.model.module if hasattr(self.model, "module") else self.model
        use_dfl = _get_detect_head(model_to_use).reg_max > 1
        loss_names = ["box_loss", "cls_loss", "dfl_loss" if use_dfl else "l1_loss"]
        if self.args.distillation:
            loss_names.append("dist_loss")
        if self.args.espreg:
            loss_names.append("espreg_loss")
        if self.args.ewc:
            loss_names.append("ewc_loss")
        if self.args.l2:
            loss_names.append("l2_loss")
        if self.args.repre:
            loss_names.append("repre_loss")
        if self.args.replay and self.args.replay_mode in {"supervised", "both"}:
            loss_names.append("replay_loss")
        if self.args.lwf:
            if self.args.lwf_mode == "lwf":
                loss_names.append("lwf_loss")
            elif self.args.lwf_mode == "prad":
                loss_names.extend(("prad_cls_loss", "prad_bg_loss", "prad_reg_loss"))
            else:
                loss_names.extend(("lwf_cls_loss", "lwf_reg_loss"))
        if self.args.prd:
            loss_names.extend(("prd_cls_loss", "prd_reg_loss"))
        return tuple(loss_names)

    def _lwf_teacher_output(self, batch):
        """Run the frozen LwF teacher on a batch and return its output plus head input features.

        The head input features are captured only for the plain LwF baseline, which
        distills features instead of head outputs.
        """
        with torch.no_grad():
            if self.args.lwf_mode == "lwf":
                return get_model_raw_output_and_features(batch, self.lwf_teacher, self.device)
            return get_model_raw_output(batch, self.lwf_teacher, self.device), None

    def _lwf_terms(self, student_output, student_feats, teacher_output, teacher_feats):
        """Return the (classification, regression) LwF terms of one student/teacher output pair."""
        if self.args.lwf_mode == "lwf":
            return None, feature_l2_loss(student_feats, teacher_feats, self.lwf_feature_channels)
        return self.lwf_loss(
            detect_branch_output(student_output),
            detect_branch_output(teacher_output),
            _get_detect_head(self.model).stride,
        )

    def _setup_patch_prototypes(self, student_head, teacher_head, prad_enabled):
        """Load the patch prototype artifact and build the prad / prd losses.

        Args:
            student_head (Detect): Student detection head.
            teacher_head (Detect): Frozen previous-task detection head.
            prad_enabled (bool): Whether the prototype-reliability distillation is active.
        """
        if not self.args.patch_prototypes:
            raise ValueError("patch_prototypes is required when prad or prd is enabled")
        if student_head.reg_max != 1 or teacher_head.reg_max != 1:
            raise ValueError(
                f"Patch prototype distillation targets direct-regression heads (reg_max=1), got student "
                f"{student_head.reg_max} and teacher {teacher_head.reg_max}"
            )
        if prad_enabled:
            if getattr(unwrap_model(self.model), "end2end", False):
                raise ValueError("prad requires end2end training to be disabled (dense one2many branch)")
            if self.args.replay_mode in {"distill", "both"}:
                raise ValueError(
                    "prad is incompatible with replay_mode 'distill'/'both': it distills the current batch only. "
                    "Use prd for prototype replay, or replay_mode 'supervised'."
                )
        subspace, bank, meta = load_patch_artifact(self.args.patch_prototypes)
        radius = head_patch_radius(student_head)
        if int(meta["patch_radius"]) != radius:
            raise ValueError(
                f"Patch artifact '{self.args.patch_prototypes}' stores radius {meta['patch_radius']} but the "
                f"current detection head needs radius {radius}"
            )
        channels = head_input_channels(student_head)
        if list(subspace.channels) != channels:
            raise ValueError(
                f"Patch artifact head channels {list(subspace.channels)} differ from the model {channels}"
            )
        if int(meta["history_nc"]) != teacher_head.nc:
            raise ValueError(
                f"Patch artifact was built for {meta['history_nc']} old classes but the teacher exposes "
                f"{teacher_head.nc}; regenerate the artifact for this stage"
            )
        bank.bind_nc(student_head.nc)
        subspace.to(self.device)
        bank.to(self.device)
        self.patch_subspace, self.patch_bank = subspace, bank
        LOGGER.info(
            f"Patch prototypes '{self.args.patch_prototypes}': {bank.num_components} components, "
            f"dims {subspace.dims}, radius {radius}"
        )
        if prad_enabled:
            self.prad_loss = PrototypeReliabilityLwFLoss(
                subspace=subspace,
                bank=bank,
                history_nc=teacher_head.nc,
                fg_threshold=self.args.prad_fg_threshold,
                bg_threshold=self.args.prad_bg_threshold,
                conf_power=self.args.prad_conf_power,
                proto_threshold=self.args.prad_proto_threshold,
                proto_temperature=self.args.prad_proto_temperature,
                bg_weight=self.args.prad_bg_weight,
                split_normalize=self.args.prad_split_normalize,
                max_fg=self.args.prad_max_fg,
            )
            criterion = RecordingDetectionLoss(unwrap_model(self.model))
            unwrap_model(self.model).criterion = criterion
        if self.args.prd:
            self.prd_loss = PatchReplayDistillation(
                student_head=student_head,
                teacher_head=teacher_head,
                subspace=subspace,
                bank=bank,
                samples=self.args.prd_samples,
                std_scale=self.args.prd_std_scale,
                smooth_l1_beta=1.0,
                seed=RANK if RANK > -1 else 0,
            )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ============================== MODIFIED: set up base model ============================================
        if self.args.reference_model:
            self.base_model, _ = load_checkpoint(self.args.reference_model, device=self.device)
            self.base_model = self.base_model.eval()
        else:
            self.base_model = deepcopy(self.model).eval()
        self.base_model.requires_grad_(False)
        _apply_train_head_args(self.base_model, self.args)
        _check_teacher_decode_alignment(self.base_model, self.model, "reference_model")
        # ============================== END: set up base model =================================================

        # ============================== MODIFIED: validate distillation settings ==============================
        if self.args.distillation and not (self.args.dist_topk == -1 or self.args.dist_topk >= 1):
            raise ValueError(f"dist_topk must be -1 (all historical channels) or >= 1, got {self.args.dist_topk}")
        # ============================== END: validate distillation settings ====================================

        # ============================== MODIFIED: set up BPF ==================================================
        if self.args.bpf:
            if not isinstance(_get_detect_head(self.model), Detect):
                raise TypeError("BPF currently supports axis-aligned Detect models only")
            if self.args.bpf_past or self.args.bpf_dwf:
                if not self.args.bpf_source_model:
                    raise ValueError("bpf_source_model is required when bpf_past or bpf_dwf is enabled")
                self.bpf_source_model, _ = load_checkpoint(self.args.bpf_source_model, device=self.device)
                self.bpf_source_model = self.bpf_source_model.eval()
                self.bpf_source_model.requires_grad_(False)
                _apply_train_head_args(self.bpf_source_model, self.args)
                _check_teacher_decode_alignment(self.bpf_source_model, self.model, "bpf_source_model")
                if not isinstance(_get_detect_head(self.bpf_source_model), Detect):
                    raise TypeError("BPF source model must use a Detect head")
            if self.args.bpf_dwf:
                if not self.args.bpf_interim_model:
                    raise ValueError("bpf_interim_model is required when bpf_dwf is enabled")
                self.bpf_interim_model, _ = load_checkpoint(self.args.bpf_interim_model, device=self.device)
                self.bpf_interim_model = self.bpf_interim_model.eval()
                self.bpf_interim_model.requires_grad_(False)
                _apply_train_head_args(self.bpf_interim_model, self.args)
                _check_teacher_decode_alignment(self.bpf_interim_model, self.model, "bpf_interim_model")
                if not isinstance(_get_detect_head(self.bpf_interim_model), Detect):
                    raise TypeError("BPF interim model must use a Detect head")
            if self.args.bpf_dwf and self.args.distillation:
                raise ValueError("BPF DwF and the KL distillation loss are mutually exclusive")
        # ============================== END: set up BPF ========================================================

        # ============================== MODIFIED: set up L2 regularization loss ==============================
        if self.args.l2:
            self.l2_loss_weight = self.args.l2_loss_weight
            self.l2_loss = L2Loss(model=self.model, ref_model=self.base_model)
        # ============================== END: set up L2 regularization loss ====================================

        # ============================== MODIFIED: set up ESPReg loss ============================================
        if self.args.espreg:
            self.espreg_loss_weight = self.args.espreg_loss_weight
            self.pca_cache = joblib.load(self.args.pca_cache_path)
            components, eigen_values, param_names = _extract_pca_spectra(self.pca_cache)
            espreg_modules = tuple(self.pca_cache)
            if self.args.espreg_scope == "backbone":
                espreg_modules = tuple(_get_nsgp_backbone_names(self.model, espreg_modules))
            elif self.args.espreg_scope != "all":
                raise ValueError(f"espreg_scope must be 'all' or 'backbone', got {self.args.espreg_scope!r}")
            self.espreg_loss = ESPRegLoss(
                self.model,
                self.base_model,
                module_names=espreg_modules,
                components={name: components[name] for name in espreg_modules},
                eigen_values={name: eigen_values[name] for name in espreg_modules},
                param_names={name: param_names[name] for name in espreg_modules},
                weight_mode=self.args.espreg_weight_mode,
                form=self.args.espreg_form,
            )
            self.governed_modules = tuple(espreg_modules)
            LOGGER.info(
                f"ESPReg over {len(espreg_modules)} modules ({self.args.espreg_scope}), "
                f"weight_mode={self.args.espreg_weight_mode}, form={self.args.espreg_form}"
            )
        # ============================== END: set up ESPReg loss =================================================

        # ============================== MODIFIED: set up NSGP ============================================
        if self.args.nsgp:
            if not hasattr(self, "pca_cache"):
                self.pca_cache = joblib.load(self.args.pca_cache_path)
            components, eigen_values, param_names = _extract_pca_spectra(self.pca_cache)
            self.nsgp_flexibility = getattr(self.args, 'nsgp_flexibility', 1.0)
            module_names = tuple(self.pca_cache)
            governed = set(getattr(self, "governed_modules", ()))
            governed.update(module_names)
            self.governed_modules = tuple(sorted(governed))
            self.nsgp_operator = NSGP(
                module_names=module_names,
                components=components,
                eigen_values=eigen_values,
                param_names=param_names,
                normalized_module_names=_get_nsgp_backbone_names(self.model, module_names),
            )
        # ============================== END: set up NSGP =================================================

        # ============================== MODIFIED: set up EWC loss ============================================
        if self.args.ewc:
            self.ewc_loss_weight = self.args.ewc_loss_weight
            exclude_prefixes = tuple(f"{name}." for name in getattr(self, "governed_modules", ()))
            ewc_state = load_ewc_state(self.args.importance_path, map_location=self.device)
            self.ewc_loss = EWCLoss(model=self.model, state=ewc_state, exclude_prefixes=exclude_prefixes)
            LOGGER.info(
                f"EWC over {len(self.ewc_loss.importance)} parameters, "
                f"excluding {len(exclude_prefixes)} NSGP/ESPReg modules"
            )
        # ============================== END: set up EWC loss =================================================

        # ============================== MODIFIED: set up RePRE ===========================================
        if self.args.repre:
            prototype_data = torch.load(self.args.repre_prototypes, map_location="cpu")
            detect_head = _get_detect_head(self.model)
            self.repre_loss_weight = self.args.repre_loss_weight
            self.repre = RegionalPrototypeReplay(
                detect_head=detect_head,
                prototype_data=prototype_data,
                device=self.device,
            )
        # ============================== END: set up RePRE ===============================================

        # ============================== MODIFIED: validate replay mode =================================
        if self.args.replay_mode not in {"supervised", "distill", "both"}:
            raise ValueError(
                f"replay_mode must be 'supervised', 'distill' or 'both', got {self.args.replay_mode!r}"
            )
        if self.args.replay_mode in {"distill", "both"} and not self.args.lwf:
            raise ValueError(f"replay_mode '{self.args.replay_mode}' needs lwf to distill the replay samples")
        # ============================== END: validate replay mode =======================================

        # ============================== MODIFIED: set up distillation teacher =============================
        self.lwf_keep_classes = None
        self.patch_subspace = None
        self.prad_loss = None
        self.prd_loss = None
        self.prad_teacher_feats = False
        prad_enabled = self.args.lwf and self.args.lwf_mode == "prad"
        if self.args.lwf or self.args.prd:
            if not self.args.lwf_teacher_model:
                raise ValueError(
                    "lwf_teacher_model (the previous-task checkpoint) is required when lwf or prd is enabled"
                )
            if self.args.lwf and self.args.lwf_mode not in {"yolo_lwf", "lwf", "prad"}:
                raise ValueError(f"lwf_mode must be 'yolo_lwf', 'lwf' or 'prad', got {self.args.lwf_mode!r}")
            self.lwf_teacher, _ = load_checkpoint(self.args.lwf_teacher_model, device=self.device)
            self.lwf_teacher = self.lwf_teacher.eval()
            self.lwf_teacher.requires_grad_(False)
            _apply_train_head_args(self.lwf_teacher, self.args)
            _check_teacher_decode_alignment(self.lwf_teacher, self.model, "lwf_teacher_model")
            student_head = _get_detect_head(self.model)
            teacher_head = _get_detect_head(self.lwf_teacher)
            self.lwf_history_nc = teacher_head.nc
            if teacher_head.reg_max != student_head.reg_max:
                raise ValueError(
                    f"LwF teacher reg_max {teacher_head.reg_max} differs from the student reg_max "
                    f"{student_head.reg_max}; the box head layout must match"
                )
            if not 0 < self.lwf_history_nc < student_head.nc:
                raise ValueError(
                    f"LwF needs history class channels below the student class count, got teacher nc "
                    f"{self.lwf_history_nc} and student nc {student_head.nc}"
                )
            if self.args.lwf and self.args.lwf_mode in {"yolo_lwf", "lwf"}:
                self.lwf_loss = YOLOLwFLoss(
                    reg_max=student_head.reg_max,
                    history_nc=self.lwf_history_nc,
                    temperature=self.args.lwf_temperature,
                    wce=self.args.lwf_wce,
                    cls_iou=self.args.lwf_cls_iou,
                )
                self.lwf_feature_channels = 4 * student_head.reg_max + self.lwf_history_nc
            if self.args.lwf and self.args.lwf_mask_old:
                # Current-task labels do not annotate the history classes; masking keeps the
                # detection loss on the new-class channels for every lwf-based mode.
                self.lwf_keep_classes = torch.arange(self.lwf_history_nc, student_head.nc, device=self.device)
            if prad_enabled or self.args.prd:
                self._setup_patch_prototypes(student_head, teacher_head, prad_enabled)
            LOGGER.info(
                f"Distillation ({self.args.lwf_mode if self.args.lwf else 'prd'}) teacher "
                f"'{self.args.lwf_teacher_model}' with {self.lwf_history_nc} history classes, "
                f"student with {student_head.nc} class channels"
            )
        # ============================== END: set up distillation teacher ==================================

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device.index], broadcast_buffers=False, find_unused_parameters=True
            )

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # ============================== MODIFIED: set up replay dataloader =====================================
        if self.args.replay:
            if not isinstance(_get_detect_head(self.model), Detect):
                # OBB loss skips the box head on zero-FG batches, which breaks the multi-forward
                # DDP path (see v8DetectionLoss); only the Detect criterion keeps the graph intact.
                raise TypeError("Replay currently supports axis-aligned Detect models only")
            if not self.args.replay_data:
                raise ValueError("replay_data is required when replay is enabled")
            replay_config = YAML.load(self.args.replay_data)
            replay_train = replay_config.get("train")
            if not isinstance(replay_train, str):
                raise KeyError(f"Replay dataset config has no 'train' split: {self.args.replay_data}")
            replay_train_path = Path(replay_train)
            if not replay_train_path.is_absolute():
                replay_train_path = Path(self.args.replay_data).parent / replay_train_path
            if not replay_train_path.is_dir():
                raise FileNotFoundError(f"Replay image directory does not exist: {replay_train_path}")
            self.replay_loader = self.get_dataloader(
                str(replay_train_path), batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
        # ============================== END: set up replay dataloader ==========================================
        # ============================== MODIFIED: validate LwF label coverage ============================
        if self.lwf_keep_classes is not None:
            annotated = sorted(
                {int(cls) for label in self.train_loader.dataset.labels for cls in np.asarray(label["cls"]).reshape(-1)}
            )
            unsupervised = sorted(set(annotated) - set(self.lwf_keep_classes.tolist()))
            if unsupervised:
                raise ValueError(
                    f"The current-task labels annotate history class ids {unsupervised}, which lwf_mask_old "
                    f"would drop from supervision; disable lwf_mask_old or fix the task split"
                )
        # ============================== END: validate LwF label coverage =================================
        # ============================== MODIFIED: DDP-safe validation setup (all ranks) ==================
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        # validate() broadcasts EMA buffers and the validator gathers stats across ranks,
        # so every rank needs its own validator, val shard, and EMA copy (mirrors BaseTrainer).
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()
        # ============================== END: DDP-safe validation setup ===================================

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        world_size = getattr(self, "world_size", None) or world_size
        if world_size > 1:
            self._setup_ddp()
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                # ============================== MODIFIED: shuffle replay sampler per epoch ====================
                if self.args.replay:
                    self.replay_loader.sampler.set_epoch(epoch)
                # ============================== END: shuffle replay sampler per epoch =========================
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            # ============================== MODIFIED: replay iterator ==========================================
            # Replay set is usually smaller than the train set, so it is cycled indefinitely.
            replay_iter = _cycle_batches(self.replay_loader) if self.args.replay else None
            # ============================== END: replay iterator ===============================================

            # ============================== MODIFIED: register hook ===========================================
            if self.args.espreg:
                self.espreg_loss.register_hook()
            
            if self.args.espreg:
                # Perform a forward in base model to hook out the base weights
                with torch.no_grad():
                    _ = self.base_model(
                        torch.randn(1, 3, self.args.imgsz, self.args.imgsz, device=self.device)
                    )
            # ============================== END: register hook ================================================
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    # ============================== MODIFIED: get raw preds ==================================
                    if self.args.pseudo_label or self.args.distillation:
                        with torch.no_grad():
                            base_model_pred = get_model_raw_output(batch, self.base_model, self.device)
                    if self.args.bpf and (self.args.bpf_past or self.args.bpf_dwf):
                        with torch.no_grad():
                            bpf_source_pred = get_model_raw_output(batch, self.bpf_source_model, self.device)
                    if self.args.bpf and self.args.bpf_dwf:
                        with torch.no_grad():
                            bpf_interim_pred = get_model_raw_output(batch, self.bpf_interim_model, self.device)
                    if self.args.lwf or self.args.prd:
                        lwf_teacher_pred, lwf_teacher_feats = self._lwf_teacher_output(batch)
                    if (self.args.bpf and self.args.bpf_future) or (self.args.lwf and self.args.lwf_mode == "lwf"):
                        model_pred, head_features = get_model_raw_output_and_features(
                            batch, self.model, self.device
                        )
                    else:
                        model_pred = get_model_raw_output(batch, self.model, self.device)
                        head_features = None
                    # ============================== END: get raw preds =======================================

                    # ============================== MODIFIED: generate pseudo labels ===============================
                    if self.args.pseudo_label:
                        batch = merge_pseudo_labels_with_gt(
                            batch,
                            self.base_model,
                            self.args.conf_threshold,
                            self.args.filter_iou_threshold,
                            self.device,
                            base_model_pred,
                            agnostic_nms=self.args.agnostic_nms,
                        )
                    # ============================== END: generate pseudo labels ====================================

                    # ============================== MODIFIED: BPF labels and future mask ===========================
                    if self.args.bpf and self.args.bpf_past:
                        source_head = _get_detect_head(self.bpf_source_model)
                        pseudo_detections = bpf_pseudo_detections(
                            head=source_head,
                            raw_output=_raw_detect_levels(bpf_source_pred),
                            image_size=batch["img"].shape[-2:],
                            score_threshold=self.args.bpf_score_threshold,
                            nms_threshold=self.args.bpf_nms_threshold,
                        )
                        batch = merge_bpf_pseudo_labels(
                            batch=batch,
                            detections=pseudo_detections,
                            iou_low=self.args.bpf_iou_low,
                            iou_high=self.args.bpf_iou_high,
                            low_weight=self.args.bpf_low_weight,
                            high_weight=self.args.bpf_high_weight,
                        )
                    if self.args.bpf and self.args.bpf_future:
                        batch["bpf_ignore_mask"] = select_future_ignore_mask(
                            head=_get_detect_head(self.model),
                            raw_output=_raw_detect_levels(model_pred),
                            head_features=head_features,
                            batch=batch,
                            object_topk=self.args.bpf_object_topk,
                            attention_topk=self.args.bpf_attention_topk,
                            iou_threshold=self.args.bpf_future_iou,
                            attention_power=self.args.bpf_attention_power,
                        )
                    # ============================== END: BPF labels and future mask ==============================
                    
                    # ============================== MODIFIED: get det loss ===============================
                    if self.lwf_keep_classes is not None:
                        batch["cls_keep_classes"] = self.lwf_keep_classes
                    loss, self.loss_items = self.model(batch, preds=model_pred)
                    # ============================== END: get det loss ====================================

                    # ============================== MODIFIED: make a copy of loss items ===============================
                    loss_items = dict(self.loss_items)  # criterion returns a dict of detached per-term scalars
                    # ============================== END: make a copy of loss items ====================================

                    # ============================== MODIFIED: snapshot the current-batch assignment ==================
                    # Later forwards (replay) overwrite the recorded assignment, so read it here.
                    prad_assignment = None
                    if self.prad_loss is not None:
                        prad_assignment = get_assignment(unwrap_model(self.model).criterion)
                    # ============================== END: snapshot the current-batch assignment =======================
                    
                    self.loss = loss.sum()

                    # ============================== MODIFIED: YOLO LwF terms on the current batch ================
                    lwf_current = None
                    if self.args.lwf and self.args.lwf_mode != "prad":
                        lwf_current = self._lwf_terms(model_pred, head_features, lwf_teacher_pred, lwf_teacher_feats)
                    # ============================== END: YOLO LwF terms on the current batch =====================

                    # ============================== MODIFIED: calculate KLD distillation loss ==========================
                    if self.args.distillation:
                        _dist_loss = get_dist_loss(model_pred, base_model_pred, self.model, self.base_model, dist_topk=self.args.dist_topk)
                        self.loss += (_dist_loss * self.args.dist_loss_weight)
                        loss_items["dist_loss"] = _dist_loss.detach()
                    # ============================== END: calculate KL distillation loss ===============================

                    # ============================== MODIFIED: BPF DwF ============================================
                    if self.args.bpf and self.args.bpf_dwf:
                        dwf_loss = compute_dwf_loss(
                            student_head=_get_detect_head(self.model),
                            student_output=_raw_detect_levels(model_pred),
                            source_head=_get_detect_head(self.bpf_source_model),
                            source_output=_raw_detect_levels(bpf_source_pred),
                            interim_head=_get_detect_head(self.bpf_interim_model),
                            interim_output=_raw_detect_levels(bpf_interim_pred),
                            batch=batch,
                            proposal_topk=self.args.bpf_proposal_topk,
                            proposal_samples=self.args.bpf_proposal_samples,
                            split_iou=self.args.bpf_split_iou,
                        )
                        self.loss += (dwf_loss.cls + dwf_loss.box) * self.args.bpf_dwf_weight
                        loss_items["bpf_dwf_cls_loss"] = dwf_loss.cls.detach()
                        loss_items["bpf_dwf_box_loss"] = dwf_loss.box.detach()
                    # ============================== END: BPF DwF ================================================

                    # ============================== MODIFIED: calculate ESPReg loss ===================================
                    if self.args.espreg:
                        _espreg_loss = self.espreg_loss.get_loss()
                        self.loss += (_espreg_loss * self.espreg_loss_weight)
                        loss_items["espreg_loss"] = _espreg_loss.detach()
                    # ============================== END: calculate ESPReg loss ========================================

                    # ============================== MODIFIED: calculate EWC loss ===================================
                    if self.args.ewc:
                        _ewc_loss = self.ewc_loss.get_loss()
                        self.loss += _ewc_loss * self.ewc_loss_weight
                        loss_items["ewc_loss"] = _ewc_loss.detach()
                    # ============================== END: calculate EWC loss ========================================

                    # ============================== MODIFIED: calculate L2 regularization loss ======================
                    if self.args.l2:
                        _l2_loss = self.l2_loss.get_loss()
                        self.loss += _l2_loss * self.l2_loss_weight
                        loss_items["l2_loss"] = _l2_loss.detach()
                    # ============================== END: calculate L2 regularization loss ===========================

                    # ============================== MODIFIED: replay regional prototypes ================================
                    if self.args.repre:
                        repre_loss = self.repre.compute_loss()
                        self.loss += repre_loss * self.repre_loss_weight
                        loss_items["repre_loss"] = repre_loss.detach()
                    # ============================== END: replay regional prototypes ====================================

                    # ============================== MODIFIED: calculate replay loss ====================================
                    lwf_replay = None
                    if self.args.replay:
                        replay_batch = self.preprocess_batch(next(replay_iter))
                        replay_teacher_pred = replay_teacher_feats = None
                        replay_pred = replay_feats = None
                        if self.args.lwf and self.args.replay_mode in {"distill", "both"}:
                            # Replay images are distilled only; no replay label takes part in it.
                            replay_teacher_pred, replay_teacher_feats = self._lwf_teacher_output(replay_batch)
                            if self.args.lwf_mode == "lwf":
                                replay_pred, replay_feats = get_model_raw_output_and_features(
                                    replay_batch, self.model, self.device
                                )
                            else:
                                replay_pred = get_model_raw_output(replay_batch, self.model, self.device)
                        if self.args.replay_mode in {"supervised", "both"}:
                            # Replay samples carry old-class GT only, so the plain detection loss on a
                            # replay batch supervises exactly the historical classes.
                            replay_loss_vec, _ = self.model(replay_batch, preds=replay_pred)
                            _replay_loss = replay_loss_vec.sum()
                            self.loss += _replay_loss * self.args.replay_loss_weight
                            loss_items["replay_loss"] = _replay_loss.detach()
                        if replay_pred is not None:
                            lwf_replay = self._lwf_terms(
                                replay_pred, replay_feats, replay_teacher_pred, replay_teacher_feats
                            )
                    # ============================== END: calculate replay loss =========================================

                    # ============================== MODIFIED: compose YOLO LwF loss ===================================
                    if lwf_current is not None:
                        lwf_cls, lwf_reg = _mean_lwf_terms(lwf_current, lwf_replay)
                        if lwf_cls is None:
                            lwf_total = lwf_reg * self.args.lwf_feature_loss_weight
                            loss_items["lwf_loss"] = lwf_reg.detach()
                        else:
                            lwf_total = (
                                lwf_cls * self.args.lwf_cls_loss_weight + lwf_reg * self.args.lwf_reg_loss_weight
                            )
                            loss_items["lwf_cls_loss"] = lwf_cls.detach()
                            loss_items["lwf_reg_loss"] = lwf_reg.detach()
                        self.loss += lwf_total * batch["img"].shape[0]
                    # ============================== END: compose YOLO LwF loss ========================================

                    # ============================== MODIFIED: prototype reliability distillation ====================
                    if self.prad_loss is not None:
                        prad_terms = self.prad_loss(
                            detect_branch_output(model_pred),
                            detect_branch_output(lwf_teacher_pred),
                            new_class_mask(prad_assignment, self.lwf_history_nc),
                            _get_detect_head(self.model).stride,
                        )
                        self.loss += (
                            prad_terms.cls * self.args.prad_cls_loss_weight
                            + prad_terms.reg * self.args.prad_reg_loss_weight
                        )
                        loss_items["prad_cls_loss"] = prad_terms.cls_fg.detach()
                        loss_items["prad_bg_loss"] = prad_terms.cls_bg.detach()
                        loss_items["prad_reg_loss"] = prad_terms.reg.detach()
                    # ============================== END: prototype reliability distillation =======================

                    # ============================== MODIFIED: patch replay distillation =============================
                    if self.prd_loss is not None:
                        prd_terms = self.prd_loss.compute_loss()
                        self.loss += (
                            prd_terms.cls * self.args.prd_loss_weight
                            + prd_terms.reg * self.args.prd_reg_loss_weight
                        )
                        loss_items["prd_cls_loss"] = prd_terms.cls.detach()
                        loss_items["prd_reg_loss"] = prd_terms.reg.detach()
                    # ============================== END: patch replay distillation =================================

                    if RANK != -1:
                        self.loss *= world_size                   
                    self.tloss = (
                        loss_items
                        if self.tloss is None
                        else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in loss_items.items()}
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = len(self.tloss)
                    pbar.set_description(
                        ("%13s" * 2 + "%13.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *self.tloss.values(),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # ============================== MODIFIED: remove hook ===========================================
            if self.args.espreg:
                self.espreg_loss.remove_handle_()  # Remove hook for ESPReg
            # ============================== END: remove hook ================================================

            # ============================== MODIFIED: update criterion (e.g. E2ELoss o2m gain decay) ==========
            model_to_use = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model_to_use.criterion, "update"):
                model_to_use.criterion.update()
            # ============================== END: update criterion ===========================================

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # ============================== MODIFIED: validate on all ranks ==============================
            # validate() and the validator run cross-rank collectives (EMA buffer broadcast,
            # stats gather, loss reduce), so every DDP rank must enter (mirrors BaseTrainer).
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()
            # ============================== END: validate on all ranks ==================================

            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        # ============================== MODIFIED: final eval on all ranks ================================
        # final_eval() runs the validator, which uses cross-rank collectives under DDP
        # (mirrors BaseTrainer: all ranks enter, only rank 0 logs/plots).
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        # ============================== END: final eval on all ranks =====================================
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")


    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update.
        NSGP projects the completed optimizer update, matching the reference optimizer.
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        # ============================== MODIFIED: capture NSGP parameters ============================
        if self.args.nsgp:
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            self.nsgp_operator.capture_parameters(dict(model_to_use.named_parameters()))
        # ============================== END: capture NSGP parameters ================================
        self.scaler.step(self.optimizer)
        # ============================== MODIFIED: apply NSGP update projection ======================
        if self.args.nsgp:
            self.nsgp_operator.apply_parameter_projection(flexibility=self.nsgp_flexibility)
        # ============================== END: apply NSGP update projection ===========================
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)