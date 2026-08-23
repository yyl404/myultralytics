import numpy as np
import warnings
from copy import deepcopy
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
    callbacks,
    colorstr,
)
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    unset_deterministic,
)

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywhn, xywh2xyxy
from ultralytics.utils.metrics import bbox_iou, batch_probiou
from ultralytics.engine.espreg import (
    EWPRegLoss,
)
from ultralytics.engine.bpf import (
    bpf_pseudo_detections,
    compute_dwf_loss,
    merge_bpf_pseudo_labels,
    select_future_ignore_mask,
)
from ultralytics.engine.ewc import EWCLoss, load_ewc_state
from ultralytics.engine.l2 import L2Loss
from ultralytics.engine.nsgp import NSGP
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
        tuple[dict, dict]: module name -> (groups, n_components) components tensor,
            and module name -> (groups, n_components) explained-variance tensor.
    """
    components, eigen_values = {}, {}
    for name, entries in pca_cache.items():
        group_components, group_eigen_values = [], []
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
    return components, eigen_values


def _head_raw_output_to_list(head, raw_output, img_h, img_w, conf_threshold):
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
    batch, base_model, conf_threshold, filter_iou_threshold, device, base_model_pred=None
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
    Returns:
        Modified batch dict with merged labels
    """
    if base_model_pred is None:
        base_model_pred = base_model(batch['img'].to(device))
    head = _get_detect_head(base_model)
    img_h, img_w = batch['img'].shape[-2:]
    base_model_pred_list = _head_raw_output_to_list(
        head, base_model_pred, img_h, img_w, conf_threshold
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
        return tuple(loss_names)

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
                if not isinstance(_get_detect_head(self.bpf_source_model), Detect):
                    raise TypeError("BPF source model must use a Detect head")
            if self.args.bpf_dwf:
                if not self.args.bpf_interim_model:
                    raise ValueError("bpf_interim_model is required when bpf_dwf is enabled")
                self.bpf_interim_model, _ = load_checkpoint(self.args.bpf_interim_model, device=self.device)
                self.bpf_interim_model = self.bpf_interim_model.eval()
                self.bpf_interim_model.requires_grad_(False)
                if not isinstance(_get_detect_head(self.bpf_interim_model), Detect):
                    raise TypeError("BPF interim model must use a Detect head")
            if self.args.bpf_dwf and self.args.distillation:
                raise ValueError("BPF DwF and the KL distillation loss are mutually exclusive")
        # ============================== END: set up BPF ========================================================

        # ============================== MODIFIED: set up EWC loss ============================================
        if self.args.ewc:
            self.ewc_loss_weight = self.args.ewc_loss_weight
            ewc_state = load_ewc_state(self.args.importance_path, map_location=self.device)
            self.ewc_loss = EWCLoss(model=self.model, state=ewc_state)
        # ============================== END: set up EWC loss =================================================

        # ============================== MODIFIED: set up L2 regularization loss ==============================
        if self.args.l2:
            self.l2_loss_weight = self.args.l2_loss_weight
            self.l2_loss = L2Loss(model=self.model, ref_model=self.base_model)
        # ============================== END: set up L2 regularization loss ====================================

        # ============================== MODIFIED: set up ESPReg loss ============================================
        if self.args.espreg:
            self.espreg_loss_weight = self.args.espreg_loss_weight
            self.pca_cache = joblib.load(self.args.pca_cache_path)
            components, eigen_values = _extract_pca_spectra(self.pca_cache)
            self.espreg_loss = EWPRegLoss(self.model, self.base_model, module_names=self.pca_cache.keys(),
                                         components=components, eigen_values=eigen_values)
        # ============================== END: set up ESPReg loss =================================================

        # ============================== MODIFIED: set up NSGP ============================================
        if self.args.nsgp:
            # Reuse pca_cache if already loaded for ESPReg, otherwise load it
            if hasattr(self, 'pca_cache'):
                pca_cache_nsgp = self.pca_cache
            else:
                pca_cache_nsgp = joblib.load(self.args.pca_cache_path)
            components, eigen_values = _extract_pca_spectra(pca_cache_nsgp)
            self.nsgp_flexibility = getattr(self.args, 'nsgp_flexibility', 1.0)
            module_names = tuple(pca_cache_nsgp)
            self.nsgp_operator = NSGP(
                module_names=module_names,
                components=components,
                eigen_values=eigen_values,
                normalized_module_names=_get_nsgp_backbone_names(self.model, module_names),
            )
        # ============================== END: set up NSGP =================================================

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
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

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
                    if self.args.bpf and self.args.bpf_future:
                        model_pred, bpf_head_features = get_model_raw_output_and_features(
                            batch, self.model, self.device
                        )
                    else:
                        model_pred = get_model_raw_output(batch, self.model, self.device)
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
                            head_features=bpf_head_features,
                            batch=batch,
                            object_topk=self.args.bpf_object_topk,
                            attention_topk=self.args.bpf_attention_topk,
                            iou_threshold=self.args.bpf_future_iou,
                            attention_power=self.args.bpf_attention_power,
                        )
                    # ============================== END: BPF labels and future mask ==============================
                    
                    # ============================== MODIFIED: get det loss ===============================
                    loss, self.loss_items = self.model(batch, preds=model_pred)
                    # ============================== END: get det loss ====================================

                    # ============================== MODIFIED: make a copy of loss items ===============================
                    loss_items = dict(self.loss_items)  # criterion returns a dict of detached per-term scalars
                    # ============================== END: make a copy of loss items ====================================
                    
                    self.loss = loss.sum()

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
        nsgp_params = None
        parameters_before_step = None
        if self.args.nsgp:
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            nsgp_params = {name: param for name, param in model_to_use.named_parameters()}
            parameters_before_step = self.nsgp_operator.capture_parameters(nsgp_params)
        # ============================== END: capture NSGP parameters ================================
        self.scaler.step(self.optimizer)
        # ============================== MODIFIED: apply NSGP update projection ======================
        if self.args.nsgp:
            self.nsgp_operator.apply_parameter_projection(
                params_dict=nsgp_params,
                parameters_before_step=parameters_before_step,
                flexibility=self.nsgp_flexibility,
            )
        # ============================== END: apply NSGP update projection ===========================
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)