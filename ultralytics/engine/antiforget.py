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
# from ultralytics.engine.distillation import (
#     KDLoss,
# )
from ultralytics.engine.espreg import (
    EWPRegLoss,
)
from ultralytics.engine.ewc import EWCLoss
from ultralytics.engine.nsgp import NSGP
from ultralytics.nn.modules.head import Detect, OBB


def _get_detect_head(model):
    """Return the detection head module (Detect or OBB)."""
    head = model.model[-1]
    if isinstance(head, (Detect, OBB)):
        return head
    raise TypeError(f"Unsupported head type: {type(head)}")


def _head_raw_output_to_list(head, raw_output, img_h, img_w):
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
            conf_thres=0.25,
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
        conf_thres=0.25,
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


def merge_pseudo_labels_with_gt(batch, base_model, filter_iou_threshold, device):
    """
    Generate pseudo labels from base model and merge with GT labels, filtering by IoU.
    Supports both Detect (axis-aligned) and OBB (oriented) heads.
    
    Args:
        batch: Dict with keys 'img' (B, C, H, W), 'bboxes' (N, 4), 'cls' (N, 1), 'batch_idx' (N,)
        base_model: Base model for pseudo label generation
        filter_iou_threshold: IoU threshold for filtering pseudo labels
        device: Device to run inference on
    
    Returns:
        Modified batch dict with merged labels
    """
    base_model_pred = base_model(batch['img'].to(device))
    head = _get_detect_head(base_model)
    img_h, img_w = batch['img'].shape[-2:]
    base_model_pred = _head_raw_output_to_list(head, base_model_pred, img_h, img_w)
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

        pseudo_labels_img = base_model_pred[img_idx]  # (num_pseudo, 6) or (num_pseudo, 7)

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

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ============================== MODIFIED: set up base model ============================================
        self.base_model = deepcopy(self.model).eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        # ============================== END: set up base model =================================================

        # ============================== MODIFIED: set up EWC loss ============================================
        if self.args.ewc:
            self.ewc_loss_weight = self.args.ewc_loss_weight
            self.ewc_importance = torch.load(self.args.importance_path)['running_importance']
            self.ewc_loss = EWCLoss(self.model, self.base_model, importance=self.ewc_importance)
        # ============================== END: set up EWC loss =================================================

        # ============================== MODIFIED: set up ESPReg loss ============================================
        if self.args.espreg:
            self.espreg_loss_weight = self.args.espreg_loss_weight
            components, eigen_values = {}, {}
            self.pca_cache = joblib.load(self.args.pca_cache_path)
            for name in self.pca_cache.keys():
                _components = []
                _eigen_values = []
                for ig in range(len(self.pca_cache[name])):
                    _components.append(self.pca_cache[name][ig].components_)
                    _eigen_values.append(self.pca_cache[name][ig].explained_variance_)
                components[name], eigen_values[name] = torch.stack(_components), torch.stack(_eigen_values)
            self.espreg_loss = EWPRegLoss(self.model, self.base_model, module_names=self.pca_cache.keys(),
                                         components=components, eigen_values=eigen_values)
        # ============================== END: set up ESPReg loss =================================================

        # ============================== MODIFIED: set up NSGP ============================================
        if self.args.nsgp:
            components, eigen_values = {}, {}
            # Reuse pca_cache if already loaded for ESPReg, otherwise load it
            if hasattr(self, 'pca_cache'):
                pca_cache_nsgp = self.pca_cache
            else:
                pca_cache_nsgp = joblib.load(self.args.pca_cache_path)
            for name in pca_cache_nsgp.keys():
                _components = []
                _eigen_values = []
                for ig in range(len(pca_cache_nsgp[name])):
                    _components.append(pca_cache_nsgp[name][ig].components_)
                    _eigen_values.append(pca_cache_nsgp[name][ig].explained_variance_)
                components[name], eigen_values[name] = torch.stack(_components), torch.stack(_eigen_values)
            self.nsgp_flexibility = getattr(self.args, 'nsgp_flexibility', 1.0)
            self.nsgp_operator = NSGP(module_names=pca_cache_nsgp.keys(), components=components, eigen_values=eigen_values)
        # ============================== END: set up NSGP =================================================

        # ============================== MODIFIED: set up KD loss ================================================
        # if self.args.kd:
        #     self.teacher_model = deepcopy(self.model).eval()
        #     for p in self.teacher_model.parameters():
        #         p.requires_grad_(False)

        #     if self.args.distill_layers is not None:
        #         distill_layers = self.args.distill_layers
        #     else:
        #         if isinstance(self.args.freeze, list):
        #             distill_layers = [x for x in range(len(self.teacher_model.model)) if x not in self.args.freeze]
        #         elif isinstance(self.args.freeze, int):
        #             distill_layers = list(range(len(self.teacher_model.model))).remove(self.args.freeze)
        #         else:
        #             distill_layers = list(range(len(self.teacher_model.model)))

        #     self.kd_loss = KDLoss(self.model, self.teacher_model, distill_layers=distill_layers,
        #                           distiller=self.args.distiller, device=self.device)
            
        #     # calculate the number of extra parameters introduced by kd loss
        #     if self.kd_loss.distill_type.lower() == "feature":
        #         kd_params = sum(p.numel() for p in self.kd_loss.D_loss_fn.parameters())
        #         LOGGER.info(f"{colorstr('Feature-level KD params:')} {kd_params/1e6:.2f} M")
        #     else:
        #         LOGGER.info(f"{colorstr('Logit-level KD enabled, no extra sub-module parameters')}")
        # ============================== END: set up KD loss ======================================================

        # ============================== MODIFIED: set up Prototype Replay loss ===================================
        if self.args.proto_rp:
            prototypes_dict = torch.load(self.args.prototypes)
            self.prototypes = prototypes_dict["prototypes"] # List[torch.Tensor]
            for lid, x in enumerate(self.prototypes):
                # x: [num_prototypes, C*5*5+4*reg_max+nc+5*5]
                self.prototypes[lid] = x.to(self.device)
                self.prototypes[lid].requires_grad_(False)
            if self.args.proto_use_neg and "prototypes_neg" in prototypes_dict:
                self.prototypes_neg = prototypes_dict["prototypes_neg"] # List[torch.Tensor]
                for lid, x in enumerate(self.prototypes_neg):
                    # x: [num_prototypes, C*5*5+nc+5*5]
                    self.prototypes_neg[lid] = x.to(self.device)
                    self.prototypes_neg[lid].requires_grad_(False)  
            self.proto_rp_loss_weight = self.args.proto_rp_loss_weight
            
            # Check if we should use base_model for distillation instead of prototype supervision
            # proto_rp_use_base_model: if True, use base_model output as supervision; if False, use prototype's built-in supervision
            self.proto_rp_use_base_model = self.args.proto_rp_use_base_model
        # ============================== END: set up Prototype Relay loss =========================================
        
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
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

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
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

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
        if world_size > 1:
            self._setup_ddp(world_size)
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
            if self.args.ewc:
                self.ewc_loss.register_hook()
            
            if self.args.espreg or self.args.ewc:
                # Perform a forward in base model to hook out the base weights
                with torch.no_grad():
                    _ = self.base_model(torch.randn(1, 3, 640, 640).to(self.device))

            # if self.args.kd:
            #     self.kd_loss.register_hook() # Register hook for KD loss
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
                    # ============================== MODIFIED: generate pseudo labels ===============================
                    if self.args.pseudo_label:
                        batch = merge_pseudo_labels_with_gt(
                            batch, self.base_model, self.args.filter_iou_threshold, self.device
                        )
                    # ============================== END: generate pseudo labels ====================================
                    loss, self.loss_items = self.model(batch)
                    # ============================== MODIFIED: make a copy of loss items ===============================
                    loss_items = deepcopy(self.loss_items)
                    # ============================== END: make a copy of loss items ====================================
                    self.loss = loss.sum()

                    # ============================== MODIFIED: calculate ESPReg loss ===================================
                    if self.args.espreg:
                        _espreg_loss = self.espreg_loss.get_loss()
                        self.loss += (_espreg_loss * self.espreg_loss_weight)
                        loss_items = torch.cat([loss_items, torch.tensor([_espreg_loss], device=loss_items.device)])
                    # ============================== END: calculate ESPReg loss ========================================

                    # ============================== MODIFIED: calculate EWC loss ===================================
                    if self.args.ewc:
                        _ewc_loss = self.ewc_loss.get_loss()
                        self.loss += (_ewc_loss*self.ewc_loss_weight)
                        loss_items = torch.cat([loss_items, torch.tensor([_ewc_loss], device=loss_items.device)])
                    # ============================== END: calculate EWC loss ========================================

                    # ============================== MODIFIED: calculate distillation loss =============================
                    # if self.args.kd:
                    #     with torch.no_grad():
                    #         _ = self.teacher_model(batch["img"])
                        
                    #     _raw_kd_loss_weight = self.kd_loss.get_kd_weight(epoch=self.epoch, total_epochs=self.epochs)
                    #     _raw_kd_loss = self.kd_loss.get_loss() * _raw_kd_loss_weight
                    #     scale = batch["img"].shape[0]  # scale distillation loss by batch size
                    #     _kd_loss = _raw_kd_loss * scale

                    #     self.loss += _kd_loss
                    #     loss_items = torch.cat([loss_items, torch.tensor([_kd_loss], device=loss_items.device)])
                    # ============================== END: calculate distillation loss ====================================

                    # ============================== MODIFIED: replay prototypes =========================================
                    if self.args.proto_rp:
                        proto_losses = self.compute_proto_replay_loss(batch_idx=i)
                        if len(proto_losses) == 3:
                            cls_loss_proto, reg_loss_proto, cls_loss_neg_proto = proto_losses
                            loss_items = torch.cat([loss_items, torch.tensor([cls_loss_proto, reg_loss_proto, cls_loss_neg_proto], device=loss_items.device)])
                            self.loss += (cls_loss_proto + reg_loss_proto + cls_loss_neg_proto)*self.proto_rp_loss_weight
                        else:
                            cls_loss_proto, reg_loss_proto = proto_losses
                            loss_items = torch.cat([loss_items, torch.tensor([cls_loss_proto, reg_loss_proto], device=loss_items.device)])
                            self.loss += (cls_loss_proto + reg_loss_proto)*self.proto_rp_loss_weight
                    # ============================== END: replay prototypes ==============================================

                    if RANK != -1:
                        self.loss *= world_size                   
                    self.tloss = (
                        (self.tloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
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
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%13s" * 2 + "%13.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
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
            # if self.args.kd:
            #     self.kd_loss.remove_handle_() # Remove hook for KD loss
            # ============================== END: remove hook ================================================

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
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

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def restore_prototypes(self, prototypes, pad_mask):
        """
        Restore padded prototypes back to 5x5 feature maps and compute offsets.

        Args:
            prototypes (Tensor): Tensor of shape (N, C, 5, 5) containing prototype features.
            pad_mask (Tensor): Tensor of shape (N, 5, 5) indicating valid (1) and padded (0) regions.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Restored prototypes (N, C, 5, 5), offset_y (N,), offset_x (N,)
        """
        num_prototypes = prototypes.shape[0]
        in_channels = prototypes.shape[1]
        restored_prototypes = torch.zeros([num_prototypes, in_channels, 5, 5], device=self.device)
        offset_y_batch = torch.zeros([num_prototypes], device=self.device, dtype=torch.long)
        offset_x_batch = torch.zeros([num_prototypes], device=self.device, dtype=torch.long)

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

    def compute_proto_replay_loss(self, batch_idx: int):
        """
        Compute prototype replay classification and regression losses.
        Returns:
            Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]: 
            (cls_loss_proto, reg_loss_proto) if no negative prototypes,
            (cls_loss_proto, reg_loss_proto, cls_loss_neg_proto) if negative prototypes exist
        """
        detect = self.model.model[-1]
        detect.eval()
        cls_loss_proto = 0.0
        reg_loss_proto = 0.0
        cls_loss_neg_proto = 0.0
        reg = detect.cv2
        cls = detect.cv3
        reg_max = detect.reg_max
        
        # Check if negative prototypes exist
        has_neg_protos = (hasattr(self, 'prototypes_neg') and 
                         self.prototypes_neg is not None and
                         any(x is not None and torch.numel(x) > 0 for x in self.prototypes_neg))

        for lid in range(detect.nl):
            if self.prototypes[lid] is None or torch.numel(self.prototypes[lid]) == 0:
                continue

            in_channels = cls[lid][0].conv.in_channels
            reg_out_channels = reg[lid][-1].out_channels
            cls_out_channels = cls[lid][-1].out_channels

            pad_mask = self.prototypes[lid][:, in_channels * 5 * 5 + reg_out_channels + cls_out_channels :].reshape(-1, 5, 5)
            prototypes = self.prototypes[lid][:, : in_channels * 5 * 5].reshape(-1, in_channels, 5, 5)
            num_prototypes_all = prototypes.shape[0]

            # Use modulo to cycle through prototypes
            start_idx = (batch_idx * self.batch_size) % num_prototypes_all
            end_idx = start_idx + self.batch_size
            if end_idx <= num_prototypes_all:
                pad_mask = pad_mask[start_idx:end_idx]
                prototypes = prototypes[start_idx:end_idx]
            else:
                # Wrap around: take from start_idx to end, then from 0 to remaining
                pad_mask = torch.cat([pad_mask[start_idx:], pad_mask[:end_idx - num_prototypes_all]], dim=0)
                prototypes = torch.cat([prototypes[start_idx:], prototypes[:end_idx - num_prototypes_all]], dim=0)
            num_prototypes = prototypes.shape[0]

            prototypes, offset_y_batch, offset_x_batch = self.restore_prototypes(prototypes, pad_mask)

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

            if self.proto_rp_use_base_model:
                with torch.no_grad():
                    base_detect = self.base_model.model[-1]
                    base_reg = base_detect.cv2
                    base_cls = base_detect.cv3
                    base_reg_out = base_reg[lid](prototypes)
                    base_cls_out = base_cls[lid](prototypes)
                    reg_supervision_list = []
                    cls_supervision_list = []
                    for i in range(num_prototypes):
                        reg_supervision_list.append(base_reg_out[i, :, y_positions[i], x_positions[i]])
                        cls_supervision_list.append(base_cls_out[i, :, y_positions[i], x_positions[i]])
                    reg_supervision = torch.stack(reg_supervision_list)  # (num_prototypes, reg_out_channels)
                    cls_supervision = torch.stack(cls_supervision_list)  # (num_prototypes, cls_out_channels)
            else:
                reg_supervision = self.prototypes[lid][:, in_channels * 5 * 5 : in_channels * 5 * 5 + reg_out_channels]
                cls_supervision = self.prototypes[lid][:, in_channels * 5 * 5 + reg_out_channels : in_channels * 5 * 5 + reg_out_channels + cls_out_channels]
                # Use the same indices as prototypes
                start_idx = (batch_idx * self.batch_size) % num_prototypes_all
                end_idx = start_idx + self.batch_size
                if end_idx <= num_prototypes_all:
                    reg_supervision = reg_supervision[start_idx:end_idx]
                    cls_supervision = cls_supervision[start_idx:end_idx]
                else:
                    # Wrap around: take from start_idx to end, then from 0 to remaining
                    reg_supervision = torch.cat([reg_supervision[start_idx:], reg_supervision[:end_idx - num_prototypes_all]], dim=0)
                    cls_supervision = torch.cat([cls_supervision[start_idx:], cls_supervision[:end_idx - num_prototypes_all]], dim=0)

            cls_loss_proto += (F.binary_cross_entropy_with_logits(cls_out, cls_supervision.sigmoid())\
                -F.binary_cross_entropy_with_logits(cls_supervision, cls_supervision.sigmoid())) # min value of cls_loss_proto

            reg_supervision_softmax = F.softmax(reg_supervision.reshape(-1, reg_max), dim=1)  # [num_prototypes*4, reg_max]
            reg_loss_proto += F.cross_entropy(reg_out.reshape(-1, reg_max), reg_supervision_softmax)\
                - F.cross_entropy(reg_supervision.reshape(-1, reg_max), reg_supervision_softmax)
            
            # Process negative prototypes
            if hasattr(self, 'prototypes_neg') and self.prototypes_neg[lid] is not None and torch.numel(self.prototypes_neg[lid]) > 0:
                # Negative prototype format: [feat(C*25) | cls_valid_mask(nc) | pad_mask(25)]
                pad_mask_neg = self.prototypes_neg[lid][:, -25:].reshape(-1, 5, 5)  # Extract pad_mask from last 25 elements
                prototypes_neg = self.prototypes_neg[lid][:, :in_channels * 5 * 5].reshape(-1, in_channels, 5, 5)  # Extract features
                cls_valid_mask_neg = self.prototypes_neg[lid][:, in_channels * 5 * 5 : in_channels * 5 * 5 + cls_out_channels]
                num_prototypes_neg_all = prototypes_neg.shape[0]
                
                # Use modulo to cycle through negative prototypes
                start_idx_neg = (batch_idx * self.batch_size) % num_prototypes_neg_all
                end_idx_neg = start_idx_neg + self.batch_size
                if end_idx_neg <= num_prototypes_neg_all:
                    pad_mask_neg = pad_mask_neg[start_idx_neg:end_idx_neg]
                    prototypes_neg = prototypes_neg[start_idx_neg:end_idx_neg]
                    cls_valid_mask_neg = cls_valid_mask_neg[start_idx_neg:end_idx_neg]
                else:
                    # Wrap around: take from start_idx to end, then from 0 to remaining
                    pad_mask_neg = torch.cat([pad_mask_neg[start_idx_neg:], pad_mask_neg[:end_idx_neg - num_prototypes_neg_all]], dim=0)
                    prototypes_neg = torch.cat([prototypes_neg[start_idx_neg:], prototypes_neg[:end_idx_neg - num_prototypes_neg_all]], dim=0)
                    cls_valid_mask_neg = torch.cat([cls_valid_mask_neg[start_idx_neg:], cls_valid_mask_neg[:end_idx_neg - num_prototypes_neg_all]], dim=0)
                num_prototypes_neg = prototypes_neg.shape[0]
                
                prototypes_neg, offset_y_batch_neg, offset_x_batch_neg = self.restore_prototypes(prototypes_neg, pad_mask_neg)
                
                # Compute classification outputs for negative prototypes (no regression needed)
                cls_out_neg = cls[lid](prototypes_neg)
                
                y_positions_neg = offset_y_batch_neg + 2  # [num_prototypes_neg]
                x_positions_neg = offset_x_batch_neg + 2  # [num_prototypes_neg]
                
                cls_out_list_neg = []
                for i in range(num_prototypes_neg):
                    cls_out_list_neg.append(cls_out_neg[i, :, y_positions_neg[i], x_positions_neg[i]])
                cls_out_neg = torch.stack(cls_out_list_neg)  # (num_prototypes_neg, cls_out_channels)
                
                if self.proto_rp_use_base_model:
                    # Use base model output as supervision (only on historical classes)
                    with torch.no_grad():
                        base_detect = self.base_model.model[-1]
                        base_cls = base_detect.cv3
                        base_cls_out_neg = base_cls[lid](prototypes_neg)
                        base_cls_list_neg = []
                        for i in range(num_prototypes_neg):
                            base_cls_list_neg.append(base_cls_out_neg[i, :, y_positions_neg[i], x_positions_neg[i]])
                        cls_supervision_neg = torch.stack(base_cls_list_neg)  # (num_prototypes_neg, cls_out_channels)
                    
                    # Apply cls_valid_mask to mask out expanded class channels, only consider historical classes
                    # First sigmoid the supervision, then apply mask (consistent with positive prototypes)
                    cls_supervision_neg_sigmoid = cls_supervision_neg.sigmoid() * cls_valid_mask_neg  # Mask the supervision
                    cls_out_neg_masked = cls_out_neg * cls_valid_mask_neg  # Mask the output
                    
                    # Compute classification loss for negative prototypes (only on historical classes)
                    cls_loss_neg = F.binary_cross_entropy_with_logits(
                        cls_out_neg_masked,
                        cls_supervision_neg_sigmoid,
                        reduction='none'
                    )  # (num_prototypes_neg, cls_out_channels)
                else:
                    # Use zero labels for negative samples
                    cls_target_neg = torch.zeros_like(cls_out_neg)  # (num_prototypes_neg, cls_out_channels)
                    
                    # Apply cls_valid_mask to mask out expanded class channels, only consider historical classes
                    cls_out_neg_masked = cls_out_neg * cls_valid_mask_neg  # Mask the output
                    cls_target_neg_masked = cls_target_neg * cls_valid_mask_neg  # Mask the target (still zeros, but masked)
                    
                    # Compute classification loss for negative prototypes (only on historical classes)
                    cls_loss_neg = F.binary_cross_entropy_with_logits(
                        cls_out_neg_masked,
                        cls_target_neg_masked,
                        reduction='none'
                    )  # (num_prototypes_neg, cls_out_channels)
                
                # Only compute loss on valid (historical) classes
                valid_mask_sum = cls_valid_mask_neg.sum(dim=1)  # (num_prototypes_neg,)
                # Avoid division by zero
                valid_mask_sum = torch.clamp(valid_mask_sum, min=1.0)
                cls_loss_neg = (cls_loss_neg * cls_valid_mask_neg).sum(dim=1) / valid_mask_sum  # (num_prototypes_neg,)
                cls_loss_neg = cls_loss_neg.mean()  # Scalar
                
                # Accumulate negative prototype classification loss separately
                cls_loss_neg_proto += cls_loss_neg
            
        detect.train()
        if has_neg_protos:
            return cls_loss_proto, reg_loss_proto, cls_loss_neg_proto
        else:
            return cls_loss_proto, reg_loss_proto

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update.
        Modified to apply NSGP gradient projection after gradient clipping.
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        # ============================== MODIFIED: apply NSGP gradient projection ====================
        if self.args.nsgp:
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            params_dict = {name: param for name, param in model_to_use.named_parameters()}
            self.nsgp_operator.apply_projection(params_dict, self.nsgp_flexibility)
        # ============================== END: apply NSGP gradient projection ========================
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)