import numpy as np
import random
import os
import cv2
import joblib
import warnings
from copy import deepcopy
import math
import time

import torch
import torch.nn as nn
from torch import distributed as dist

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
from ultralytics.engine.distillation import (
    KDLoss,
)
from ultralytics.engine.vspreg import (
    RealTimeMemoryMonitor,
    VSPRegLoss,
    PCAHooker,
)


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

    def _do_pca(self):
        # Create PCA Hooker
        if self.args.projection_layers is not None:
            projection_layers = self.args.projection_layers
        else:
            if isinstance(self.args.freeze, list):
                projection_layers = [x for x in range(len(self.base_model.model)) if x not in self.args.freeze]
            elif isinstance(self.args.freeze, int):
                projection_layers = list(range(len(self.base_model.model))).remove(self.args.freeze)
            else:
                projection_layers = list(range(len(self.base_model.model)))
        self.pca_hooker = PCAHooker(self.base_model, projection_layers, device=self.device)

        if self.args.pca_cache_load_path:
            LOGGER.info(f"Loading PCA cache from {self.args.pca_cache_load_path}")
            if os.path.exists(self.args.pca_cache_load_path):
                with open(self.args.pca_cache_load_path, "rb") as f:
                    pca_cache = joblib.load(f)
                components = {}
                variances = {}
                means = {}
                for n, pca_operator in pca_cache.items():
                    self.pca_hooker.set_pca_operator(n, pca_operator)
                for n in self.pca_hooker.names:
                    component_matrix, variance_array, mean_array = self.pca_hooker.get_pca_results(n)
                    components[n] = component_matrix.to(self.device).detach()
                    variances[n] = variance_array.to(self.device).detach()
                    means[n] = mean_array.to(self.device).detach()
                if self.args.pca_cache_save_path:
                    LOGGER.info(f"Saving PCA cache to {self.args.pca_cache_save_path}")
                    with open(self.args.pca_cache_save_path, "wb") as f:
                        joblib.dump(pca_cache, f)
                return components, variances, means
            else:
                LOGGER.warning(f"PCA cache {self.args.pca_cache_load_path} is not loaded because it does not exist")

        memory_monitor = RealTimeMemoryMonitor(update_interval=0.2) # Monitor memory and CUDA memory usage
        pbar = TQDM(range(self.args.pca_sample_num), desc="PCA computing", total=self.args.pca_sample_num)
        memory_monitor.set_progress_bar(pbar)
        memory_monitor.start_monitoring()

        if self.args.sample_images is not None:
            if isinstance(self.args.sample_images, list) or isinstance(self.args.sample_images, tuple):
                sample_files = []
                for _dir in self.args.sample_images:
                    sample_files.extend(os.path.join(_dir, x) for x in os.listdir(_dir))
            else:
                sample_files = [os.path.join(self.args.sample_images, x) for x in os.listdir(self.args.sample_images)]
            random.shuffle(sample_files)
            sample_files = sample_files[:self.args.pca_sample_num]
            for i in pbar:
                image = cv2.imread(sample_files[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (640, 640))
                image = image.transpose(2, 0, 1) / 255.0
                image = torch.from_numpy(image).float()
                image = image.to(self.device)
                
                self.pca_hooker.register_hook()
                with torch.no_grad():
                    _ = self.base_model(image.unsqueeze(0))
                self.pca_hooker.remove_handle_()
        else:
            LOGGER.warning("No sample images provided, using random images for PCA")
            for i in pbar:
                image = torch.randn(3, 640, 640).to(self.device)
                self.pca_hooker.register_hook()
                with torch.no_grad():
                    _ = self.base_model(image.unsqueeze(0))
                self.pca_hooker.remove_handle_()
        
        memory_monitor.stop_monitoring()
        self.pca_hooker.clear_feature_cache()

        components = {}
        variances = {}
        means = {}
        if self.args.pca_cache_save_path:
            pca_cache = {}
        for n in self.pca_hooker.names:
            if self.args.pca_cache_save_path:
                pca_cache[n] = self.pca_hooker.get_pca_operator(n)
            component_matrix, variance_array, mean_array = self.pca_hooker.get_pca_results(n)
            components[n] = component_matrix.to(self.device).detach()
            variances[n] = variance_array.to(self.device).detach()
            means[n] = mean_array.to(self.device).detach()
        
        if self.args.pca_cache_save_path:
            LOGGER.info(f"Saving PCA cache to {self.args.pca_cache_save_path}")
            with open(self.args.pca_cache_save_path, "wb") as f:
                joblib.dump(pca_cache, f)

        return components, variances, means

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ============================== MODIFIED: set up VSPReg loss ============================================
        if self.args.vspreg:
            self.base_model = deepcopy(self.model).eval()
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            components, variances, biases = self._do_pca()
            self.vspreg_loss = VSPRegLoss(self.model, self.base_model, module_names=self.pca_hooker.names,
                                        components=components, variances=variances, means=biases)
        # ============================== END: set up VSPReg loss =================================================

        # ============================== MODIFIED: set up KD loss ================================================
        if self.args.kd:
            self.teacher_model = deepcopy(self.model).eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

            if self.args.distill_layers is not None:
                distill_layers = self.args.distill_layers
            else:
                if isinstance(self.args.freeze, list):
                    distill_layers = [x for x in range(len(self.teacher_model.model)) if x not in self.args.freeze]
                elif isinstance(self.args.freeze, int):
                    distill_layers = list(range(len(self.teacher_model.model))).remove(self.args.freeze)
                else:
                    distill_layers = list(range(len(self.teacher_model.model)))

            self.kd_loss = KDLoss(self.model, self.teacher_model, distill_layers=distill_layers,
                                  distiller=self.args.distiller, device=self.device)
            
            # calculate the number of extra parameters introduced by kd loss
            if self.kd_loss.distill_type.lower() == "feature":
                kd_params = sum(p.numel() for p in self.kd_loss.D_loss_fn.parameters())
                LOGGER.info(f"{colorstr('Feature-level KD params:')} {kd_params/1e6:.2f} M")
            else:
                LOGGER.info(f"{colorstr('Logit-level KD enabled, no extra sub-module parameters')}")
        # ============================== END: set up KD loss ======================================================
        
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
            if self.args.vspreg:
                self.vspreg_loss.register_hook() # Register hook for VSPRegLoss
                # Perform a forward in base model to hook out the base weights
                with torch.no_grad():
                    _ = self.base_model(torch.randn(1, 3, 640, 640).to(self.device))

            if self.args.kd:
                self.kd_loss.register_hook() # Register hook for KD loss
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
                    loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()

                    # ============================== MODIFIED: calculate VSPReg loss ===================================
                    if self.args.vspreg:
                        _vspreg_loss = self.vspreg_loss.get_loss()
                        self.loss += _vspreg_loss
                        loss_items = torch.cat([self.loss_items, torch.tensor([_vspreg_loss], device=self.loss_items.device)])
                    # ============================== END: calculate VSPReg loss ========================================

                    # ============================== MODIFIED: calculate distillation loss =============================
                    if self.args.kd:
                        with torch.no_grad():
                            _ = self.teacher_model(batch["img"])
                        
                        _raw_kd_loss_weight = self.kd_loss.get_kd_weight(epoch=self.epoch, total_epochs=self.epochs)
                        _raw_kd_loss = self.kd_loss.get_loss() * _raw_kd_loss_weight
                        scale = batch["img"].shape[0]  # scale distillation loss by batch size
                        _kd_loss = _raw_kd_loss * scale

                        self.loss += _kd_loss
                        loss_items = torch.cat([loss_items, torch.tensor([_kd_loss], device=loss_items.device)])
                    # ============================== END: calculate distillation loss ====================================

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
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
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
            if self.args.vspreg:
                self.vspreg_loss.remove_handle_() # Remove hook for VSPRegLoss
            if self.args.kd:
                self.kd_loss.remove_handle_() # Remove hook for KD loss
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