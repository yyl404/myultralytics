import numpy as np
import random
import os
import cv2
# from sklearn.decomposition import IncrementalPCA
from gpu_pca import IncrementalPCAonGPU as IncrementalPCA # 2:32
import threading
import psutil
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


class RealTimeMemoryMonitor:
    """Real-time memory monitor"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # store progress bar reference
        
    def get_gpu_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        self.pbar = pbar
    
    def _monitor_loop(self):
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # Real-time update progress bar description
            if self.pbar is not None:
                self.pbar.set_description(f"PCA computing - GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


class VSPRegLoss:
    """ Variance-Scaled Projection Regularization loss(VSP-Reg)
    Regularization loss based on variance-scaled projection length of weight updates in the principal component subspace
    """
    def __init__(self, model_update, model_base, module_names, components, variances, means, alpha=0.9, beta=0.1):
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = module_names
        self.components = components
        self.variances = variances
        self.means = means
        self.alpha = alpha
        self.beta = beta
        
        # Freeze PCA results
        for _component, _scale, _bias in zip(components.values(), variances.values(), means.values()):
            _component.requires_grad_(False)
            _scale.requires_grad_(False)
            _bias.requires_grad_(False)

        self.update_modules, self.base_modules = {}, {}
        for n, m in model_update.named_modules():
            if n in module_names:
                self.update_modules[n] = m
        for n, m in model_base.named_modules():
            if n in module_names:
                self.base_modules[n] = m
        
        self.update_weights, self.base_weights = {}, {}
        self._handles = []

    def register_hook(self):
        self.remove_handle_()
        for n in self.module_names:
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    # If bias exists, augment the weight matrix to represent convolution
                    # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                    weight = module.weight.reshape(module.weight.shape[0], -1)
                    bias = module.bias.reshape(module.bias.shape[0], -1)
                    dict_w[n] = torch.cat([weight, bias], dim=1)
                else:
                    dict_w[n] = module.weight.reshape(module.weight.shape[0], -1)
            else:
                raise RuntimeError(f"Module {n}'s type {type(module)} is not supported")
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self):
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [c_out, c_in*k*k]
            scale = torch.sqrt(self.variances[n]) # [c_in*k*k]
            bias = self.means[n] # [c_in*k*k]
            
            update_w = self.update_weights[n]
            base_w = self.base_weights[n]
            delta_w = update_w - base_w
            
            proj = proj.to(delta_w.device, delta_w.dtype)
            bias = bias.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)

            if proj.shape[1] < delta_w.shape[1]:
                # when the sampled features are less than feature dimension, we need to pad the components matrix and scales array
                proj = torch.cat([proj, torch.zeros((proj.shape[0], delta_w.shape[1] - proj.shape[1]), device=proj.device, dtype=proj.dtype)], dim=1)
                scale = torch.cat([scale, torch.zeros((delta_w.shape[1] - scale.shape[0]), device=scale.device, dtype=scale.dtype)], dim=0)
            
            loss += self.alpha * (delta_w @ proj.T @ torch.diag(scale)).norm(dim=1).mean() \
                + self.beta * (delta_w @ bias.unsqueeze(-1)).square().mean()
        return loss
    
    def set_parameters(self, components, variances, means, alpha=0.9, beta=0.1):
        self.components = components
        self.variances = variances
        self.means = means
        # Freeze PCA results
        for _component, _scale, _bias in zip(components.values(), variances.values(), means.values()):
            _component.requires_grad_(False)
            _scale.requires_grad_(False)
            _bias.requires_grad_(False)
        
        self.alpha = alpha
        self.beta = beta


class PCAHooker:
    def __init__(self, model, layers, device="cuda", check=False):
        self.model = model
        self.layers = layers
        self.modules = {}
        self.pca_operators = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        self.check = check

        def _match(n, m, lid):
            # dfl layer is not a trainable layer
            return f"model.{lid}." in n and "dfl" not in n and isinstance(m, nn.Conv2d)

        self.feature_caches, self._handles = {}, []

        for lid in layers:
            for n, m in model.named_modules():
                if _match(n, m, lid):
                    k, c_in = m.kernel_size, m.in_channels
                    self.modules[n] = m
                    if m.bias is not None:
                        # If bias exists, augment the feature matrix to represent convolution
                        # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                        self.pca_operators[n] = IncrementalPCA(n_components=c_in*k[0]*k[1]+1)
                    else:
                        self.pca_operators[n] = IncrementalPCA(n_components=c_in*k[0]*k[1])
                    self.feature_caches[n] = []

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        # randomly sample feature indices
        sample_feature_indices = torch.randperm(bs*h_out*w_out, device=self.device)[:100]
        return sample_feature_indices

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(n)))

    def _hook(self, module_name):
        def fn(module, feat_in, feat_out):
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise RuntimeError(f"Group convolution is not supported")
                k, s, p = module.kernel_size, module.stride, module.padding

                feat_in = feat_in[0] # module may accept multiple input feats, and we only extract the first
                bs, nc_in, h, w = feat_in.shape
                
                if p[0] > 0 or p[1] > 0:
                    feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
                else:
                    feat_in_padded = feat_in
                
                # Use the sliding window with the same settings as convolutin kernels
                # to unfold input features into a sequence of vectors
                # [bs, nc_in, h, w] --> [bs, nc_in, h_out, w, k[0]]
                feat_unfold_h = feat_in_padded.unfold(2, k[0], s[0])
                # [bs, nc_in, h_out, w, k[0]] --> [bs, nc_in, h_out, w_out, k[0], k[1]]
                feat_unfold = feat_unfold_h.unfold(3, k[1], s[1])
                # Permute the dims: [bs, nc_in, h_out, w_out, k[0], k[1]] -> [nc_in, k[0], k[1], bs, h_out, w_out]
                feat_unfold = feat_unfold.permute(1, 4, 5, 0, 2, 3).contiguous()
                # Squeeze: [nc_in, k[0], k[1], bs, h_out, w_out] --> [nc_in*k[0]*k[1], bs*h_out*w_out]
                h_out, w_out = feat_unfold.shape[4], feat_unfold.shape[5]
                feat_reshaped = feat_unfold.view(nc_in*k[0]*k[1], bs*h_out*w_out)
                if module.bias is not None:
                    # If bias exists, augment the feature matrix to represent convolution
                    # as a homogeneous linear transformation: [W; b] * [x^T; 1]^T = W*x + b
                    feat_reshaped = torch.concat([feat_reshaped, torch.ones((1, bs*h_out*w_out), device=feat_reshaped.device)], dim=0)

                # The following code is used to check whether the unfolding can represent the convolution operation
                # as a homogeneous linear transformation
                if self.check:
                    weight = module.weight.reshape(module.weight.shape[0], -1)
                    if module.bias is not None:
                        bias = module.bias.reshape(module.bias.shape[0], -1) # [c_out, 1]
                        weight = torch.concat([weight, bias], dim=1) # [c_out, c_in*k*k+1]
                    feat_out_reshaped = weight @ feat_reshaped # [c_out, bs*h_out*w_out]

                    # [c_out, bs*h_out*w_out] --> [c_out, bs, h_out, w_out] --> [bs, c_out, h_out, w_out]
                    feat_out_reshaped_reversed = feat_out_reshaped.view(-1, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()
                    LOGGER.info(f"Module {module_name}'s unfolding error: {torch.mean((feat_out - feat_out_reshaped_reversed).abs())}")

                sample_feature_indices = self._get_sample_feature_indices(bs, h_out, w_out)
                # Extract selected features for PCA computation
                feat_sampled = feat_reshaped[:, sample_feature_indices] # [c_in*k*k, len(sample_feature_indices)]
                
                feature_cache = self.feature_caches[module_name]
                feature_cache.append(feat_sampled)

                pca_operator = self.pca_operators[module_name]
                if sum([x.shape[1] for x in feature_cache]) >= pca_operator.n_components: # Incremental PCA requires the first batch's size is larger than n_components
                    feat_sampled = torch.cat(feature_cache, dim=1)
                    feature_cache.clear()
                    # pca_operator.partial_fit(feat_sampled.cpu().T.numpy())
                    pca_operator.partial_fit(feat_sampled.T)
            else:
                raise RuntimeError(f"Module type {type(module)} is not supported")

        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_feature_cache(self):
        for n, cache in self.feature_caches.items():
            if len(cache) > 0:
                # before clearing the cache, we need to fit the PCA operator with the final batch of features
                if not hasattr(self.pca_operators[n], 'components_'):
                    # If components_ is not calculated, it means the PCA operator has not been called yet, and that
                    # means the number of features in this cache has never reached n_components, so we need to
                    # reduce n_components to the number of features in this cache.
                    LOGGER.warning(f"Too few samples to fit PCA in module {n}. Could result in instability.")
                    self.pca_operators[n].n_components = torch.cat(cache, dim=1).shape[1]
                # self.pca_operators[n].partial_fit(torch.cat(cache, dim=1).cpu().T.numpy())
                self.pca_operators[n].partial_fit(torch.cat(cache, dim=1).T)
            cache.clear()
    
    def get_pca_results(self, name):
        return (self.pca_operators[name].components_, 
                self.pca_operators[name].explained_variance_, 
                self.pca_operators[name].mean_)

    def get_pca_operator(self, name):
        return self.pca_operators[name]
    
    def set_pca_operator(self, name, pca_operator):
        self.pca_operators[name] = pca_operator


class VSPRegTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the VSPRegTrainer class.

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
        self.base_model = deepcopy(self.model).eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        # Do PCA
        components, variances, biases = self._do_pca()
        # Initialize VSPRegLoss
        self.vspreg_loss = VSPRegLoss(self.model, self.base_model, module_names=self.pca_hooker.names,
                                      components=components, variances=variances, means=biases)
        # ============================== END: set up VSPReg loss ==================================================

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
            self.vspreg_loss.register_hook() # Register hook for VSPRegLoss
            # Perform a forward in base model to hook out the base weights
            with torch.no_grad():
                _ = self.base_model(torch.randn(1, 3, 640, 640).to(self.device))
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
                    _vspreg_loss = self.vspreg_loss.get_loss()
                    self.loss += _vspreg_loss
                    loss_items = torch.cat([self.loss_items, torch.tensor([_vspreg_loss], device=self.loss_items.device)])
                    # ============================== END: calculate VSPReg loss ========================================
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
            self.vspreg_loss.remove_handle_() # Remove hook for VSPRegLoss
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