import torch
import torch.nn as nn
import numpy as np
import random
import os
import cv2
from sklearn.decomposition import IncrementalPCA
import threading
import psutil

import math
import time

from ultralytics.utils.tqdm import TQDM


class RealTimeMemoryMonitor:
    """实时内存监控器"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # 存储进度条引用
        
    def get_gpu_mem_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        """设置进度条引用"""
        self.pbar = pbar
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # 实时更新进度条描述
            if self.pbar is not None:
                self.pbar.set_description(f"Processing images - GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        """获取当前状态"""
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


import warnings
import torch.nn.functional as F
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


_YOLOV8_PROJECTION_LAYER = [i for i in range(9, 22)]


class SubspaceProjectionLoss:
    def __init__(self, model_update, model_base, projection_module_names, projections, biases):
        self.model_update = model_update
        self.model_base = model_base
        self.projection_module_names = projection_module_names
        self.projections = projections
        self.biases = biases
        
        for _matrix in projections.values():
            if _matrix.requires_grad:
                print("Projections are not frozen, this may cause instability in training.")
                break
        for _bias in biases.values():
            if _bias.requires_grad:
                print("Biases are not frozen, this may cause instability in training.")
                break

        # 缓存YOLO的任务类型（detect/segment/pose），以及类别数
        self.task = getattr(model_update, 'task', 'detect')
        self.nc = getattr(model_update, 'nc', None)

        # 对应的 hook modules
        self.update_modules = {}
        self.base_modules = {}
        for n, m in model_update.named_modules():
            if n in projection_module_names:
                assert isinstance(m, nn.Conv2d), f"module type {type(m)} is not supported"
                self.update_modules[n] = m
        for n, m in model_base.named_modules():
            if n in projection_module_names:
                assert isinstance(m, nn.Conv2d), f"module type {type(m)} is not supported"
                self.base_modules[n] = m
        
        # 统一缓存 & hook handle
        self.update_weights, self.base_weights, self._handles = {}, {}, []

    def register_hook(self):
        self.remove_handle_()
        for n in self.projection_module_names:
            assert n in self.update_modules.keys(), f"{n} not found in training model"
            assert n in self.base_modules.keys(), f"{n} not found in base model"
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            dict_w[n] = module.weight.reshape(module.weight.shape[0], -1)
        return fn

    def remove_handle_(self):
        """ 训练完成/不再需要约束时，移除所有hook，释放显存，防止内存泄漏。 """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self):
        loss = 0
        for n, p in self.projections.items():
            assert n in self.update_weights.keys(), f"{n} not found in training model"
            assert n in self.base_weights.keys(), f"{n} not found in base model"
            u_w = self.update_weights[n]
            b_w = self.base_weights[n]
            b = self.biases[n]
            delta_w = u_w - b_w
            p = p.to(delta_w.device, delta_w.dtype)
            loss += (delta_w @ p).norm(dim=1).mean()
        return loss
    
    def set_projections_and_biases(self, projections, biases):
        self.projections = projections
        self.biases = biases


class ConvInputPCAHooker:
    def __init__(self, model, layers_hooked):
        self.model = model
        self.layers_hooked = layers_hooked
        self.modules = {}
        self.bboxes = []
        self.pca_operators = {}
        self.len = 0

        def _match(n, m, lid):
            return f"model.{lid}" in n and isinstance(m, nn.Conv2d)

        self.feature_caches, self._handles = {}, []

        for lid in layers_hooked:
            for n, m in model.named_modules():
                if _match(n, m, lid):
                    k, c_in = m.kernel_size, m.in_channels
                    self.modules[n] = m
                    self.pca_operators[n] = IncrementalPCA(n_components=c_in*k[0]*k[1])
                    self.feature_caches[n] = []
                    self.len += 1

    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(self.pca_operators[n], self.feature_caches[n])))

    def _hook(self, pca_operator, feature_cache):
        def fn(module, feat_in, feat_out):
            assert isinstance(module, nn.Conv2d), f"Module type {type(module)} is not supported"
            assert module.groups == 1, "Group convolution is not supported"
            k, s, p = module.kernel_size, module.stride, module.padding
            
            feat_in = feat_in[0]
            bs, nc_in, h, w = feat_in.shape
            
            if p[0] > 0 or p[1] > 0:
                feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
            else:
                feat_in_padded = feat_in
            
            # 使用unfold操作按照卷积核大小展开感受野
            # 第一次unfold: 在高度维度展开，结果形状
            # [bs, nc_in, h, w] --> [bs, nc_in, h_out, w, k[0]]
            feat_unfold_h = feat_in_padded.unfold(2, k[0], s[0])
            # 第二次unfold: 在宽度维度展开，结果形状
            # [bs, nc_in, h_out, w, k[0]] --> [bs, nc_in, h_out, w_out, k[0], k[1]]
            feat_unfold = feat_unfold_h.unfold(3, k[1], s[1])
            
            # 重新排列维度: [bs, nc_in, h_out, w_out, k[0], k[1]] -> [nc_in, k[0], k[1], bs, h_out, w_out]
            feat_unfold = feat_unfold.permute(1, 4, 5, 0, 2, 3).contiguous()
            
            # 计算感受野数量
            h_out, w_out = feat_unfold.shape[4], feat_unfold.shape[5]
            
            # 重塑为 [nc_in*k*k, bs*h_out*w_out] 格式
            feat_reshaped = feat_unfold.view(nc_in*k[0]*k[1], bs*h_out*w_out)

            # 下面这一段是用于检查展开后的卷积计算结果和原本的输出是否一致
            # weight = module.weight.reshape(module.weight.shape[0], -1)
            # feat_out_reshaped = weight.unsqueeze(0) @ feat_reshaped

            # # 将feat_out_reshaped变换回标准形状 [bs, nc_out, num_perception] -> [bs, nc_out, h_out, w_out]
            # nc_out = module.weight.shape[0]
            # feat_out_reshaped_reversed = feat_out_reshaped.view(nc_out, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()
            
            # # 添加bias（如果存在）
            # if module.bias is not None:
            #     feat_out_reshaped_reversed = feat_out_reshaped_reversed + module.bias.view(1, -1, 1, 1)
            
            # print(torch.mean((feat_out - feat_out_reshaped_reversed).abs()))
            
            # 使用张量运算加速边界框内特征索引提取
            if not self.bboxes or not any(self.bboxes):
                return
            
            # 收集所有边界框信息
            all_bboxes = []
            batch_ids = []
            for batch_id, _bboxes in enumerate(self.bboxes):
                for _bbox in _bboxes:
                    all_bboxes.append(_bbox)
                    batch_ids.append(batch_id)
            
            if not all_bboxes:
                return
            
            # 转换为张量进行向量化计算
            bbox_tensor = torch.tensor(all_bboxes, device=feat_in.device)  # [N, 4]
            batch_tensor = torch.tensor(batch_ids, device=feat_in.device)  # [N]
            
            # 缩放bbox坐标到特征图尺寸
            feat_coords = bbox_tensor * torch.tensor([w_out, h_out, w_out, h_out], device=feat_in.device)
            feat_x_min = torch.clamp(feat_coords[:, 0].int(), 0, w_out-1)
            feat_y_min = torch.clamp(feat_coords[:, 1].int(), 0, h_out-1)
            feat_x_max = torch.clamp(feat_coords[:, 2].int(), 0, w_out-1)
            feat_y_max = torch.clamp(feat_coords[:, 3].int(), 0, h_out-1)
            
            # 创建所有可能的坐标网格
            h_indices = torch.arange(h_out, device=feat_in.device)
            w_indices = torch.arange(w_out, device=feat_in.device)
            h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
            
            # 展平坐标网格
            h_flat = h_grid.flatten()  # [h_out*w_out]
            w_flat = w_grid.flatten()  # [h_out*w_out]
            
            # 使用广播进行向量化掩码计算
            # 扩展维度以支持广播: [N, 1] 和 [1, h_out*w_out]
            feat_y_min_exp = feat_y_min.unsqueeze(1)  # [N, 1]
            feat_y_max_exp = feat_y_max.unsqueeze(1)  # [N, 1]
            feat_x_min_exp = feat_x_min.unsqueeze(1)  # [N, 1]
            feat_x_max_exp = feat_x_max.unsqueeze(1)  # [N, 1]
            
            h_flat_exp = h_flat.unsqueeze(0)  # [1, h_out*w_out]
            w_flat_exp = w_flat.unsqueeze(0)  # [1, h_out*w_out]
            
            # 创建所有边界框的掩码 [N, h_out*w_out]
            mask = ((h_flat_exp >= feat_y_min_exp) & (h_flat_exp <= feat_y_max_exp) & 
                    (w_flat_exp >= feat_x_min_exp) & (w_flat_exp <= feat_x_max_exp))
            
            # 获取所有有效的坐标索引
            valid_coords = torch.nonzero(mask, as_tuple=False)  # [M, 2] where M is number of valid pixels
            
            if len(valid_coords) == 0:
                return
            
            # 计算特征索引
            bbox_idx = valid_coords[:, 0]  # 边界框索引
            coord_idx = valid_coords[:, 1]  # 坐标索引
            
            # 获取对应的坐标
            valid_h = h_flat[coord_idx]
            valid_w = w_flat[coord_idx]
            valid_batch = batch_tensor[bbox_idx]
            
            # 计算最终的特征索引
            bbox_feature_indices = valid_batch * h_out * w_out + valid_h * w_out + valid_w
            
            # 如果特征太多，随机采样
            if len(bbox_feature_indices) > 100:
                sampled_indices = torch.randperm(len(bbox_feature_indices), device=feat_in.device)[:100]
                bbox_feature_indices = bbox_feature_indices[sampled_indices]
                
            # 提取选中的特征
            feat_sampled = feat_reshaped[:, bbox_feature_indices]
            feature_cache.append(feat_sampled)

            if sum([x.shape[1] for x in feature_cache]) >= pca_operator.n_components:
                feat_sampled = torch.cat(feature_cache, dim=1)
                feature_cache.clear()
                pca_operator.partial_fit(feat_sampled.cpu().T.numpy())
            
        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_feature_cache(self):
        for cache in self.feature_caches.values():
            cache.clear()
    
    def get_pca_results(self, name):
        try:
            return (self.pca_operators[name].components_, 
                    self.pca_operators[name].explained_variance_, 
                    self.pca_operators[name].mean_)
        except AttributeError:
            print(f"Warning: Too few samples to fit PCA in module {name}. Skip this module.")
            n_components = self.pca_operators[name].n_components
            return (np.zeros((n_components, n_components)), 
                    np.zeros(n_components), 
                    np.zeros(n_components))


class ProjectionLossTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the ProjectionLossTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        # ============================== MODIFIED: add projection parameter ===========================================
        # 指定基础模型和投影层
        self.base_model = overrides["base_model"]
        if overrides and "projection_layers" in overrides:
            self.projection_layers = overrides['projection_layers']
        else:
            self.projection_layers = _YOLOV8_PROJECTION_LAYER # default to _YOLOV8_PROJECTION_LAYER
        if overrides and "sample_images" in overrides and "sample_labels" in overrides:
            self.sample_images = overrides['sample_images']
            self.sample_labels = overrides['sample_labels']
        if overrides and "pca_sample_num" in overrides:
            self.pca_sample_num = overrides['pca_sample_num']
        else:
            self.pca_sample_num = 100
        # ============================== END: add projection parameter ================================================
        super().__init__(cfg, overrides, _callbacks)
    
    def _do_pca(self):
        image_files = os.listdir(self.sample_images)
        random.shuffle(image_files)
        image_files = image_files[:self.pca_sample_num]

        label_files = [image_file.replace(".jpg", ".txt") for image_file in image_files]
        
        # 创建实时内存监控器
        memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)
        
        # 创建进度条
        pbar = TQDM(zip(image_files, label_files), desc="PCA computing", total=len(image_files))
        
        # 将进度条传递给监控器
        memory_monitor.set_progress_bar(pbar)
        
        # 开始监控（监控器会自动更新进度条）
        memory_monitor.start_monitoring()
        
        for image_file, label_file in pbar:
            image = cv2.imread(os.path.join(self.sample_images, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).float()
            image = image.to(self.device)

            bboxes = []
            with open(os.path.join(self.sample_labels, label_file), "r") as f:
                labels = f.readlines()
                for _label in labels:
                    _label = _label.strip().split()
                    x, y, w, h = float(_label[1]), float(_label[2]), float(_label[3]), float(_label[4])
                    x_min, y_min, x_max, y_max = x - w/2, y - h/2, x + w/2, y + h/2
                    bboxes.append([x_min, y_min, x_max, y_max])
            
            self.pca_hooker.set_bboxes([bboxes])
            self.pca_hooker.register_hook()
            with torch.no_grad():
                _ = self.base_model(image.unsqueeze(0))
            self.pca_hooker.remove_handle_()

        # 停止内存监控
        memory_monitor.stop_monitoring()
        
        self.pca_hooker.clear_feature_cache()

        projections = {}
        biases = {}
        for n in self.pca_hooker.names:
            components, lambdas, mean = self.pca_hooker.get_pca_results(n)
            projections[n] = torch.from_numpy(components).to(self.device).T @ torch.diag(torch.sqrt(torch.from_numpy(lambdas).to(self.device))).requires_grad_(False)
            biases[n] = torch.from_numpy(mean).to(self.device).requires_grad_(False)

        return projections, biases
    
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ============================== MODIFIED: add projection parameter ============================================
        # teacher 放到相同 device
        self.base_model = self.base_model.to(self.device).eval()
        for p in self.base_model.parameters():  # 彻底冻结
            p.requires_grad_(False)
        # 创建PCA钩子管理器
        self.pca_hooker = ConvInputPCAHooker(self.base_model, self.projection_layers)
        # 进行PCA
        projections, biases = self._do_pca()
        # 创建投影损失实例
        self.projection_loss = SubspaceProjectionLoss(self.model, self.base_model, projection_module_names=self.pca_hooker.names,
                                                      projections=projections, biases=biases) # 初始化投影损失实例
        # ============================== END: add projection parameter ==================================================

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

            # ============================== MODIFIED: add projection parameter ===========================================
            self.projection_loss_sum = 0.0  # 投影损失
            self.or_loss_sum = 0.0  # 原始损失
            self.loss_count = 0     # epoch的step数
            self.projection_loss.register_hook() # 为基础模型注册hook函数
            # ============================== END: add projection parameter ================================================

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
                    if RANK != -1:
                        self.loss *= world_size
                    # ============================== MODIFIED: add projection parameter ===================================
                    with torch.no_grad():
                        _ = self.base_model(batch['img'])

                    bs = batch['img'].shape[0]                  # 本进程 mini-batch
                    ws = world_size if RANK != -1 else 1        # DDP 时为 8、16…；单机=1
                    scale = bs * ws
                    
                    # 获取蒸馏损失及其衰减权重
                    raw_p_loss = self.projection_loss.get_loss()
                    
                    self.projection_loss_sum += raw_p_loss.item()
                    self.or_loss_sum += (self.loss.detach().item()) / scale if scale else 0 # 原始损失
                    self.loss_count += 1

                    self.p_loss = raw_p_loss * scale
                    # print(f"or_loss: {self.loss / scale:.2f}, kd_loss: {raw_d_loss:.2f}, ratio: {self.d_loss / self.loss:.2f} kd_weight: {raw_d_loss_weight:.6f}")

                    self.loss += self.p_loss
                    
                    # 将投影损失添加到loss_items中用于显示
                    # 创建包含投影损失的loss_items
                    loss_items_with_projection = torch.cat([self.loss_items, torch.tensor([raw_p_loss.item()], device=self.loss_items.device)])
                    
                    self.tloss = (
                        (self.tloss * i + loss_items_with_projection) / (i + 1) if self.tloss is not None else loss_items_with_projection
                    )
                    # ============================== END: add projection parameter ========================================

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

            # ============================== MODIFIED: add distillation parameter ===========================================
            self.projection_loss.remove_handle_() # 移除教师模型的hook函数
 
            if self.loss_count: # 避免loss_count为0
                prj_mean = self.projection_loss_sum / self.loss_count
                or_mean = self.or_loss_sum / self.loss_count
                ratio = prj_mean / or_mean if or_mean else 0
                # 保存到 trainer 上，给回调用
                self.tb_prj_mean = prj_mean
                self.tb_or_mean = or_mean
                self.tb_prj_ratio = ratio
                print(f"prj_mean: {prj_mean:.2f}, or_mean: {or_mean:.2f}, ratio: {ratio:.2f}")
                self.run_callbacks("on_show_projection_loss")    # 触发回调
            # ============================== END: add distillation parameter ================================================

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