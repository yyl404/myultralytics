# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from typing import Any, Dict, List, Optional
import os
import cv2
import joblib

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.distillation import DistillationTrainer
from ultralytics.engine.vspreg import VSPRegTrainer, RealTimeMemoryMonitor, PCAHooker
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, TQDM, DEFAULT_CFG
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (Dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_metrics: Plot metrics from a CSV file.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (Dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (Dict): Preprocessed batch with normalized images.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: Optional[List[float]] = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (List[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)

class DistillationDetectionTrainer(DistillationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (Dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (Dict): Preprocessed batch with normalized images.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: Optional[List[float]] = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (List[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)


class DetectionPCAHooker(PCAHooker):
    def __init__(self, model, layers, bboxes=None, device="cuda", check=False):
        super().__init__(model, layers, device, check)
        self.bboxes = bboxes
        
    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        """Use tensor operation to accelerate the extraction of feature indices within bounding boxes
        """
        # Collect all bounding box information
        all_bboxes = []
        batch_ids = []
        for batch_id, _bboxes in enumerate(self.bboxes):
            for _bbox in _bboxes:
                all_bboxes.append(_bbox)
                batch_ids.append(batch_id)
        
        # Convert to tensor for vectorized calculation
        bbox_tensor = torch.tensor(all_bboxes, device=self.device)  # [N, 4]
        batch_tensor = torch.tensor(batch_ids, device=self.device)  # [N]
        
        # Scale bbox coordinates to feature map size
        feat_coords = bbox_tensor * torch.tensor([w_out, h_out, w_out, h_out], device=self.device)
        feat_x_min = torch.clamp(feat_coords[:, 0].int(), 0, w_out-1)
        feat_y_min = torch.clamp(feat_coords[:, 1].int(), 0, h_out-1)
        feat_x_max = torch.clamp(feat_coords[:, 2].int(), 0, w_out-1)
        feat_y_max = torch.clamp(feat_coords[:, 3].int(), 0, h_out-1)
        
        # Create coordinate grids
        h_indices = torch.arange(h_out, device=self.device)
        w_indices = torch.arange(w_out, device=self.device)
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        h_flat = h_grid.flatten()  # [h_out*w_out]
        w_flat = w_grid.flatten()  # [h_out*w_out]
        
        # Use broadcasting for vectorized mask calculation
        # Expand dimensions to support broadcasting: [N, 1] and [1, h_out*w_out]
        feat_y_min_exp = feat_y_min.unsqueeze(1)  # [N, 1]
        feat_y_max_exp = feat_y_max.unsqueeze(1)  # [N, 1]
        feat_x_min_exp = feat_x_min.unsqueeze(1)  # [N, 1]
        feat_x_max_exp = feat_x_max.unsqueeze(1)  # [N, 1]
        
        h_flat_exp = h_flat.unsqueeze(0)  # [1, h_out*w_out]
        w_flat_exp = w_flat.unsqueeze(0)  # [1, h_out*w_out]
        
        # Create mask for all bounding boxes [N, h_out*w_out]
        mask = ((h_flat_exp >= feat_y_min_exp) & (h_flat_exp <= feat_y_max_exp) & 
                (w_flat_exp >= feat_x_min_exp) & (w_flat_exp <= feat_x_max_exp))
        
        # Get all valid coordinate indices
        valid_coords = torch.nonzero(mask, as_tuple=False)  # [M, 2] where M is number of valid pixels
        
        if len(valid_coords) == 0:
            return
        
        # Calculate feature indices
        bbox_idx = valid_coords[:, 0]  # bbox indices
        coord_idx = valid_coords[:, 1]  # coordinate indices
        
        # Get corresponding coordinates
        valid_h = h_flat[coord_idx]
        valid_w = w_flat[coord_idx]
        valid_batch = batch_tensor[bbox_idx]
        
        # Calculate final feature indices
        sample_feature_indices = valid_batch * h_out * w_out + valid_h * w_out + valid_w

        # If too many features, randomly sample to accelerate the PCA computation
        if len(sample_feature_indices) > 100:
            sampled_indices = torch.randperm(len(sample_feature_indices), device=self.device)[:100]
            sample_feature_indices = sample_feature_indices[sampled_indices]
        
        return sample_feature_indices


class VSPRegDetectionTrainer(VSPRegTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def _do_pca(self):
        # Create PCA Hooker
        if self.args.projection_layers is not None:
            projection_layers = self.args.projection_layers
        else:
            if isinstance(self.args.freeze, list):
                projection_layers = [x for x in range(len(self.base_model.model)) if x not in self.args.freeze]
            elif isinstance(self.args.freeze, int):
                projection_layers = list(range(len(self.base_model.model)))
                projection_layers.remove(self.args.freeze)
            else:
                projection_layers = list(range(len(self.base_model.model)))
        self.pca_hooker = DetectionPCAHooker(self.base_model, projection_layers, device=self.device)

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
        
        if self.args.sample_images is None or self.args.sample_labels is None:
            LOGGER.warning("Insufficient data(images and labels) for local PCA, fallback to global PCA")
            return super()._do_pca()
        
        if isinstance(self.args.sample_images, list) or isinstance(self.args.sample_images, tuple):
            sample_files = []
            for _dir_img, _dir_label in zip(self.args.sample_images, self.args.sample_labels):
                sample_files.extend({
                    "image_file": os.path.join(_dir_img, x),
                    "label_file": os.path.join(_dir_label, x.replace(".jpg", ".txt"))
                } for x in os.listdir(_dir_img))
        else:
            sample_files = [
                {
                    "image_file": os.path.join(self.args.sample_images, x),
                    "label_file": os.path.join(self.args.sample_labels, x.replace(".jpg", ".txt"))
                } for x in os.listdir(self.args.sample_images)
            ]
        random.shuffle(sample_files)
        sample_files = sample_files[:self.args.pca_sample_num]
        
        memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)
        pbar = TQDM(sample_files, desc="PCA computing", total=len(sample_files))
        memory_monitor.set_progress_bar(pbar)
        memory_monitor.start_monitoring()

        for sample_file in pbar:
            image = cv2.imread(sample_file["image_file"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).float()
            image = image.to(self.device)

            bboxes = []
            with open(sample_file["label_file"], "r") as f:
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

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (Dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (Dict): Preprocessed batch with normalized images.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "vsp_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: Optional[List[float]] = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (List[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)