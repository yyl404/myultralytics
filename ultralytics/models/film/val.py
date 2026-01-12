# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch

from ultralytics.data import build_dataloader
from ultralytics.data.dataset_json import JSONAttributeDataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks_film import DetectionModelFiLM
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model


class FiLMValidator(DetectionValidator):
    """Validator for YOLOFiLM models that handles attribute text inputs.
    
    This validator extends DetectionValidator to support validation with attribute text,
    which is required for FiLM-based models that modulate features based on text attributes.
    
    Examples:
        >>> from ultralytics.models.film import FiLMValidator
        >>> validator = FiLMValidator(args=args)
        >>> stats = validator(model=model)
    """
    
    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLOFiLM validation.
        
        This method extends the base preprocess to ensure attr_text is properly
        handled in the batch.
        
        Args:
            batch (dict[str, Any]): Batch containing images, annotations, and attr_text.
            
        Returns:
            (dict[str, Any]): Preprocessed batch with attr_text preserved.
        """
        # Call parent preprocess to handle images and other tensors
        batch = super().preprocess(batch)
        
        # Ensure attr_text is a list (it's already a list from collate_fn, but double-check)
        if 'attr_text' in batch:
            if not isinstance(batch['attr_text'], list):
                batch['attr_text'] = [batch['attr_text']] if isinstance(batch['attr_text'], str) else []
        else:
            # If no attr_text, use default empty list (model will handle it)
            batch['attr_text'] = []
        
        return batch
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """Build JSONAttributeDataset for validation.
        
        This method overrides the parent's build_dataset to use JSONAttributeDataset
        instead of the standard YOLODataset, since we're using JSON files instead of
        image directories.
        
        Args:
            img_path: Path to the JSON file (not an image directory).
            mode: Dataset mode ("val" or "test").
            batch: Batch size.
            
        Returns:
            JSONAttributeDataset instance.
        """
        # Note: img_path is actually the JSON file path
        return JSONAttributeDataset(
            img_path=None,  # Don't scan directories
            json_path=img_path,
            data=self.data,
            task="detect",
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # No augmentation during validation
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            classes=None  # Use all classes
        )
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="val"):
        """Get dataloader for JSON-based dataset.
        
        This method overrides the parent's get_dataloader to use our custom
        build_dataset method that handles JSON files.
        
        Args:
            dataset_path: Path to the JSON file.
            batch_size: Batch size.
            rank: Process rank for distributed training.
            mode: Dataset mode ("val" or "test").
            
        Returns:
            DataLoader instance.
        """
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,  # No shuffling during validation
            rank=rank
        )
        return loader
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation process for FiLM models with attribute text support.
        
        This method extends the base __call__ to ensure attr_texts are passed to the model
        during inference. It follows the same structure as BaseValidator.__call__ but
        modifies the inference step to include attr_texts.
        
        Args:
            trainer: Trainer object containing the model to validate.
            model: Model to validate if not using a trainer.
            
        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")
            
            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))
            
            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            
            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))
        
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []
        
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            
            # Inference with attr_texts (KEY MODIFICATION)
            with dt[1]:
                attr_texts = batch.get('attr_text', [])
                if not isinstance(attr_texts, list):
                    attr_texts = [attr_texts] if isinstance(attr_texts, str) else []
                preds = model(batch["img"], augment=augment, attr_texts=attr_texts)
            
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
            
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            
            self.run_callbacks("on_val_batch_end")
        
        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")
        
        if self.training:
            model.float()
            # Reduce loss across all GPUs
            import torch.distributed as dist
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            if RANK > 0:
                return
            return stats

