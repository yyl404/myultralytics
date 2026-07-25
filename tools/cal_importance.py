"""Estimate and persist per-task diagonal Fisher information for EWC.

The estimator follows the object-detection procedure used by NSGP-RePRE: keep the
trained model fixed in evaluation mode, run the complete task detection loss over
the training loader, and average squared gradients. No clipping, tensor-wise
normalization, AMP, or optimizer update is applied.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ultralytics import YOLO
from ultralytics.engine.ewc import EWC_STATE_VERSION, load_ewc_state, validate_ewc_state
from ultralytics.models.yolo.detect import AntiForgetDetectionTrainer, DetectionTrainer
from ultralytics.models.yolo.obb import AntiForgetOBBTrainer, OBBTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.callbacks import default_callbacks
from ultralytics.utils.torch_utils import unwrap_model


def _normalization_parameter_names(model: nn.Module) -> set[str]:
    """Return affine parameter names belonging to parameterized normalization modules."""
    normalization_types = (
        nn.modules.batchnorm._NormBase,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
    )
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, normalization_types):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter.requires_grad:
                    names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names


def _select_parameter_names(
    model: nn.Module,
    scope: str,
    module_patterns: list[str] | None = None,
) -> set[str]:
    """Select trainable model parameters for full-model or normalization-only EWC."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if scope == "all":
        selected = trainable
    elif scope == "normalization":
        selected = trainable & _normalization_parameter_names(model)
    else:
        raise ValueError(f"Unsupported EWC parameter scope: {scope}")

    if module_patterns:
        selected = {
            name
            for name in selected
            if any(fnmatch.fnmatch(name.rsplit(".", 1)[0], pattern) for pattern in module_patterns)
        }
    if not selected:
        raise ValueError(
            f"No trainable parameters matched EWC scope='{scope}', module_patterns={module_patterns}"
        )
    return selected


class ImportanceCalculator:
    """Accumulate a sample-weighted average of squared task-loss gradients."""

    def __init__(
        self,
        model: nn.Module,
        scope: str = "all",
        module_patterns: list[str] | None = None,
        history: dict[str, Any] | None = None,
    ):
        """Bind selected parameters and optional prior task history."""
        self.model = unwrap_model(model)
        selected_names = _select_parameter_names(
            self.model, scope=scope, module_patterns=module_patterns
        )
        model_params = dict(self.model.named_parameters())
        self.parameters = {name: model_params[name] for name in sorted(selected_names)}
        self.importance_sum = {
            name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for name, parameter in self.parameters.items()
        }
        self.sample_count = 0
        self.scope = scope
        self.history = history

        if history is not None:
            validate_ewc_state(history)
            history_names = set(history["importance"])
            if history_names != selected_names:
                raise KeyError(
                    "Historical EWC parameters do not match the selected current parameters: "
                    f"history_only={sorted(history_names - selected_names)}, "
                    f"current_only={sorted(selected_names - history_names)}"
                )
            for name, parameter in self.parameters.items():
                for task_idx, fisher in enumerate(history["importance"][name]):
                    if fisher.shape != parameter.shape:
                        raise ValueError(
                            f"Historical EWC shape mismatch for '{name}' task {task_idx}: "
                            f"history={tuple(fisher.shape)}, current={tuple(parameter.shape)}"
                        )

    def process_gradients(self, batch_samples: int) -> None:
        """Add one batch's squared gradients with its sample-count weight."""
        if batch_samples <= 0:
            raise ValueError(f"batch_samples must be positive, got {batch_samples}")
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite Fisher gradient for parameter '{name}'")
            self.importance_sum[name].add_(gradient.square(), alpha=batch_samples)
        self.sample_count += batch_samples

    def build_state(self) -> dict[str, Any]:
        """Append the current Fisher and parameter optimum to prior task history."""
        if self.sample_count == 0:
            raise RuntimeError("No gradients were collected while estimating EWC importance")

        if self.history is None:
            importance = {name: [] for name in self.parameters}
            task_params = {name: [] for name in self.parameters}
            sample_counts = []
        else:
            importance = {
                name: [tensor.detach().cpu().clone() for tensor in history]
                for name, history in self.history["importance"].items()
            }
            task_params = {
                name: [tensor.detach().cpu().clone() for tensor in history]
                for name, history in self.history["task_params"].items()
            }
            sample_counts = list(self.history.get("sample_counts", []))

        for name, parameter in self.parameters.items():
            fisher = (self.importance_sum[name] / self.sample_count).detach().cpu()
            importance[name].append(fisher)
            task_params[name].append(parameter.detach().cpu().clone())
        sample_counts.append(self.sample_count)
        state = {
            "version": EWC_STATE_VERSION,
            "importance": importance,
            "task_params": task_params,
            "sample_counts": sample_counts,
            "scope": self.scope,
        }
        validate_ewc_state(state)
        return state


def _disable_validation(trainer) -> None:
    """Disable validation and final evaluation for the Fisher-only pass."""
    trainer.args.val = False
    trainer.validate = lambda: ({}, 0.0)
    trainer.final_eval = lambda: None


def calculate_importance(
    model: YOLO,
    dataset: str,
    save_path: str,
    workers: int = 8,
    device: str = "cuda",
    batch_size: int = 8,
    scope: str = "all",
    module_patterns: list[str] | None = None,
    load_hist: str | None = None,
    reference_model: str | None = None,
    conf_threshold: float = 0.25,
    filter_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Estimate one task Fisher at fixed parameters and save the complete EWC history."""
    history = load_ewc_state(load_hist, map_location="cpu") if load_hist else None
    calculator = None
    optimizer_step_count = 0

    train_kwargs = {
        "data": dataset,
        "epochs": 1,
        "device": device,
        "workers": workers,
        "batch": batch_size,
        "nbs": batch_size,
        "warmup_epochs": 0.0,
        "val": False,
        "plots": False,
        "save": False,
        "amp": False,
        "model": model.ckpt_path,
        "ewc": False,
        "pseudo_label": reference_model is not None,
        "reference_model": reference_model,
        "conf_threshold": conf_threshold,
        "filter_iou_threshold": filter_iou_threshold,
    }

    def on_train_start(trainer) -> None:
        """Initialize collection after the trainer has built and placed its model."""
        nonlocal calculator
        trainer.model.eval()
        trainer._model_train = trainer.model.eval
        calculator = ImportanceCalculator(
            trainer.model,
            scope=scope,
            module_patterns=module_patterns,
            history=history,
        )

        def collect_without_step() -> None:
            nonlocal optimizer_step_count
            if calculator is None:
                raise RuntimeError("EWC importance calculator was not initialized")
            dataset_size = len(trainer.train_loader.dataset)
            consumed = optimizer_step_count * batch_size
            current_batch_size = min(batch_size, dataset_size - consumed)
            if current_batch_size <= 0:
                raise RuntimeError(
                    f"Invalid Fisher batch accounting: dataset_size={dataset_size}, consumed={consumed}"
                )
            calculator.process_gradients(batch_samples=current_batch_size)
            optimizer_step_count += 1
            trainer.optimizer.zero_grad()

        trainer.optimizer_step = collect_without_step

    callbacks = default_callbacks.copy()
    callbacks["on_train_start"] = callbacks["on_train_start"] + [
        on_train_start,
        _disable_validation,
    ]

    task = getattr(model, "task", "detect")
    if task == "obb":
        trainer_class = AntiForgetOBBTrainer if reference_model else OBBTrainer
    elif task == "detect":
        trainer_class = AntiForgetDetectionTrainer if reference_model else DetectionTrainer
    else:
        raise TypeError(f"EWC importance supports detect and obb tasks, got '{task}'")

    trainer = trainer_class(overrides=train_kwargs, _callbacks=callbacks)
    trainer.train()
    if calculator is None:
        raise RuntimeError("EWC importance calculation did not start")

    state = calculator.build_state()
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    LOGGER.info(
        f"Saved EWC task {len(next(iter(state['importance'].values())))} state "
        f"for {len(state['importance'])} parameters to {output}"
    )
    return state


def main() -> None:
    """Parse command-line arguments and estimate EWC importance."""
    parser = argparse.ArgumentParser(description="Calculate diagonal Fisher information for EWC")
    parser.add_argument("--model", required=True, help="Trained model checkpoint")
    parser.add_argument("--dataset", required=True, help="Current task dataset YAML")
    parser.add_argument("--save_path", required=True, help="Output EWC artifact")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scope", choices=("all", "normalization"), default="all")
    parser.add_argument("--module_pattern", nargs="+", default=None)
    parser.add_argument("--load_hist", default=None, help="Expanded EWC artifact from prior tasks")
    parser.add_argument("--reference_model", default=None, help="Frozen teacher used for pseudo labels")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--filter_iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    calculate_importance(
        YOLO(args.model),
        dataset=args.dataset,
        save_path=args.save_path,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        scope=args.scope,
        module_patterns=args.module_pattern,
        load_hist=args.load_hist,
        reference_model=args.reference_model,
        conf_threshold=args.conf_threshold,
        filter_iou_threshold=args.filter_iou_threshold,
    )


if __name__ == "__main__":
    main()
