"""Elastic Weight Consolidation state validation and loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

EWC_STATE_VERSION = 2


def validate_ewc_state(state: dict[str, Any]) -> None:
    """Validate per-task diagonal Fishers and parameter snapshots."""
    if state.get("version") != EWC_STATE_VERSION:
        raise ValueError(
            f"Unsupported EWC artifact version: expected {EWC_STATE_VERSION}, got {state.get('version')}"
        )
    importance = state.get("importance")
    task_params = state.get("task_params")
    if not isinstance(importance, dict) or not isinstance(task_params, dict):
        raise TypeError("EWC artifact must contain 'importance' and 'task_params' dictionaries")
    if not importance:
        raise ValueError("EWC artifact contains no tracked parameters")
    if importance.keys() != task_params.keys():
        raise KeyError(
            "EWC importance and task parameter keys differ: "
            f"importance_only={sorted(importance.keys() - task_params.keys())}, "
            f"task_params_only={sorted(task_params.keys() - importance.keys())}"
        )

    task_count = None
    for name in importance:
        fisher_history = importance[name]
        param_history = task_params[name]
        if not isinstance(fisher_history, list) or not isinstance(param_history, list):
            raise TypeError(f"EWC history for '{name}' must be stored as lists")
        if not fisher_history or len(fisher_history) != len(param_history):
            raise ValueError(
                f"Invalid EWC history length for '{name}': "
                f"importance={len(fisher_history)}, task_params={len(param_history)}"
            )
        if task_count is None:
            task_count = len(fisher_history)
        elif len(fisher_history) != task_count:
            raise ValueError(
                f"Inconsistent EWC task count for '{name}': expected {task_count}, got {len(fisher_history)}"
            )
        for task_idx, (fisher, task_param) in enumerate(zip(fisher_history, param_history)):
            if not isinstance(fisher, Tensor) or not isinstance(task_param, Tensor):
                raise TypeError(f"EWC entry '{name}' task {task_idx} is not a tensor pair")
            if fisher.shape != task_param.shape:
                raise ValueError(
                    f"EWC shape mismatch for '{name}' task {task_idx}: "
                    f"importance={tuple(fisher.shape)}, task_param={tuple(task_param.shape)}"
                )
            if not fisher.dtype.is_floating_point or not task_param.dtype.is_floating_point:
                raise TypeError(f"EWC tensors for '{name}' task {task_idx} must be floating point")
            if not torch.isfinite(fisher).all():
                raise ValueError(f"EWC importance for '{name}' task {task_idx} contains non-finite values")
            if (fisher < 0).any():
                raise ValueError(f"EWC importance for '{name}' task {task_idx} contains negative values")


def load_ewc_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate an EWC artifact."""
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"EWC artifact '{path}' must contain a dictionary, got {type(state)}")
    validate_ewc_state(state)
    return state


class EWCLoss:
    """Compute ``0.5 * sum_t sum_i F[t,i] * (theta[i] - theta_star[t,i])**2``."""

    def __init__(self, model: nn.Module, state: dict[str, Any]):
        """Bind validated EWC history to current model parameters."""
        validate_ewc_state(state)
        self.model = model
        model_params = dict(model.named_parameters())
        tracked_names = state["importance"].keys()
        missing = tracked_names - model_params.keys()
        if missing:
            raise KeyError(f"EWC parameters are missing from the current model: {sorted(missing)}")

        self.importance: dict[str, list[Tensor]] = {}
        self.task_params: dict[str, list[Tensor]] = {}
        for name in tracked_names:
            current = model_params[name]
            fisher_history = state["importance"][name]
            param_history = state["task_params"][name]
            for task_idx, fisher in enumerate(fisher_history):
                if fisher.shape != current.shape:
                    raise ValueError(
                        f"EWC/current shape mismatch for '{name}' task {task_idx}: "
                        f"artifact={tuple(fisher.shape)}, current={tuple(current.shape)}"
                    )
            self.importance[name] = [
                fisher.to(device=current.device, dtype=current.dtype).detach().requires_grad_(False)
                for fisher in fisher_history
            ]
            self.task_params[name] = [
                task_param.to(device=current.device, dtype=current.dtype).detach().requires_grad_(False)
                for task_param in param_history
            ]

    def get_loss(self) -> Tensor:
        """Return the scalar paper EWC penalty, including its explicit one-half factor."""
        model_params = dict(self.model.named_parameters())
        first_param = model_params[next(iter(self.importance))]
        loss = first_param.new_zeros(())
        for name, fisher_history in self.importance.items():
            current = model_params[name]
            for fisher, task_param in zip(fisher_history, self.task_params[name]):
                loss = loss + 0.5 * (fisher * (current - task_param).square()).sum()
        return loss
