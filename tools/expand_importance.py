"""Expand every historical EWC Fisher and parameter snapshot for a larger detection head."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ultralytics import YOLO
from ultralytics.engine.ewc import load_ewc_state, validate_ewc_state
from ultralytics.utils import LOGGER


def _class_channel_map(old_model: YOLO, new_model: YOLO) -> dict[int, int]:
    """Map old classifier channel indices to expanded channels by class name."""
    old_names = [old_model.names[index] for index in sorted(old_model.names)]
    new_names = [new_model.names[index] for index in sorted(new_model.names)]
    if len(set(new_names)) != len(new_names):
        raise ValueError("Expanded model contains duplicate class names")
    missing = sorted(set(old_names) - set(new_names))
    if missing:
        raise ValueError(f"Expanded model is missing old classes: {missing}")
    return {old_idx: new_names.index(name) for old_idx, name in enumerate(old_names)}


def _is_classifier_output_parameter(name: str, old: Tensor, new: Tensor) -> bool:
    """Return whether a changed tensor is a YOLO classification output weight or bias."""
    return (
        "cv3" in name
        and (name.endswith(".2.weight") or name.endswith(".2.bias"))
        and old.ndim == new.ndim
        and old.shape[1:] == new.shape[1:]
        and old.shape[0] < new.shape[0]
    )


def _expand_history_tensor(
    old_tensor: Tensor,
    new_template: Tensor,
    channel_map: dict[int, int],
    fill_from_template: bool,
) -> Tensor:
    """Expand one historical tensor, copying old class channels and initializing new channels."""
    expanded = new_template.detach().cpu().clone() if fill_from_template else torch.zeros_like(new_template, device="cpu")
    old_cpu = old_tensor.detach().cpu()
    for old_idx, new_idx in channel_map.items():
        if old_idx >= old_cpu.shape[0] or new_idx >= expanded.shape[0]:
            raise IndexError(
                f"Class channel mapping ({old_idx} -> {new_idx}) exceeds "
                f"old/new shapes {tuple(old_cpu.shape)} -> {tuple(expanded.shape)}"
            )
        expanded[new_idx] = old_cpu[old_idx]
    return expanded


def expand_ewc_state(
    state: dict[str, Any],
    old_model: YOLO,
    new_model: YOLO,
) -> dict[str, Any]:
    """Return EWC history aligned to an expanded model.

    New classifier channels receive zero Fisher. Their historical parameter snapshots use
    the expanded model initialization, making the zero-Fisher delta well-defined without
    imposing any old-task constraint.
    """
    validate_ewc_state(state)
    old_parameters = dict(old_model.model.named_parameters())
    new_parameters = dict(new_model.model.named_parameters())
    channel_map = _class_channel_map(old_model, new_model)

    expanded_importance: dict[str, list[Tensor]] = {}
    expanded_task_params: dict[str, list[Tensor]] = {}
    for name in state["importance"]:
        if name not in old_parameters or name not in new_parameters:
            raise KeyError(f"Tracked EWC parameter '{name}' is missing from the old or expanded model")
        old_parameter = old_parameters[name]
        new_parameter = new_parameters[name]
        fisher_history = state["importance"][name]
        param_history = state["task_params"][name]

        if old_parameter.shape == new_parameter.shape:
            expanded_importance[name] = [tensor.detach().cpu().clone() for tensor in fisher_history]
            expanded_task_params[name] = [tensor.detach().cpu().clone() for tensor in param_history]
            continue
        if not _is_classifier_output_parameter(name, old_parameter, new_parameter):
            raise ValueError(
                f"Unsupported EWC parameter expansion for '{name}': "
                f"{tuple(old_parameter.shape)} -> {tuple(new_parameter.shape)}"
            )

        expanded_importance[name] = [
            _expand_history_tensor(
                old_tensor=fisher,
                new_template=new_parameter,
                channel_map=channel_map,
                fill_from_template=False,
            )
            for fisher in fisher_history
        ]
        expanded_task_params[name] = [
            _expand_history_tensor(
                old_tensor=task_param,
                new_template=new_parameter,
                channel_map=channel_map,
                fill_from_template=True,
            )
            for task_param in param_history
        ]

    expanded_state = {
        **state,
        "importance": expanded_importance,
        "task_params": expanded_task_params,
    }
    validate_ewc_state(expanded_state)
    return expanded_state


def expand_importance(
    old_importance_path: str,
    old_model_path: str,
    new_model_path: str,
    save_path: str,
) -> dict[str, Any]:
    """Load, expand, validate, and save EWC history."""
    state = load_ewc_state(old_importance_path, map_location="cpu")
    expanded = expand_ewc_state(state, YOLO(old_model_path), YOLO(new_model_path))
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, output)
    LOGGER.info(f"Saved expanded EWC state for {len(expanded['importance'])} parameters to {output}")
    return expanded


def main() -> None:
    """Parse command-line arguments and expand an EWC artifact."""
    parser = argparse.ArgumentParser(description="Expand EWC history for a larger YOLO detection head")
    parser.add_argument("--old_importance", required=True)
    parser.add_argument("--old_model", required=True)
    parser.add_argument("--new_model", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    expand_importance(
        old_importance_path=args.old_importance,
        old_model_path=args.old_model,
        new_model_path=args.new_model,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
