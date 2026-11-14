from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from ultralytics.engine.spectrum import find_elbow_point
from ultralytics.nn import BaseModel


def adjust_eigen_values(eigen_values: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Adjust eigen values by finding elbow point and normalizing.
    
    The adjustment process:
    1. Find the elbow point in the eigen value spectrum
    2. Calculate a scale factor to normalize the elbow point value to 1.0
    3. Set all values at and before the elbow point to 1.0
    4. Scale all values after the elbow point by the same factor
    
    Args:
        eigen_values: Dictionary of tensors, each with shape [g, c_in//g*k*k]
    
    Returns:
        Dictionary of adjusted eigen values with same shapes as input
    """
    adjusted = {}
    
    for name, eigen_vals in eigen_values.items():
        g, n = eigen_vals.shape
        adjusted_eigen_vals = torch.zeros_like(eigen_vals)
        
        for i in range(g):
            group_eigen_vals = eigen_vals[i]  # shape: [c_in//g*k*k]
            
            elbow_idx = find_elbow_point(group_eigen_vals)
            elbow_idx = max(0, min(elbow_idx, n - 1))  # Ensure valid index
            
            # Calculate scale factor to normalize elbow point value to 1.0
            elbow_value = group_eigen_vals[elbow_idx]
            eps = 1e-8
            scale_factor = 1.0 / (elbow_value + eps)
            
            # Initialize all values to 1.0
            adjusted_group = torch.ones_like(group_eigen_vals)
            
            # For values after elbow point, apply scale factor
            if elbow_idx + 1 < n:
                adjusted_group[elbow_idx + 1:] = group_eigen_vals[elbow_idx + 1:] * scale_factor
            
            adjusted_eigen_vals[i] = adjusted_group
        
        adjusted[name] = adjusted_eigen_vals
    
    return adjusted


class ESPRegLoss:
    """ESPReg (Eigen-value Scaled Projection Regularization) loss.

    Penalizes the projection of the accumulated parameter update ``Delta W`` onto the
    historical principal directions, weighted by the historical spectrum of the
    non-centered input second moment ``Sigma = E[x x^T]``. Two reductions share the
    interface:

    - ``form="original"``: mean over output channels of the weighted projection length,
      ``mean_o sqrt(sum_k w_k^2 (Delta W_o . u_k)^2)`` with the truncated/normalized
      spectrum ``w = lambda*``. This is the implementation used by the earlier runs.
    - ``form="frobenius"``: the squared weighted projection energy,
      ``sum_o sum_k w_k (Delta W_o . u_k)^2 = Tr(Delta W U diag(w) U^T Delta W^T)``,
      which equals the local response disturbance when ``w = lambda`` and ``alpha_l = 1``.

    With ``weight_mode="raw"`` and ``form="frobenius"`` the penalty is exactly the
    expected squared pre-activation drift on historical inputs, so the eigen-values
    must be those of the non-centered second moment (the PCA cache is built with
    ``tools/pca.py --uncentered``).

    Args:
        model_update: The model being updated during training
        model_base: The base model for comparison
        module_names: List of module names to apply ESPReg loss
        components: Dictionary of PCA components for each module
        eigen_values: Dictionary of second-moment eigen values for each module
        weight_mode: ``"raw"`` uses the eigenvalues as they are, ``"nt"`` uses the
            elbow-normalized and truncated spectrum
        form: ``"original"`` or ``"frobenius"`` reduction, see the class docstring
    """

    WEIGHT_MODES = ("raw", "nt")
    FORMS = ("original", "frobenius")

    def __init__(self, model_update: BaseModel, model_base: BaseModel, module_names: List,
                 components: Dict[str, Tensor], eigen_values: Dict[str, Tensor],
                 param_names: Dict[str, str] | None = None,
                 weight_mode: str = "nt", form: str = "original"):
        """
        Args:
            model_update: The model being updated during training
            model_base: The base model for comparison
            module_names: List of module names to apply ESPReg loss
            components: Dictionary of PCA components for each module
            eigen_values: Dictionary of eigen values for each module
            param_names: Module name -> governed parameter attribute (default ``weight``)
            weight_mode: ``"raw"`` or ``"nt"``
            form: ``"original"`` or ``"frobenius"``
        """
        if weight_mode not in self.WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {self.WEIGHT_MODES}, got {weight_mode!r}")
        if form not in self.FORMS:
            raise ValueError(f"form must be one of {self.FORMS}, got {form!r}")
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = list(module_names)
        self.param_names = dict(param_names or {})
        self.components = components
        self.eigen_values = eigen_values
        self.weight_mode = weight_mode
        self.form = form
        self.eigen_values_adjusted = adjust_eigen_values(eigen_values) if weight_mode == "nt" else None

        for key in components.keys():
            # Freeze PCA attributes
            self.components[key].requires_grad_(False)
            self.eigen_values[key].requires_grad_(False)
            if self.eigen_values_adjusted is not None:
                self.eigen_values_adjusted[key].requires_grad_(False)

        self.update_modules, self.base_modules = {}, {}
        for n, m in model_update.named_modules():
            if n in self.module_names:
                self.update_modules[n] = m
        for n, m in model_base.named_modules():
            if n in self.module_names:
                self.base_modules[n] = m
        missing = [n for n in self.module_names if n not in self.update_modules or n not in self.base_modules]
        if missing:
            raise KeyError(f"ESPReg modules are missing from the student/base model: {missing}")
        
        self.update_weights, self.base_weights = {}, {}
        self._handles = []
        self._device = None

    def register_hook(self):
        """Register the weight-capture hooks used by one training epoch."""
        self.remove_handle_()
        for n in self.module_names:
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            # This hook is responsible for extracting given module's weight
            if isinstance(module, nn.Conv2d):
                groups = module.groups
            elif isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                groups = 1
            else:
                raise TypeError(
                    f"ESPReg only supports Conv2d/Linear/MultiheadAttention modules, got {type(module)} for '{n}'"
                )
            param = getattr(module, self.param_names.get(n, "weight"))
            out_channels = param.shape[0]
            flat_weight = param.reshape(groups, out_channels // groups, -1)
            # A bias column is appended only when the cached basis was fitted with one
            # (centered PCA on biased convolutions). Uncentered second-moment caches have
            # no bias dimension.
            bias = getattr(module, "bias", None)
            if bias is not None and self.components[n].shape[-1] == flat_weight.shape[-1] + 1:
                flat_weight = torch.concat(
                    (flat_weight, bias.data.reshape(groups, out_channels // groups, 1)), dim=2
                )
            dict_w[n] = flat_weight
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _ensure_device(self, device: torch.device) -> None:
        """Move the cached spectra to the training device once per process."""
        if self._device == device:
            return
        for table in (self.components, self.eigen_values, self.eigen_values_adjusted):
            if table is None:
                continue
            for name, value in table.items():
                table[name] = value.to(device)
        self._device = device

    def get_loss(self) -> Tensor:
        """Return the ESPReg penalty of the current accumulated parameter update."""
        missing = [
            name
            for name in self.module_names
            if name not in self.update_weights or name not in self.base_weights
        ]
        if missing:
            raise RuntimeError(
                f"ESPReg weights were not captured for {len(missing)} module(s), e.g. '{missing[0]}'; "
                "run a forward pass through both models after register_hook()"
            )
        self._ensure_device(next(iter(self.update_weights.values())).device)
        source = self.eigen_values_adjusted if self.weight_mode == "nt" else self.eigen_values
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [g, D, D], rows are the principal directions u_k
            scale = source[n] # [g, D] weights w_k

            update_w = self.update_weights[n] # [g, c_out//g, D]
            base_w = self.base_weights[n] # [g, c_out//g, D]
            if update_w.shape != base_w.shape:
                raise RuntimeError(
                    f"ESPReg weight mismatch for '{n}': update {tuple(update_w.shape)}, "
                    f"base {tuple(base_w.shape)}"
                )
            delta_w = update_w - base_w # [g, c_out//g, D]
            if delta_w.shape[-1] != proj.shape[-1]:
                raise RuntimeError(
                    f"ESPReg component dimension mismatch for '{n}': update {delta_w.shape[-1]}, "
                    f"components {proj.shape[-1]}. Generate the PCA cache with tools/pca.py --uncentered."
                )

            # Projections onto every principal direction: [g, c_out//g, D] with entry (o, k) = Delta W_o . u_k
            projected = delta_w @ proj.transpose(1, 2)
            if self.form == "frobenius":
                # sum_k w_k (Delta W_o . u_k)^2, summed over output channels and groups
                loss += 100 * (projected * scale.sqrt().unsqueeze(1)).pow(2).sum()
            else:
                # mean over output channels/groups of sqrt(sum_k w_k^2 (Delta W_o . u_k)^2)
                loss += 100 * (projected * scale.unsqueeze(1)).norm(dim=2).mean()
        loss = loss / len(self.module_names)
        return loss
