from typing import Dict, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

from ultralytics.nn import BaseModel


def find_elbow_point(svals: torch.Tensor) -> int:
    """Find the elbow point in singular value spectrum using second-order derivatives.
    
    Args:
        svals: Singular values in descending order, shape (N,)
    
    Returns:
        int: Index of the elbow point
    """
    points = svals.cpu().numpy()
    assert points.ndim == 1
    
    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        _delta = 1
        diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        _drop_ratio = 0.03
        drop_num = int(len(points) * _drop_ratio / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
    
    i_thres = np.arange(len(points))[points >= thres_val].max()
    return int(i_thres)


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
    historical principal directions, weighted by the historical spectrum. Two reductions
    share the interface:

    - ``form="original"``: mean over output channels of the weighted projection length,
      ``mean_o sqrt(sum_k w_k^2 (Delta W_o . u_k)^2)`` with the truncated/normalized
      spectrum ``w = lambda*``. This is the implementation used by the earlier runs.
    - ``form="frobenius"``: the squared weighted projection energy,
      ``sum_o sum_k w_k (Delta W_o . u_k)^2 = Tr(Delta W U diag(w) U^T Delta W^T)``,
      which equals the local response disturbance when ``w = lambda`` and ``alpha_l = 1``.

    Args:
        model_update: The model being updated during training
        model_base: The base model for comparison
        module_names: List of module names to apply ESPReg loss
        components: Dictionary of PCA components for each module
        eigen_values: Dictionary of eigen values for each module
        weight_mode: ``"raw"`` uses the eigenvalues as they are, ``"nt"`` uses the
            elbow-normalized and truncated spectrum
        form: ``"original"`` or ``"frobenius"`` reduction, see the class docstring
    """

    WEIGHT_MODES = ("raw", "nt")
    FORMS = ("original", "frobenius")

    def __init__(self, model_update: BaseModel, model_base: BaseModel, module_names: List,
                 components: Dict[str, Tensor], eigen_values: Dict[str, Tensor],
                 weight_mode: str = "nt", form: str = "original"):
        """
        Args:
            model_update: The model being updated during training
            model_base: The base model for comparison
            module_names: List of module names to apply ESPReg loss
            components: Dictionary of PCA components for each module
            eigen_values: Dictionary of eigen values for each module
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
            if not isinstance(module, nn.Conv2d):
                raise TypeError(f"ESPReg only supports Conv2d modules, got {type(module)} for '{n}'")
            dict_w[n] = module.weight.reshape(module.groups, module.weight.shape[0]//module.groups, -1) # [g, c_out//g, c_in//g*k*k]
            if module.bias is not None:
                dict_w[n] = torch.concat((dict_w[n], module.bias.data.reshape(module.groups, module.out_channels//module.groups, 1)), dim=2)
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self) -> Tensor:
        """Return the ESPReg penalty of the current accumulated parameter update."""
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

            proj = proj.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)

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
