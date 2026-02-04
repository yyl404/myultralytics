from typing import Dict, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

from ultralytics.nn import BaseModel
from ultralytics.utils import LOGGER


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


class EWPRegLoss:
    """ESPReg (Eigen-value Scaled Projection Regularization) loss.
    Regularization loss based on eigen-value weighted projection length of parameter updates
    in the principal component subspace. Also referred to as EWPReg in implementation.
    """
    def __init__(self, model_update: BaseModel, model_base: BaseModel, module_names: List,
                 components: Dict[str, Tensor], eigen_values: Dict[str, Tensor]):
        """
        Args:
            model_update: The model being updated during training
            model_base: The base model for comparison
            module_names: List of module names to apply EWPReg loss
            components: Dictionary of PCA components for each module
            eigen_values: Dictionary of eigen values for each module
        """
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = module_names
        self.components = components
        self.eigen_values = eigen_values
        self.eigen_values_adjusted = adjust_eigen_values(eigen_values)

        for key in components.keys():
            # Freeze PCA attributes
            self.components[key].requires_grad_(False)
            self.eigen_values[key].requires_grad_(False)
            self.eigen_values_adjusted[key].requires_grad_(False)

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
            if n not in self.update_modules:
                LOGGER.warning(f"Module '{n}' not found in update_modules, skipping hook registration")
                continue
            if n not in self.base_modules:
                LOGGER.warning(f"Module '{n}' not found in base_modules, skipping hook registration")
                continue
            
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            # This hook is responsible for extracting given module's weight
            if isinstance(module, nn.Conv2d):
                dict_w[n] = module.weight.reshape(module.groups, module.weight.shape[0]//module.groups, -1) # [g, c_out//g, c_in//g*k*k]
                if module.bias is not None:
                    dict_w[n] = torch.concat((dict_w[n], module.bias.data.reshape(module.groups, module.out_channels//module.groups, 1)), dim=2)
            else:
                LOGGER.warning(f"Module {n}'s type {type(module)} is not supported, skipped")
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self) -> Tensor:
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [g, c_in//g*k*k, c_in//g*k*k]
            scale = self.eigen_values_adjusted[n] # [g, c_in//g*k*k]
            
            update_w = self.update_weights[n] # [g, c_out//g, c_in//g*k*k]
            base_w = self.base_weights[n] # [g, c_out//g, c_in//g*k*k]
            delta_w = update_w - base_w # [g, c_out//g, c_in//g*k*k]
            
            proj = proj.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)
            
            # ([g, c_out//g, c_in//g*k*k] @ [g, c_in//g*k*k, c_in//g*k*k]) * [g, 1, c_in//g*k*k]
            #       = [g, c_out//g, c_in//g*k*k] * [g, 1, c_in//g*k*k]
            #       = [g, c_out//g, c_in//g*k*k]
            #       -> norm([g, c_out//g, c_in//g*k*k], dim=2) = [g, c_out//g]
            #       -> mean([g, c_out//g]) = scalar
            loss += 100 * ((delta_w @ proj.transpose(1, 2)) * scale.unsqueeze(1)).norm(dim=2).mean()
        loss = loss / len(self.module_names)
        return loss
