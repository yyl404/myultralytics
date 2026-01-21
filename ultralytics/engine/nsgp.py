from typing import Dict, List

import torch
from torch import Tensor

from ultralytics.engine.ewpr import find_elbow_point
from ultralytics.utils import LOGGER, colorstr


class NSGP:
    """Null Space Gradient Projection (NSGP)
    
    Projects gradients onto the principal component directions before the elbow point,
    then subtracts the projected component from the original gradient with a flexibility weight.
    This effectively removes gradient components in important principal directions,
    constraining updates to the less important (null space) directions.
    """
    
    def __init__(self, module_names: List[str], components: Dict[str, Tensor], eigen_values: Dict[str, Tensor]):
        """
        Args:
            module_names: List of module names to apply NSGP
            components: Dictionary of PCA components for each module, shape [g, c_in//g*k*k, c_in//g*k*k]
            eigen_values: Dictionary of eigen values for each module, shape [g, c_in//g*k*k]
            flexibility: Weight for the projected component when subtracting from original gradient
        """
        self.module_names = module_names
        self.components = components
        self.eigen_values = eigen_values
        
        # Freeze PCA attributes
        for key in components.keys():
            self.components[key].requires_grad_(False)
            self.eigen_values[key].requires_grad_(False)
        
        # Compute elbow points and create projection masks
        self.elbow_indices = {}
        self.projection_masks = {}
        self._compute_projection_masks()
    
    def _compute_projection_masks(self):
        """Compute elbow points and create projection masks for each module.
        
        Now computes pre-elbow components (before and including elbow point) for projection.
        """
        for name in self.module_names:
            if name not in self.components:
                continue
            
            eigen_vals = self.eigen_values[name]  # [g, c_in//g*k*k]
            components = self.components[name]  # [g, c_in//g*k*k, c_in//g*k*k]
            g, n = eigen_vals.shape
            
            elbow_indices = []
            projection_components = []
            
            for i in range(g):
                group_eigen_vals = eigen_vals[i]  # [c_in//g*k*k]
                elbow_idx = find_elbow_point(group_eigen_vals)
                elbow_idx = max(0, min(elbow_idx, n - 1))  # Ensure valid index
                elbow_indices.append(elbow_idx)
                
                # Get components before and including elbow point (0 to elbow_idx)
                # Each column in components[i] is a principal component
                # We want to project onto components from index 0 to elbow_idx (inclusive)
                if elbow_idx >= 0:
                    # Extract principal components before and including elbow point
                    # components[i]: [c_in//g*k*k, c_in//g*k*k], each column is a PC
                    # We want columns from 0 to elbow_idx (inclusive)
                    pre_elbow_components = components[i, :, :elbow_idx + 1]  # [c_in//g*k*k, elbow_idx + 1]
                    projection_components.append(pre_elbow_components)
                else:
                    # If no valid elbow point, no components to project onto
                    # Create zero projection (will keep original gradient)
                    projection_components.append(torch.zeros(
                        (n, 0), 
                        device=components.device, 
                        dtype=components.dtype
                    ))
            
            self.elbow_indices[name] = elbow_indices
            
            # Debug: output elbow indices for each group
            # LOGGER.info(f"{colorstr('NSGP elbow indices for module')} '{name}': {elbow_indices} (total groups: {g}, dimension: {n})")
            
            # Store pre-elbow components for projection
            # Note: num_pre_elbow_components may vary per group, so we store as a list
            self.projection_masks[name] = projection_components
    
    def apply_projection(self, params_dict: Dict[str, torch.nn.Parameter], flexibility: float = 1.0):
        """Manually apply gradient projection (alternative to hooks).
        
        This method can be called before optimizer step to project gradients.
        
        Args:
            params_dict: Dictionary mapping parameter names (e.g., 'module_name.weight') to 
                        parameter objects. These should be the exact parameter objects from 
                        optimizer.param_groups to ensure we modify the gradients that optimizer 
                        actually uses.
            flexibility: Weight for the projected component when subtracting from original gradient
        """
        for name in self.module_names:
            # Get the parameter name for weight
            param_name = name + '.weight'
            
            # Get parameter from params_dict
            if param_name not in params_dict:
                continue
            
            weight_param = params_dict[param_name]
            
            if weight_param is None or weight_param.grad is None:
                continue
            
            if name not in self.projection_masks:
                continue
            
            # Get the gradient from the parameter
            grad = weight_param.grad  # [c_out, c_in, k, k]
            
            # Get the number of groups from projection_masks length
            # projection_masks[name] is a list with one element per group
            projection_components = self.projection_masks[name]
            g = len(projection_components)  # Number of groups
            
            # Reshape gradient to match weight format: [g, c_out//g, c_in//g*k*k]
            # Note: Only weight gradients are projected, bias is not considered (consistent with EWPR)
            c_out, c_in, k_h, k_w = grad.shape
            grad_reshaped = grad.reshape(g, c_out // g, c_in // g, k_h * k_w)
            grad_reshaped = grad_reshaped.reshape(g, c_out // g, -1)  # [g, c_out//g, c_in//g*k*k]
            
            # Project gradients for each group
            # New rule: project onto pre-elbow components, then subtract from original gradient
            projected_weight_grad = grad_reshaped.clone()  # Start with original gradient
            
            for i in range(g):
                pre_elbow_comp = projection_components[i]  # [c_in//g*k*k, num_pre_elbow]
                # Move to same device as gradient
                pre_elbow_comp = pre_elbow_comp.to(grad.device, grad.dtype)
                # Project gradient onto pre-elbow principal components
                grad_group = grad_reshaped[i]  # [c_out//g, c_in//g*k*k]
                # Project: grad_group @ pre_elbow_comp @ pre_elbow_comp.T
                grad_proj = grad_group @ pre_elbow_comp @ pre_elbow_comp.T
                # Subtract the projected component from original gradient (with flexibility weight)
                projected_weight_grad[i] = grad_group - flexibility * grad_proj
            
            # Reshape weight gradient back to original shape
            projected_weight_grad = projected_weight_grad.reshape(
                g, c_out // g, c_in // g, k_h * k_w
            ).reshape(c_out, c_in, k_h, k_w)
            
            # Update weight gradient in-place
            weight_param.grad.data.copy_(projected_weight_grad)

