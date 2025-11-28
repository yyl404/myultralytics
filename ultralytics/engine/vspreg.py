import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER


class VSPRegLoss:
    """ Variance-Scaled Projection Regularization loss(VSPReg)
    Regularization loss based on variance-scaled projection length of weight updates in the principal component subspace
    """
    def __init__(self, model_update, model_base, module_names, components, variances,
                 keep_ratio=0.9, center_ratio=0.9, steepness=100):
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = module_names
        self.components = components
        self.variances = variances
        self.keep_ratio = keep_ratio
        self.center_ratio = center_ratio
        self.steepness = steepness
        
        self.kappa = {}
        self.mu = {}
        
        # Initialize kappa and mu for arctan transformation: arctan(kappa * log(scale) - mu)
        for key in components.keys():
            component = components[key]  # [num_groups, num_components, c_in//g*k*k]
            scale = variances[key]  # [num_groups, num_components]

            # scale_logged = torch.log(scale + 1e-6)
            scale_cumsum = torch.cumsum(scale, dim=-1)
            scale_cumsum_normalized = scale_cumsum / scale_cumsum[:, -1:]
            
            num_groups = scale_cumsum_normalized.shape[0]
            num_components = scale_cumsum_normalized.shape[1]
            
            # Find max_r: largest index where cumulative scale exceeds keep_ratio
            max_r = None
            for group_idx in range(num_groups):
                for comp_idx in range(num_components):
                    if scale_cumsum_normalized[group_idx, comp_idx] > self.keep_ratio:
                        max_r = max(max_r, comp_idx) if max_r is not None else comp_idx
                        break
            
            if max_r is None:
                max_r = num_components

            # Compute kappa and mu for arctan transformation
            # self.kappa[key], self.mu[key] = self._compute_kappa_and_mu(scale_logged, scale_cumsum_normalized)
            
            # Retain components up to max_r
            components[key] = component[:, :max_r]
            variances[key] = scale[:, :max_r]

            # Freeze gradients
            # self.kappa[key].requires_grad_(False)
            # self.mu[key].requires_grad_(False)
            components[key].requires_grad_(False)
            variances[key].requires_grad_(False)

        self.update_modules, self.base_modules = {}, {}
        for n, m in model_update.named_modules():
            if n in module_names:
                self.update_modules[n] = m
        for n, m in model_base.named_modules():
            if n in module_names:
                self.base_modules[n] = m
        
        self.update_weights, self.base_weights = {}, {}
        self._handles = []

    def _compute_kappa_and_mu(self, scale_logged, scale_cumsum_normalized):
        """Compute kappa and mu values for arctan transformation.
        
        Args:
            scale_logged: [num_groups, num_components] - logged scale values
            scale_cumsum_normalized: [num_groups, num_components] - normalized cumulative sum
        
        Returns:
            kappa: [num_groups] - scaling factor for log(scale)
            mu: [num_groups] - center value for log(scale) adjustment
        """
        num_groups = scale_cumsum_normalized.shape[0]
        num_components = scale_cumsum_normalized.shape[1]
        
        kappa = torch.zeros(num_groups)
        mu = torch.zeros(num_groups)
        
        # Compute kappa and mu for each group
        for group_idx in range(num_groups):
            # Find center component index based on center_ratio (used as mu reference point)
            center_comp_idx = None
            for comp_idx in range(num_components):
                if scale_cumsum_normalized[group_idx, comp_idx] > self.center_ratio:
                    center_comp_idx = comp_idx
                    break
            
            if center_comp_idx is None:
                center_comp_idx = num_components - 1
            
            mu_value = scale_logged[group_idx, center_comp_idx].item()
            mu[group_idx] = mu_value

            # Saturation threshold: map arctan(-π/2, π/2) to (-1, 1), so 0.85 -> tan(0.85 * π/2)
            saturation_threshold = torch.tan(torch.tensor(0.85 * torch.pi / 2)).item()

            # Determine saturation region bounds: 1/steepness fraction of components
            saturation_region_half_size = int(num_components / (2 * self.steepness))
            saturation_upper_idx = min(center_comp_idx + saturation_region_half_size, num_components - 1)
            saturation_lower_idx = max(center_comp_idx - saturation_region_half_size, 0)
            
            # Compute kappa: solve kappa * log(scale) - mu = ±saturation_threshold
            scale_upper = scale_logged[group_idx, saturation_upper_idx].item()
            scale_lower = scale_logged[group_idx, saturation_lower_idx].item()
            
            kappa_upper = (saturation_threshold + mu_value) / (scale_upper + 1e-6)
            kappa_lower = (-saturation_threshold + mu_value) / (scale_lower + 1e-6)
            kappa_value = (kappa_upper + kappa_lower) / 2.0
            kappa[group_idx] = kappa_value
        
        return kappa, mu

    def register_hook(self):
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
                dict_w[n] = module.weight.reshape(module.groups, module.weight.shape[0]//module.groups, -1) # [g, c_out//g, c_in//g*k*k]
                if module.bias is not None:
                    dict_w[n] = torch.concat((dict_w[n], module.bias.data.reshape(module.groups, module.out_channels//module.groups, 1)), dim=2)
            else:
                LOGGER.warning(f"WARNING ⚠️ VSPRegLoss: Module {n}'s type {type(module)} is not supported, skipped")
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self):
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [g, r, c_in//g*k*k]
            scale = torch.sqrt(self.variances[n]) # [g, r]
            
            update_w = self.update_weights[n] # [g, c_out//g, c_in//g*k*k]
            base_w = self.base_weights[n] # [g, c_out//g, c_in//g*k*k]
            delta_w = F.normalize(update_w - base_w, p=2, dim=2) # [g, c_out//g, c_in//g*k*k]
            
            proj = proj.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)
            # kappa = self.kappa[n].to(delta_w.device, delta_w.dtype)
            # mu = self.mu[n].to(delta_w.device, delta_w.dtype)
            # scale_adjusted = torch.arctan(kappa.unsqueeze(-1) * torch.log(scale + 1e-6) - mu.unsqueeze(-1))
            
            # ([g, c_out//g, c_in//g*k*k] @ [g, c_in//g*k*k, r]) * [g, 1, r]
            #       = [g, c_out//g, r] * [g, 1, r]
            #       = [g, c_out//g, r]
            #       -> norm([g, c_out//g, r], dim=2) = [g, c_out//g]
            #       -> mean([g, c_out//g]) = scalar
            # loss += ((delta_w @ proj.transpose(1, 2)) * scale_adjusted.unsqueeze(1)).norm(dim=2).mean()
            loss += (delta_w @ proj.transpose(1, 2)).norm(dim=2).mean()
        loss = loss / len(self.module_names)
        return loss
    
    def set_parameters(self, components, variances, keep_ratio=1.0, center_ratio=0.9, steepness=100):
        """Update components, variances, then recompute kappa and mu.
        
        Args:
            components: Dictionary of component tensors
            variances: Dictionary of variance tensors
            keep_ratio: Ratio for retaining components based on cumulative scale
            center_ratio: Ratio for determining center component index (mu reference point)
            steepness: Parameter for determining saturation region size (1/steepness fraction)
        """
        self.components = components
        self.variances = variances
        self.keep_ratio = keep_ratio
        self.center_ratio = center_ratio
        self.steepness = steepness
        
        # Update kappa and mu for new components and variances
        for key in components.keys():
            component = components[key]  # [num_groups, num_components, c_in//g*k*k]
            scale = variances[key]  # [num_groups, num_components]

            scale_logged = torch.log(scale + 1e-6)
            scale_cumsum = torch.cumsum(scale, dim=-1)
            scale_cumsum_normalized = scale_cumsum / scale_cumsum[:, -1:]
            
            num_groups = scale_cumsum_normalized.shape[0]
            num_components = scale_cumsum_normalized.shape[1]
            
            # Find max_r: largest index where cumulative scale exceeds keep_ratio
            max_r = None
            for group_idx in range(num_groups):
                for comp_idx in range(num_components):
                    if scale_cumsum_normalized[group_idx, comp_idx] > self.keep_ratio:
                        max_r = max(max_r, comp_idx) if max_r is not None else comp_idx
                        break
            
            if max_r is None:
                max_r = num_components

            # Compute kappa and mu for arctan transformation
            self.kappa[key], self.mu[key] = self._compute_kappa_and_mu(scale_logged, scale_cumsum_normalized)
            
            # Retain components up to max_r (not including max_r)
            components[key] = component[:, :max_r]
            variances[key] = scale[:, :max_r]

            # Freeze gradients
            self.kappa[key].requires_grad_(False)
            self.mu[key].requires_grad_(False)
            components[key].requires_grad_(False)
            variances[key].requires_grad_(False)
