import torch
import torch.nn as nn
import torch.nn.functional as F


class VSPRegLoss:
    """ Variance-Scaled Projection Regularization loss(VSP-Reg)
    Regularization loss based on variance-scaled projection length of weight updates in the principal component subspace
    """
    def __init__(self, model_update, model_base, module_names, components, variances, means, alpha=1.0, beta=0.0):
        self.model_update = model_update
        self.model_base = model_base
        self.module_names = module_names
        self.components = components
        self.variances = variances
        self.means = means
        self.alpha = alpha
        self.beta = beta
        
        # Freeze components, variances, and means
        for key in components.keys():
            _component = components[key]
            _scale = variances[key]
            _bias = means[key]

            _scale_cumsum = torch.cumsum(_scale, dim=-1)
            _scale_cumsum_normalized = _scale_cumsum / _scale_cumsum[:, -1]
            max_r = 0
            for i in range(len(_scale_cumsum_normalized)):
                for j in range(len(_scale_cumsum_normalized[i])):
                    if _scale_cumsum_normalized[i][j] > 0.9:
                        max_r = max(max_r, j)
                        break
            
            components[key] = _component[:, :max_r]
            variances[key] = _scale[:, :max_r]

            components[key].requires_grad_(False)
            variances[key].requires_grad_(False)
            means[key].requires_grad_(False)

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
            u_mod = self.update_modules[n]
            b_mod = self.base_modules[n]
            self._handles.append(u_mod.register_forward_hook(self._hook(self.update_weights, n)))
            self._handles.append(b_mod.register_forward_hook(self._hook(self.base_weights, n)))
 
    def _hook(self, dict_w, n):
        def fn(module, _, __):
            if isinstance(module, nn.Conv2d):
                dict_w[n] = module.weight.reshape(module.groups, module.weight.shape[0]//module.groups, -1) # [g, c_out//g, c_in//g*k*k]
            else:
                raise RuntimeError(f"Module {n}'s type {type(module)} is not supported")
        return fn

    def remove_handle_(self):
        """ When training is complete/no longer needed, remove all hooks, release memory, and prevent memory leaks. """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self):
        loss = 0
        for n in self.module_names:
            proj = self.components[n] # [g, n_components, c_in//g*k*k]
            scale = torch.sqrt(self.variances[n]) # [g, n_components]
            bias = self.means[n] # [g, c_in//g*k*k]
            
            update_w = self.update_weights[n] # [g, c_out//g, c_in//g*k*k]
            base_w = self.base_weights[n] # [g, c_out//g, c_in//g*k*k]
            delta_w = F.normalize(update_w - base_w, p=2, dim=2) # [g, c_out//g, c_in//g*k*k]
            
            proj = proj.to(delta_w.device, delta_w.dtype)
            bias = bias.to(delta_w.device, delta_w.dtype)
            scale = scale.to(delta_w.device, delta_w.dtype)
            
            # ([g, c_out//g, c_in//g*k*k] @ [g, c_in//g*k*k, n_components]) * [g, 1, n_components]
            #       = [g, c_out//g, n_components] * [g, 1, n_components]
            #       = [g, c_out//g, n_components]
            #       -> norm([g, c_out//g, n_components], dim=2) = [g, c_out//g]
            #       -> mean([g, c_out//g]) = scalar
            # [g, c_out//g, c_in//g*k*k] @ [g, c_in//g*k*k, 1] = [g, c_out//g, 1]
            #       -> squeeze([g, c_out//g, 1]) = [g, c_out//g]
            #       -> norm([g, c_out//g], dim=1) = [g]
            #       -> mean([g]) = scalar
            loss += self.alpha * (delta_w @ proj.transpose(1, 2)).norm(dim=2).mean() \
                + self.beta * (delta_w @ bias.unsqueeze(-1)).squeeze(-1).norm(dim=1).mean()
        loss = loss / len(self.module_names)
        return loss
    
    def set_parameters(self, components, variances, means, alpha=0.9, beta=0.1):
        self.components = components
        self.variances = variances
        self.means = means
        # Freeze components, variances, and means
        for key in components.keys():
            _component = components[key]
            _scale = variances[key]
            _bias = means[key]

            _scale_cumsum = torch.cumsum(_scale, dim=-1)
            _scale_cumsum_normalized = _scale_cumsum / _scale_cumsum[:, -1]
            max_r = 0
            for i in range(len(_scale_cumsum_normalized)):
                for j in range(len(_scale_cumsum_normalized[i])):
                    if _scale_cumsum_normalized[i][j] > 0.9:
                        max_r = max(max_r, j)
                        break
            
            components[key] = _component[:, :max_r]
            variances[key] = _scale[:, :max_r]

            components[key].requires_grad_(False)
            variances[key].requires_grad_(False)
            means[key].requires_grad_(False)
        
        self.alpha = alpha
        self.beta = beta
