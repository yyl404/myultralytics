from typing import Dict

from torch import Tensor

from ultralytics.nn import BaseModel
from ultralytics.utils import LOGGER


class EWCLoss:
    """Elastic Weight Consolidation (EWC) loss for incremental learning.
    
    EWC is a regularization method that prevents catastrophic forgetting by penalizing
    changes to parameters that are important for previous tasks. The importance is
    approximated using the Fisher Information Matrix (diagonal approximation).
    """
    
    def __init__(self, model_update: BaseModel, model_base: BaseModel, importance: Dict[str, Tensor]):
        """Initialize EWCLoss.
        
        Args:
            model_update: The model being updated during training
            model_base: The base model (from previous task) for comparison
            importance: Dictionary mapping parameter names to their importance values
                (Fisher Information Matrix diagonal approximation)
        """
        self.model_update = model_update
        self.model_base = model_base
        self.importance = importance

        for x in self.importance.values():
            # Freeze importance attributes
            x.requires_grad_(False)

        self.update_weights, self.base_weights = {}, {}
        self._handles = []

    def register_hook(self):
        self.remove_handle_()
        self._handles.append(self.model_update.register_forward_hook(self._hook(self.update_weights)))
        self._handles.append(self.model_base.register_forward_hook(self._hook(self.base_weights)))
 
    def _hook(self, dict_w: Dict[str, Tensor]):
        def fn(module, _, __):
            # This hook is responsible for extracting parameters from the entire model
            # Record all parameters that match keys in importance
            for param_name, param in module.named_parameters():
                # Check if this parameter name is in importance
                if param_name in self.importance:
                    # Record parameter value using parameter name as key
                    dict_w[param_name] = param.data.clone()
        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self) -> Tensor:
        """Calculate EWC loss.
        
        EWC loss formula: 0.5 * sum(importance * (theta - theta_star)^2) / num_params
        where theta is current parameter value and theta_star is base parameter value.
        
        Returns:
            EWC loss as a scalar tensor, averaged over number of parameters
        """
        loss = 0
        # Iterate over all recorded weights and importance to calculate EWC loss
        for param_name in self.importance.keys():
            if param_name not in self.update_weights:
                LOGGER.warning(f"Parameter '{param_name}' not found in update_weights, skipping")
                continue
            if param_name not in self.base_weights:
                LOGGER.warning(f"Parameter '{param_name}' not found in base_weights, skipping")
                continue
            
            update_w = self.update_weights[param_name]
            base_w = self.base_weights[param_name]
            importance = self.importance[param_name]
            
            # Ensure tensors are on the same device
            importance = importance.to(update_w.device, update_w.dtype)
            base_w = base_w.to(update_w.device, update_w.dtype)
            
            # EWC loss: 0.5 * sum(importance * (theta - theta_star)^2)
            delta_w = update_w - base_w
            loss += 0.5 * (importance * (delta_w ** 2)).sum()
        
        # Average loss over number of parameters
        num_params = len([p for p in self.importance.keys() 
                         if p in self.update_weights and p in self.base_weights])
        if num_params > 0:
            loss = loss / num_params
        
        return loss
