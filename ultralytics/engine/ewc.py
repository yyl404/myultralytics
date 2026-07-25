from typing import Dict

from torch import Tensor

from ultralytics.nn import BaseModel


class EWCLoss:
    """Elastic Weight Consolidation (EWC) loss for incremental learning.
    
    EWC is a regularization method that prevents catastrophic forgetting by penalizing
    changes to parameters that are important for previous tasks. The importance is
    approximated using the Fisher Information Matrix (diagonal approximation).
    """
    
    def __init__(
        self,
        model_update: BaseModel,
        model_base: BaseModel,
        importance: Dict[str, Tensor],
        internal_scale: float = 100.0,
        average_parameters: bool = True,
    ):
        """Initialize EWCLoss.
        
        Args:
            model_update: The model being updated during training
            model_base: The base model (from previous task) for comparison
            importance: Dictionary mapping parameter names to their importance values
                (Fisher Information Matrix diagonal approximation)
            internal_scale: Scale applied before the trainer-level EWC loss weight.
            average_parameters: Whether to average the penalty over tracked parameter tensors.
        """
        self.model_update = model_update
        self.model_base = model_base
        self.importance = importance
        self.internal_scale = internal_scale
        self.average_parameters = average_parameters

        for x in self.importance.values():
            # Freeze importance attributes
            x.requires_grad_(False)

        self.update_weights, self.base_weights = {}, {}
        self._handles = []

    def register_hook(self):
        self.remove_handle_()
        self._handles.append(self.model_update.register_forward_hook(self._hook(self.update_weights, detach=False)))
        self._handles.append(self.model_base.register_forward_hook(self._hook(self.base_weights, detach=True)))
 
    def _hook(self, dict_w: Dict[str, Tensor], detach: bool):
        def fn(module, _, __):
            # This hook is responsible for extracting parameters from the entire model
            # Record all parameters that match keys in importance
            for param_name, param in module.named_parameters():
                # Check if this parameter name is in importance
                if param_name in self.importance:
                    # Keep student parameters connected to autograd; the frozen base is a constant target.
                    dict_w[param_name] = param.detach() if detach else param
        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_loss(self) -> Tensor:
        """Calculate EWC loss.
        
        EWC loss formula: scale * 0.5 * sum(importance * (theta - theta_star)^2).
        where theta is current parameter value and theta_star is base parameter value.
        
        Returns:
            EWC loss as a scalar tensor, optionally averaged over parameter tensors.
        """
        missing_update = self.importance.keys() - self.update_weights.keys()
        missing_base = self.importance.keys() - self.base_weights.keys()
        if missing_update or missing_base:
            raise RuntimeError(
                "EWC parameters were not captured by model forwards: "
                f"student_missing={sorted(missing_update)}, teacher_missing={sorted(missing_base)}"
            )

        first_weight = next(iter(self.update_weights.values()))
        loss = first_weight.new_zeros(())
        # Iterate over all recorded weights and importance to calculate EWC loss
        for param_name in self.importance.keys():
            update_w = self.update_weights[param_name]
            base_w = self.base_weights[param_name]
            importance = self.importance[param_name]
            
            # Ensure tensors are on the same device
            importance = importance.to(update_w.device, update_w.dtype)
            base_w = base_w.to(update_w.device, update_w.dtype)
            
            # EWC loss: internal_scale * 0.5 * sum(importance * (theta - theta_star)^2)
            delta_w = update_w - base_w
            loss += 0.5 * self.internal_scale * (importance * (delta_w ** 2)).sum()
        
        # Average loss over number of parameters
        num_params = len(self.importance)
        if self.average_parameters and num_params > 0:
            loss = loss / num_params
        
        return loss
