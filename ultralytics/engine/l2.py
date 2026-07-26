"""Naive L2 regularization towards a reference (previous-task) model."""

from __future__ import annotations

from torch import Tensor, nn

from ultralytics.utils import LOGGER


class L2Loss:
    """Compute ``0.5 * sum_i (theta[i] - theta_ref[i])**2`` over shared parameters.

    Only parameters present in both models with matching shapes are penalized; reference
    parameters with a different shape (e.g. an expanded detection head) are skipped.
    """

    def __init__(self, model: nn.Module, ref_model: nn.Module):
        """Bind frozen reference parameters to the current model's named parameters."""
        self.model = model
        model_params = dict(model.named_parameters())
        ref_params = dict(ref_model.named_parameters())
        shared_names = model_params.keys() & ref_params.keys()
        if not shared_names:
            raise ValueError("L2 regularization found no parameter names shared with the reference model")

        self.ref_params: dict[str, Tensor] = {}
        skipped = []
        for name in sorted(shared_names):
            current = model_params[name]
            ref = ref_params[name]
            if ref.shape != current.shape:
                skipped.append(name)  # e.g. expanded head channels have no previous-task counterpart
                continue
            self.ref_params[name] = ref.to(device=current.device, dtype=current.dtype).detach().requires_grad_(False)
        if not self.ref_params:
            raise ValueError(
                f"L2 regularization found no shape-matching parameters with the reference model "
                f"(skipped {len(skipped)} mismatched: {skipped})"
            )
        if skipped:
            LOGGER.info(f"L2 regularization skips {len(skipped)} shape-mismatched parameters: {skipped}")

    def get_loss(self) -> Tensor:
        """Return the scalar L2 penalty, including its explicit one-half factor."""
        model_params = dict(self.model.named_parameters())
        first_param = model_params[next(iter(self.ref_params))]
        loss = first_param.new_zeros(())
        for name, ref in self.ref_params.items():
            loss = loss + 0.5 * (model_params[name] - ref).square().sum()
        return loss
