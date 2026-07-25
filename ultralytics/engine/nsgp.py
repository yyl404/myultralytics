from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ultralytics.engine.espreg import find_elbow_point
from ultralytics.utils import LOGGER


class NSGP:
    """Project convolution updates into null spaces estimated from old-task inputs."""

    def __init__(
        self,
        module_names: Sequence[str],
        components: Mapping[str, Tensor],
        eigen_values: Mapping[str, Tensor],
        normalized_module_names: Sequence[str] = (),
    ) -> None:
        """Build null-space bases.

        Args:
            module_names: Names of convolution modules governed by NSGP.
            components: PCA row vectors, each shaped (groups, components, input_dim).
            eigen_values: Descending PCA eigenvalues, shaped (groups, components).
            normalized_module_names: Modules using the Frobenius-normalized projector.
        """
        self.module_names = tuple(module_names)
        self.normalized_module_names = frozenset(normalized_module_names)
        self.elbow_indices: dict[str, tuple[int, ...]] = {}
        self.null_bases: dict[str, tuple[Tensor, ...]] = {}

        missing = [name for name in self.module_names if name not in components or name not in eigen_values]
        if missing:
            raise KeyError(f"Missing PCA values for NSGP modules: {missing}")

        for name in self.module_names:
            module_components = components[name].detach()
            module_eigen_values = eigen_values[name].detach()
            if module_components.ndim != 3 or module_eigen_values.ndim != 2:
                raise ValueError(
                    f"Invalid PCA rank for '{name}': components={tuple(module_components.shape)}, "
                    f"eigen_values={tuple(module_eigen_values.shape)}"
                )
            if module_components.shape[:2] != module_eigen_values.shape:
                raise ValueError(
                    f"Incompatible PCA shapes for '{name}': components={tuple(module_components.shape)}, "
                    f"eigen_values={tuple(module_eigen_values.shape)}"
                )

            elbow_indices = []
            null_bases = []
            for group_components, group_values in zip(module_components, module_eigen_values):
                elbow_idx = find_elbow_point(group_values)
                elbow_idx = max(0, min(elbow_idx, group_values.numel() - 1))
                elbow_indices.append(elbow_idx)

                # PCA vectors are rows. The official NSGP implementation retains
                # directions from the adaptive elbow onward as the null subspace.
                null_bases.append(group_components[elbow_idx:].T.contiguous())

            self.elbow_indices[name] = tuple(elbow_indices)
            self.null_bases[name] = tuple(null_bases)
            LOGGER.info(
                f"NSGP module '{name}': null-space starts at PCA components {self.elbow_indices[name]}"
            )

    @torch.no_grad()
    def capture_parameters(self, params_dict: Mapping[str, nn.Parameter]) -> dict[str, Tensor]:
        """Clone governed weights immediately before the optimizer step."""
        captured = {}
        for name in self.module_names:
            weight_name = f"{name}.weight"
            if weight_name not in params_dict:
                raise KeyError(f"NSGP parameter '{weight_name}' was not found in the model")
            captured[weight_name] = params_dict[weight_name].detach().clone()
        return captured

    @torch.no_grad()
    def apply_parameter_projection(
        self,
        params_dict: Mapping[str, nn.Parameter],
        parameters_before_step: Mapping[str, Tensor],
        flexibility: float = 1.0,
    ) -> None:
        """Project completed optimizer updates and write the projected parameters.

        ``flexibility=1`` applies the exact NSGP projection. Values in ``[0, 1]``
        interpolate between the original update and its null-space projection.
        """
        if not 0.0 <= flexibility <= 1.0:
            raise ValueError(f"NSGP flexibility must be in [0, 1], got {flexibility}")

        for name in self.module_names:
            weight_name = f"{name}.weight"
            if weight_name not in params_dict:
                raise KeyError(f"NSGP parameter '{weight_name}' was not found in the model")
            if weight_name not in parameters_before_step:
                raise KeyError(f"Missing pre-step NSGP parameter snapshot: '{weight_name}'")

            weight_param = params_dict[weight_name]
            weight_before = parameters_before_step[weight_name]
            if weight_param.ndim != 4:
                raise ValueError(f"NSGP supports Conv2d weights, got shape {tuple(weight_param.shape)}")
            bases = self.null_bases[name]
            out_channels, in_per_group, kernel_h, kernel_w = weight_param.shape
            groups = len(bases)
            if out_channels % groups:
                raise ValueError(f"Output channels {out_channels} are not divisible by PCA groups {groups}")
            parameter_update = weight_param - weight_before
            grouped_update = parameter_update.reshape(
                groups, out_channels // groups, in_per_group * kernel_h * kernel_w
            )

            projected_groups = []
            for group_idx, basis in enumerate(bases):
                basis = basis.to(device=grouped_update.device, dtype=grouped_update.dtype)
                group_update = grouped_update[group_idx]  # (out_per_group, input_dim)
                if basis.shape[0] != group_update.shape[1]:
                    raise ValueError(
                        f"NSGP input dimension mismatch for '{name}' group {group_idx}: "
                        f"basis={basis.shape[0]}, update={group_update.shape[1]}"
                    )
                null_update = group_update @ basis @ basis.T
                if name in self.normalized_module_names:
                    null_update = null_update / max(basis.shape[1] ** 0.5, 1.0)
                projected_groups.append(torch.lerp(group_update, null_update, flexibility))
            projected = torch.stack(projected_groups)
            weight_param.copy_(
                weight_before
                + projected.reshape(out_channels, in_per_group, kernel_h, kernel_w)
            )

