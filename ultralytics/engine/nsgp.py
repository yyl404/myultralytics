from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ultralytics.engine.spectrum import find_elbow_point
from ultralytics.utils import LOGGER


class NSGP:
    """Project convolution updates into null spaces estimated from old-task inputs."""

    def __init__(
        self,
        module_names: Sequence[str],
        components: Mapping[str, Tensor],
        eigen_values: Mapping[str, Tensor],
        normalized_module_names: Sequence[str] = (),
        param_names: Mapping[str, str] | None = None,
    ) -> None:
        """Build null-space projection matrices.

        The cached basis is folded into ``P = B B^T`` once, so every optimizer step
        only needs one matmul per group. Modules listed in ``normalized_module_names``
        use ``P / sqrt(k)``, matching the reference implementation's Frobenius
        normalization for the backbone.

        Args:
            module_names: Names of convolution modules governed by NSGP.
            components: PCA row vectors, each shaped (groups, components, input_dim).
            eigen_values: Descending PCA eigenvalues, shaped (groups, components).
            normalized_module_names: Modules using the Frobenius-normalized projector.
            param_names: Module name -> governed parameter attribute (default ``weight``).
        """
        self.module_names = tuple(module_names)
        param_names = param_names or {}
        self.weight_names = {
            name: f"{name}.{param_names.get(name, 'weight')}" for name in self.module_names
        }
        self.normalized_module_names = frozenset(normalized_module_names)
        self.elbow_indices: dict[str, tuple[int, ...]] = {}
        self.projections: dict[str, tuple[Tensor, ...]] = {}

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
            projections = []
            for group_components, group_values in zip(module_components, module_eigen_values):
                elbow_idx = find_elbow_point(group_values)
                elbow_idx = max(0, min(elbow_idx, group_values.numel() - 1))
                elbow_indices.append(elbow_idx)

                # PCA vectors are rows. The official NSGP implementation retains
                # directions from the adaptive elbow onward as the null subspace.
                basis = group_components[elbow_idx:].T.contiguous()  # (input_dim, num_null)
                projection = basis @ basis.T
                if name in self.normalized_module_names:
                    projection = projection / max(basis.shape[1] ** 0.5, 1.0)
                projections.append(projection)

            self.elbow_indices[name] = tuple(elbow_indices)
            self.projections[name] = tuple(projections)
            LOGGER.info(f"NSGP module '{name}': null-space starts at PCA components {self.elbow_indices[name]}")

        self.params: dict[str, torch.nn.Parameter] | None = None
        self.snapshot: dict[str, Tensor] | None = None

    @torch.no_grad()
    def _bind(self, params_dict: Mapping[str, torch.nn.Parameter]) -> None:
        """Resolve the governed weights once and allocate their pre-step snapshot."""
        if self.params is not None:
            return
        params, snapshot = {}, {}
        for name in self.module_names:
            weight_name = self.weight_names[name]
            if weight_name not in params_dict:
                raise KeyError(f"NSGP parameter '{weight_name}' was not found in the model")
            weight = params_dict[weight_name].detach()
            if weight.ndim not in (2, 4):
                raise ValueError(f"NSGP supports Conv2d/Linear weights, got shape {tuple(weight.shape)}")
            params[weight_name] = params_dict[weight_name]
            snapshot[weight_name] = torch.empty_like(weight)
        self.params, self.snapshot = params, snapshot

        device = next(iter(params.values())).device
        self.projections = {
            name: tuple(projection.to(device) for projection in projections)
            for name, projections in self.projections.items()
        }

    @torch.no_grad()
    def capture_parameters(self, params_dict: Mapping[str, torch.nn.Parameter]) -> None:
        """Record the governed weights immediately before the optimizer step."""
        self._bind(params_dict)
        for weight_name, param in self.params.items():
            self.snapshot[weight_name].copy_(param.detach())

    @torch.no_grad()
    def apply_parameter_projection(self, flexibility: float = 1.0) -> None:
        """Project the completed optimizer update and write the projected parameters.

        ``flexibility=1`` applies the exact NSGP projection. Values in ``[0, 1]``
        interpolate between the original update and its null-space projection.
        """
        if not 0.0 <= flexibility <= 1.0:
            raise ValueError(f"NSGP flexibility must be in [0, 1], got {flexibility}")
        if self.params is None:
            raise RuntimeError("NSGP parameters were not captured; call capture_parameters() before the optimizer step")

        for name in self.module_names:
            weight_name = self.weight_names[name]
            weight_param = self.params[weight_name]
            weight_before = self.snapshot[weight_name]
            bases = self.projections[name]
            if weight_param.ndim == 2:
                # nn.Linear: (out_features, in_features), one PCA group.
                out_channels, in_features = weight_param.shape
                if len(bases) != 1:
                    raise ValueError(f"Linear module '{name}' must have exactly one PCA group, got {len(bases)}")
                projection = bases[0]
                if projection.shape[0] != in_features:
                    raise ValueError(
                        f"NSGP input dimension mismatch for '{name}': projection={projection.shape[0]}, "
                        f"update={in_features}"
                    )
                update = weight_param - weight_before
                weight_param.copy_(weight_before + torch.lerp(update, update @ projection, flexibility))
                continue

            out_channels, in_per_group, kernel_h, kernel_w = weight_param.shape
            groups = len(bases)
            if out_channels % groups:
                raise ValueError(f"Output channels {out_channels} are not divisible by PCA groups {groups}")

            grouped_update = (weight_param - weight_before).reshape(
                groups, out_channels // groups, in_per_group * kernel_h * kernel_w
            )
            projected_groups = []
            for group_idx, projection in enumerate(bases):
                group_update = grouped_update[group_idx]  # (out_per_group, input_dim)
                if projection.shape[0] != group_update.shape[1]:
                    raise ValueError(
                        f"NSGP input dimension mismatch for '{name}' group {group_idx}: "
                        f"projection={projection.shape[0]}, update={group_update.shape[1]}"
                    )
                null_update = group_update @ projection
                projected_groups.append(torch.lerp(group_update, null_update, flexibility))
            projected = torch.stack(projected_groups)
            weight_param.copy_(
                weight_before + projected.reshape(out_channels, in_per_group, kernel_h, kernel_w)
            )
