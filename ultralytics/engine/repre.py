"""Regional Prototype Replay for incremental YOLO detection."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER


def _restore_prototype_patches(features: Tensor, valid_masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Center valid regions in prototype patches.

    Args:
        features: Prototype features shaped (N, C, 5, 5).
        valid_masks: Valid-pixel masks shaped (N, 5, 5).

    Returns:
        Centered features (N, C, 5, 5) and output row/column indices (N,).
    """
    if features.ndim != 4 or features.shape[-2:] != (5, 5):
        raise ValueError(f"RePRE features must have shape (N, C, 5, 5), got {tuple(features.shape)}")
    if valid_masks.shape != (features.shape[0], 5, 5):
        raise ValueError(
            f"RePRE mask shape must be ({features.shape[0]}, 5, 5), got {tuple(valid_masks.shape)}"
        )

    restored = torch.zeros_like(features)
    output_rows = torch.full((features.shape[0],), 2, device=features.device, dtype=torch.long)
    output_cols = torch.full((features.shape[0],), 2, device=features.device, dtype=torch.long)
    for prototype_idx in range(features.shape[0]):
        mask = valid_masks[prototype_idx] > 0
        valid_rows = torch.where(mask.any(dim=1))[0]
        valid_cols = torch.where(mask.any(dim=0))[0]
        if valid_rows.numel() == 0 or valid_cols.numel() == 0:
            raise ValueError(f"RePRE prototype {prototype_idx} has an empty valid mask")

        row_start, row_end = valid_rows[0].item(), valid_rows[-1].item() + 1
        col_start, col_end = valid_cols[0].item(), valid_cols[-1].item() + 1
        patch = features[prototype_idx, :, row_start:row_end, col_start:col_end]
        target_row_start = (5 - patch.shape[-2]) // 2
        target_col_start = (5 - patch.shape[-1]) // 2
        restored[
            prototype_idx,
            :,
            target_row_start : target_row_start + patch.shape[-2],
            target_col_start : target_col_start + patch.shape[-1],
        ] = patch
        output_rows[prototype_idx] = target_row_start + (2 - row_start)
        output_cols[prototype_idx] = target_col_start + (2 - col_start)
    return restored, output_rows, output_cols


class RegionalPrototypeReplay:
    """Replay coarse and density-aware old-class prototypes through YOLO's classification branch."""

    def __init__(
        self,
        detect_head: Detect,
        prototype_data: Mapping,
        device: torch.device,
    ) -> None:
        """Initialize RePRE.

        Args:
            detect_head: Student YOLO detection head.
            prototype_data: Artifact containing one RePRE entry per detection scale.
            device: Training device.
        """
        if not isinstance(detect_head, Detect):
            raise TypeError(f"RePRE requires a Detect-compatible head, got {type(detect_head)}")
        if "repre" not in prototype_data:
            raise KeyError("Prototype artifact has no 'repre' data; regenerate it with --selection density")
        self.detect_head = detect_head
        self.device = device
        artifact_levels = tuple(prototype_data["repre"])
        if len(artifact_levels) != detect_head.nl:
            raise ValueError(f"RePRE has {len(artifact_levels)} levels, but detection head has {detect_head.nl}")

        levels = []
        old_class_count = 0
        for level_idx, level in enumerate(artifact_levels):
            required = {"features", "valid_masks", "labels"}
            missing = required.difference(level)
            if missing:
                raise KeyError(f"RePRE level {level_idx} is missing keys: {sorted(missing)}")
            num_prototypes = level["features"].shape[0]
            if level["valid_masks"].shape[0] != num_prototypes or level["labels"].shape != (num_prototypes,):
                raise ValueError(f"Inconsistent RePRE prototype counts at level {level_idx}")
            if num_prototypes and int(level["labels"].max()) >= detect_head.nc:
                raise ValueError(
                    f"RePRE level {level_idx} contains class {int(level['labels'].max())}, "
                    f"but model has {detect_head.nc} classes"
                )
            if num_prototypes:
                old_class_count = max(old_class_count, int(level["labels"].max()) + 1)
            levels.append(
                {
                    "features": level["features"].to(device),
                    "valid_masks": level["valid_masks"].to(device),
                    "labels": level["labels"].to(device=device, dtype=torch.long),
                }
            )
        self.levels = tuple(levels)
        self.old_class_count = old_class_count
        self.has_prototypes = old_class_count > 0
        if not self.has_prototypes:
            LOGGER.warning(
                "RePRE artifact contains no prototypes; replay loss will be zero until "
                "a later task produces valid prototypes"
            )

    def compute_loss(self) -> Tensor:
        """Return classification-only replay loss for one training iteration."""
        if not self.has_prototypes:
            return torch.zeros((), device=self.device)

        level_losses = []
        classifier_states = [module.training for module in self.detect_head.cv3]
        try:
            self.detect_head.cv3.eval()
            for level_idx, level in enumerate(self.levels):
                features = level["features"]
                if features.shape[0] == 0:
                    continue

                features_batch = features
                masks_batch = level["valid_masks"]
                labels_batch = level["labels"]

                features_batch, output_rows, output_cols = _restore_prototype_patches(
                    features_batch, masks_batch
                )
                logits_map = self.detect_head.cv3[level_idx](features_batch)
                prototype_idx = torch.arange(features_batch.shape[0], device=self.device)
                logits = logits_map[prototype_idx, :, output_rows, output_cols]  # (N, nc)
                logits = logits[:, : self.old_class_count]
                # Match the official implementation, which passes probabilities
                # rather than raw logits to cross_entropy.
                level_losses.append(F.cross_entropy(logits.float().softmax(dim=-1), labels_batch))
        finally:
            for classifier, was_training in zip(self.detect_head.cv3, classifier_states):
                classifier.train(was_training)

        if not level_losses:
            raise RuntimeError("RePRE artifact contains no prototypes")
        return torch.stack(level_losses).mean()
