# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stage-aware detection validation for the periodic cumulative test pass.

The validator aggregates the standard per-class APs into the class groups of every
incremental stage and prints one mAP50 / mAP50-95 line per stage. It is used only by
the diagnostic cumulative test evaluation; the regular training validation and
checkpoint selection keep using the standard DetectionValidator.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER


class StageAwareDetectionValidator(DetectionValidator):
    """DetectionValidator that also aggregates metrics over stage class groups.

    Args:
        stage_classes: Per-stage class-name lists, ordered from the first to the
            current incremental stage.
    """

    def __init__(self, *args, stage_classes=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage_classes = [list(stage) for stage in (stage_classes or [])]
        self.stage_results: list[dict] = []

    def get_stats(self) -> dict:
        """Compute the standard metrics, then the per-stage aggregates."""
        stats = super().get_stats()
        self._compute_stage_metrics()
        return stats

    def _compute_stage_metrics(self) -> None:
        """Aggregate per-class AP into each stage's class group."""
        self.stage_results = []
        if not self.stage_classes:
            return
        names = self.names
        name_to_id = (
            {name: class_id for class_id, name in names.items()}
            if isinstance(names, dict)
            else {name: class_id for class_id, name in enumerate(names)}
        )
        box = self.metrics.box
        ap50, ap = box.ap50, box.ap
        if len(ap50) == 0 or len(ap) == 0:
            LOGGER.warning("Stage-wise metrics unavailable: no validation labels were found")
            return
        present = {int(class_id): index for index, class_id in enumerate(box.ap_class_index)}
        for stage_index, stage in enumerate(self.stage_classes, start=1):
            class_ids = [name_to_id[name] for name in stage if name in name_to_id]
            selected = [present[class_id] for class_id in class_ids if class_id in present]
            if not selected:
                self.stage_results.append(
                    {"stage": stage_index, "classes": len(class_ids), "present": 0, "mAP50": None, "mAP": None}
                )
                continue
            self.stage_results.append(
                {
                    "stage": stage_index,
                    "classes": len(class_ids),
                    "present": len(selected),
                    "mAP50": float(ap50[selected].mean()),
                    "mAP": float(ap[selected].mean()),
                }
            )

    def print_results(self) -> None:
        """Print the standard results and one aggregated line per stage."""
        super().print_results()
        for result in self.stage_results:
            label = f"stage {result['stage']}"
            if result is self.stage_results[-1]:
                label += " (current)"
            if result["mAP"] is None:
                counts = f"0/{result['classes']} classes"
                LOGGER.info(f"{label:>22s}{counts:>22s}  no instances")
                continue
            counts = f"{result['present']}/{result['classes']} classes"
            LOGGER.info(f"{label:>22s}{counts:>22s}  mAP50={result['mAP50']:.3f}  mAP50-95={result['mAP']:.3f}")
