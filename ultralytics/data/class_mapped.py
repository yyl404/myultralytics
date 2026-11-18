# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Class-id remapping applied when a YOLO dataset loads its labels.

Incremental stages keep their labels in the dataset's own class space. The loader
applies ``data["class_id_map"]`` after reading the label cache, so labels match the
model's output channels without writing a converted label tree or a second cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import repeat

import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import verify_image_label
from ultralytics.utils import LOGGER


def build_class_id_map(
    source_names: Mapping[int, str],
    target_names: Mapping[int, str],
) -> tuple[dict[int, int], dict[int, str]]:
    """Map source class IDs to target class IDs by class name.

    Args:
        source_names: Dataset class names, keyed by the ID stored in the label files.
        target_names: Model class names, keyed by the model output channel.

    Returns:
        (tuple): ``(class_id_map, output_names)`` where ``class_id_map`` maps source IDs
            to target IDs and ``output_names`` is the model class space. Source classes
            absent from the target are skipped with a warning.
    """
    target_ids_by_name = {}
    for class_id, class_name in target_names.items():
        target_ids_by_name.setdefault(class_name, class_id)

    class_id_map = {}
    for source_id, class_name in source_names.items():
        target_id = target_ids_by_name.get(class_name)
        if target_id is None:
            LOGGER.warning(
                f"Class '{class_name}' is not present in the model class space and will be skipped"
            )
        else:
            class_id_map[source_id] = target_id
    return class_id_map, dict(target_names)


class ClassMappedYOLODataset(YOLODataset):
    """YOLODataset that remaps label class IDs into the model output space at load time.

    The on-disk labels and the label cache stay in the source class space. The map in
    ``data["class_id_map"]`` is applied after the cache is read, so one cache can be
    shared by every model class space and no converted label/image tree is written.

    Args:
        *args: Positional arguments forwarded to :class:`YOLODataset`.
        data: Dataset config; must contain ``class_id_map``.
        **kwargs: Keyword arguments forwarded to :class:`YOLODataset`.
    """

    def __init__(self, *args, data: dict | None = None, **kwargs):
        """Store the class map and build the dataset."""
        data = data or {}
        raw_map = data.get("class_id_map")
        if raw_map is None:
            raise KeyError("ClassMappedYOLODataset requires data['class_id_map']")
        self.class_id_map = {int(source): int(target) for source, target in raw_map.items()}
        self._class_id_map_applied = False
        super().__init__(*args, data=data, **kwargs)

    def _drop_all_labels(self, labels: list[dict]) -> list[dict]:
        """Keep every image but remove all annotations when no source class maps."""
        for label in labels:
            label["cls"] = np.zeros((0, 1), dtype=label["cls"].dtype)
            label["bboxes"] = label["bboxes"][:0]
            if label.get("segments"):
                label["segments"] = []
            if label.get("keypoints") is not None:
                label["keypoints"] = label["keypoints"][:0]
        return labels

    def verify_args(self) -> tuple:
        """Verify labels against the SOURCE class count, not the target model space.

        The cache is shared by every model class space; using the target count here
        would drop source classes that a smaller target model does not know yet.
        """
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        source_nc = int(self.data.get("source_nc", max(self.class_id_map, default=-1) + 1))
        return verify_image_label, zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(source_nc),
            repeat(nkpt),
            repeat(ndim),
            repeat(self.single_cls),
        )

    def get_labels(self) -> list[dict]:
        """Return cached labels with class IDs remapped into the model output space."""
        # The on-disk cache must be built in the SOURCE class space. A model whose
        # target space is smaller (e.g. the 15-class stage-1 on a 20-class cumulative
        # test set) would otherwise make the validator drop the new-class labels
        # while writing the shared cache.
        target_nc = int(self.data.get("nc", len(self.data.get("names", {}))))
        source_nc = int(self.data.get("source_nc", max(self.class_id_map, default=-1) + 1))
        self.nc = max(source_nc, target_nc)
        try:
            labels = super().get_labels()
        finally:
            self.nc = target_nc
        if self._class_id_map_applied:
            return labels
        self._class_id_map_applied = True
        if not self.class_id_map:
            return self._drop_all_labels(labels)

        for label in labels:
            cls = label["cls"].reshape(-1)
            mapped = np.array(
                [self.class_id_map.get(int(value), -1) for value in cls],
                dtype=label["cls"].dtype,
            )
            keep = mapped >= 0
            label["cls"] = mapped[keep].reshape(-1, 1)
            label["bboxes"] = label["bboxes"][keep]
            if label.get("segments"):
                label["segments"] = [
                    segment for segment, keep_item in zip(label["segments"], keep) if keep_item
                ]
            if label.get("keypoints") is not None:
                label["keypoints"] = label["keypoints"][keep]
        return labels
