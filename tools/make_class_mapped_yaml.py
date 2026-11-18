"""Write a dataset YAML that remaps class IDs when the labels are loaded.

The generated YAML keeps the original image and label directories and adds
``class_id_map`` plus the target class names. Dataset loading applies the map after
reading the label cache, so no converted image/label tree is written.

Examples:
    $ python tools/make_class_mapped_yaml.py \
        --model runs/<run>/task-2/best.pt \
        --dataset data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --output runs/<run>/task-2/task_2_class_map.yaml

"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import normalize_names, resolve_split_roots

from ultralytics import YOLO
from ultralytics.data import build_class_id_map
from ultralytics.utils import LOGGER, YAML

SPLITS = ("train", "val", "test", "minival")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Source dataset YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset YAML")
    parser.add_argument("--model", type=Path, default=None, help="Model whose class space labels should match")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to keep (default: every train/val/test/minival split present in the source YAML)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a source class is absent from the model class space",
    )
    args = parser.parse_args()
    if args.model is None:
        parser.error("--model is required")
    return args


def _dataset_root(dataset_path: Path, config: dict) -> Path:
    """Resolve the dataset root used by the source YAML."""
    configured_root = config.get("path")
    if configured_root is None:
        return dataset_path.parent.resolve()
    root = Path(configured_root)
    if not root.is_absolute():
        root = dataset_path.parent / root
    return root.resolve()


def main() -> None:
    """Build the mapped YAML."""
    args = parse_args()
    dataset_path = Path(args.dataset)
    config = YAML().load(dataset_path)
    if "names" not in config:
        raise KeyError(f"Dataset config has no 'names': {dataset_path}")
    source_names = normalize_names(config["names"], source=f"dataset '{dataset_path}'")

    model = YOLO(args.model)
    model_names = normalize_names(model.names, source=f"model '{args.model}'")
    class_id_map, output_names = build_class_id_map(source_names, model_names)

    missing = [
        class_name
        for source_id, class_name in source_names.items()
        if source_id not in class_id_map
    ]
    if args.strict and missing:
        raise ValueError(
            f"Classes {missing} from '{dataset_path}' are missing from the model class space "
            f"{list(output_names.values())}"
        )

    splits = args.splits or [split for split in SPLITS if split in config]
    if not splits:
        raise KeyError(f"No train/val/test/minival split found in '{dataset_path}'")
    split_paths = {}
    for split in splits:
        if split not in config:
            raise KeyError(f"Split '{split}' is not present in '{dataset_path}'")
        image_roots, _ = resolve_split_roots(dataset_path, config, split)
        split_paths[split] = (
            str(image_roots[0]) if len(image_roots) == 1 else [str(root) for root in image_roots]
        )

    output_config = {
        "path": str(_dataset_root(dataset_path, config)),
        "names": output_names,
        "nc": len(output_names),
        "source_nc": len(source_names),
        "class_id_map": class_id_map,
    }
    if "channels" in config:
        output_config["channels"] = config["channels"]
    output_config.update(split_paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    YAML().save(args.output, output_config)
    LOGGER.info(
        f"Class-mapped dataset YAML written to '{args.output}' "
        f"({len(class_id_map)}/{len(source_names)} source classes mapped, {len(output_names)} target classes)"
    )


if __name__ == "__main__":
    main()
