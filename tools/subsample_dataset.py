"""Subsample a YOLO dataset into a smaller variant (e.g. VOC -> VOC-TINY for debugging).

Each requested split is uniformly subsampled to the given fraction (seeded, without
replacement); images and labels are symlinked into the output directory so the subsampled
dataset costs no extra disk space.

Usage:
    $ python tools/subsample_dataset.py \
        --source_cfg data/VOC-YOLO/VOC.yaml \
        --output_dir data/VOC-TINY-YOLO \
        --fraction 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp")


def _resolve_split_dirs(source_cfg: Path, dataset_config: dict, split: str) -> tuple[Path, Path] | None:
    """Resolve the image and label root directories of one split; None when the split is absent."""
    split_value = dataset_config.get(split)
    if split_value is None:
        return None
    if not isinstance(split_value, str):
        raise TypeError(f"Split '{split}' in '{source_cfg}' must be a path, got {split_value!r}")
    image_root = Path(split_value)
    if not image_root.is_absolute():
        image_root = source_cfg.parent / image_root
    image_root = image_root.resolve()
    label_root = Path(str(image_root).replace("images", "labels"))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory for split '{split}' does not exist: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory for split '{split}' does not exist: {label_root}")
    return image_root, label_root


def _subsample_split(image_root: Path, label_root: Path, output_dir: Path, split: str, fraction: float, rng: random.Random) -> int:
    """Symlink a seeded random subset of one split into the output dataset directory."""
    images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No images found under {image_root}")
    keep = round(len(images) * fraction)
    selected = rng.sample(images, min(keep, len(images)))
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path in selected:
        (image_dir / image_path.name).symlink_to(image_path)
        label_path = label_root / f"{image_path.stem}.txt"
        if label_path.is_file():
            (label_dir / label_path.name).symlink_to(label_path)
    return len(selected)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cfg", type=Path, required=True, help="Source dataset YAML")
    parser.add_argument("--output_dir", type=Path, required=True, help="Subsampled dataset directory to create")
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of each split to keep, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    return args


def main() -> None:
    """Subsample every requested split and save the derived dataset config."""
    args = parse_args()
    dataset_config = YAML().load(args.source_cfg)
    if "names" not in dataset_config:
        raise KeyError(f"Source dataset config has no 'names': {args.source_cfg}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    output_config = {"names": dataset_config["names"]}
    for split in args.splits:
        resolved = _resolve_split_dirs(args.source_cfg, dataset_config, split)
        if resolved is None:
            LOGGER.warning(f"Source dataset config '{args.source_cfg}' has no '{split}' split; skipping")
            continue
        image_root, label_root = resolved
        kept = _subsample_split(image_root, label_root, args.output_dir, split, args.fraction, rng)
        output_config[split] = f"images/{split}"
        LOGGER.info(f"{split}: kept {kept} images ({args.fraction:.0%} of source)")

    YAML().save(args.output_dir / args.source_cfg.name, output_config)
    LOGGER.info(f"Subsampled dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
