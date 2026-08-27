"""Select replay samples from a finished task's training set for experience replay.

After each incremental task, a fixed number of training samples (images + ground-truth
labels, already in the model's global class-id space) is kept as replay data for the
next tasks. Selection is pluggable: REPLAY_STRATEGIES maps a strategy name to a callable

    strategy(records: list[ReplayRecord], num: int, rng: random.Random) -> list[ReplayRecord]

so distribution-aware strategies (e.g. feature-based herding) can be registered next to
the default "random" strategy without changing the CLI or the trainer.

Usage:
    $ python tools/select_replay_samples.py \
        --dataset data/VOC_15+5/task_1_cls_15/dataset.yaml \
        --output_dir runs/<run>/task-1/replay_dataset \
        --num 100 --strategy random --seed 0 \
        [--load_hist runs/<run>/task-0/replay_dataset]
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp")


@dataclass(frozen=True)
class ReplayRecord:
    """One replay sample: an image and its (possibly absent) YOLO label file."""

    image_path: Path
    label_path: Path | None  # None when the image has no label file (background image)


def select_random(records: list[ReplayRecord], num: int, rng: random.Random) -> list[ReplayRecord]:
    """Uniformly sample replay records without replacement."""
    return rng.sample(records, num)


# Registry of replay selection strategies. Add distribution-aware strategies here.
REPLAY_STRATEGIES = {
    "random": select_random,
}


def _resolve_split_dirs(dataset_path: Path, dataset_config: dict, split: str) -> tuple[Path, Path]:
    """Resolve the image and label root directories of one dataset split."""
    dataset_path = dataset_path.resolve()
    split_value = dataset_config.get(split)
    if not isinstance(split_value, str):
        raise KeyError(f"Dataset config '{dataset_path}' has no '{split}' split")
    configured_root = dataset_config.get("path")
    dataset_root = dataset_path.parent if configured_root is None else Path(configured_root)
    if not dataset_root.is_absolute():
        dataset_root = dataset_path.parent / dataset_root
    image_root = Path(split_value)
    if not image_root.is_absolute():
        image_root = dataset_root / image_root
    image_root = image_root.resolve()
    label_root = Path(str(image_root).replace("images", "labels"))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory for split '{split}' does not exist: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory for split '{split}' does not exist: {label_root}")
    return image_root, label_root


def _collect_records(image_root: Path, label_root: Path) -> list[ReplayRecord]:
    """Pair every train image with its label file, sorted for deterministic sampling."""
    records = []
    for image_path in sorted(image_root.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        label_path = label_root / f"{image_path.stem}.txt"
        records.append(ReplayRecord(image_path=image_path, label_path=label_path if label_path.is_file() else None))
    return records


def _materialize_records(records: list[ReplayRecord], output_dir: Path, split: str = "train") -> int:
    """Link images and copy labels of the selected records into the replay dataset directory."""
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        destination_image = image_dir / record.image_path.name
        if destination_image.exists():
            # Tasks can share images (sample-filter splits); replaying the same image twice adds nothing
            continue
        destination_image.symlink_to(record.image_path.resolve())
        if record.label_path is not None:
            shutil.copy2(record.label_path, label_dir / record.label_path.name)
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Finished task dataset YAML (global class IDs)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Replay dataset directory to create")
    parser.add_argument("--num", type=int, required=True, help="Number of replay samples to keep from this task")
    parser.add_argument("--strategy", type=str, default="random", choices=sorted(REPLAY_STRATEGIES))
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--load_hist",
        type=Path,
        default=None,
        help="Previous task's replay dataset directory; its samples are carried over (cumulative replay)",
    )
    args = parser.parse_args()
    if args.num < 1:
        parser.error("--num must be at least 1")
    return args


def main() -> None:
    """Select this task's replay samples and merge them with the replay history."""
    args = parse_args()
    dataset_config = YAML().load(args.dataset)
    if "names" not in dataset_config:
        raise KeyError(f"Dataset config has no 'names': {args.dataset}")

    image_root, label_root = _resolve_split_dirs(args.dataset, dataset_config, split="train")
    records = _collect_records(image_root, label_root)
    if not records:
        raise RuntimeError(f"No replay candidates found under {image_root}")

    rng = random.Random(args.seed)
    num = min(args.num, len(records))
    if num < args.num:
        LOGGER.warning(f"Requested {args.num} replay samples but only {len(records)} candidates exist; keeping all")
    selected = REPLAY_STRATEGIES[args.strategy](records, num, rng)
    LOGGER.info(f"Selected {len(selected)}/{len(records)} replay samples with strategy '{args.strategy}'")

    history_names = {}
    history_records: list[ReplayRecord] = []
    if args.load_hist is not None:
        history_config = YAML().load(args.load_hist / "dataset.yaml")
        history_names = dict(history_config.get("names", {}))
        history_image_root, history_label_root = _resolve_split_dirs(
            args.load_hist / "dataset.yaml", history_config, split="train"
        )
        history_records = _collect_records(history_image_root, history_label_root)
        LOGGER.info(f"Carrying over {len(history_records)} replay samples from {args.load_hist}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    written = _materialize_records(history_records, args.output_dir)
    written += _materialize_records(selected, args.output_dir)
    names = {**history_names, **dict(dataset_config["names"])}
    YAML().save(args.output_dir / "dataset.yaml", {"names": names, "train": "images/train"})
    LOGGER.info(f"Replay dataset with {written} samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
