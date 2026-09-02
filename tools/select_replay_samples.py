"""Select replay samples from a finished task's training set for experience replay.

After each incremental task, a fixed number of training samples (images + ground-truth
labels, in the finished task's class-id space) is kept as replay data for the next tasks.

Class-id space invariant: head expansion keeps existing class ids and order, and appends
unseen classes after them (tools/expand_model_head.py). Each task's replay_dataset is kept
uniformly in that task's class-id space, matching its dataset.yaml names: carried-over
history labels are remapped by class name on carry-over, and the next task's prepare step
remaps the whole replay dataset again into the expanded model's class-id space with
tools/convert_dataset_class_ids.py (same conversion as the task training data).

Selection is pluggable: REPLAY_STRATEGIES maps a strategy name to a callable

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

from utils import convert_class_ids, normalize_names


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


def _build_history_id_map(history_names: dict[int, str], current_names: dict[int, str]) -> dict[int, int]:
    """Map history replay label ids into the finished task's class-id space by class name."""
    current_ids_by_name = {}
    for class_id, class_name in current_names.items():
        current_ids_by_name.setdefault(class_name, class_id)
    id_map = {}
    missing = []
    for history_id, class_name in history_names.items():
        current_id = current_ids_by_name.get(class_name)
        if current_id is None:
            missing.append(class_name)
        else:
            id_map[history_id] = current_id
    if missing:
        raise KeyError(f"History replay classes missing from the finished task's class space: {missing}")
    return id_map


def _materialize_records(
    records: list[ReplayRecord],
    output_dir: Path,
    split: str = "train",
    class_id_map: dict[int, int] | None = None,
) -> int:
    """Link images and write labels of the selected records into the replay dataset directory.

    Labels are copied verbatim, unless class_id_map is given (history carry-over), in which
    case class ids are rewritten through the map.
    """
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
            destination_label = label_dir / record.label_path.name
            if class_id_map is None:
                shutil.copy2(record.label_path, destination_label)
            else:
                lines = record.label_path.read_text(encoding="utf-8").splitlines(keepends=True)
                destination_label.write_text("".join(convert_class_ids(lines, class_id_map)), encoding="utf-8")
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Finished task dataset YAML (labels in that task's class-id space)")
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
    current_names = normalize_names(dataset_config["names"], source=f"dataset '{args.dataset}'")

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

    history_records: list[ReplayRecord] = []
    history_id_map = None
    if args.load_hist is not None:
        history_config = YAML().load(args.load_hist / "dataset.yaml")
        if "names" not in history_config:
            raise KeyError(f"Dataset config has no 'names': {args.load_hist / 'dataset.yaml'}")
        history_names = normalize_names(
            history_config["names"], source=f"replay history '{args.load_hist / 'dataset.yaml'}'"
        )
        # History labels live in the previous task's class-id space; remap them by class name
        # into this task's space so the merged replay dataset stays uniformly labeled.
        history_id_map = _build_history_id_map(history_names, current_names)
        history_image_root, history_label_root = _resolve_split_dirs(
            args.load_hist / "dataset.yaml", history_config, split="train"
        )
        history_records = _collect_records(history_image_root, history_label_root)
        LOGGER.info(f"Carrying over {len(history_records)} replay samples from {args.load_hist}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    written = _materialize_records(history_records, args.output_dir, class_id_map=history_id_map)
    written += _materialize_records(selected, args.output_dir)
    YAML().save(args.output_dir / "dataset.yaml", {"names": current_names, "train": "images/train"})
    LOGGER.info(f"Replay dataset with {written} samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
