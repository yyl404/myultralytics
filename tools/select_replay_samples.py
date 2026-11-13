"""Select replay samples from a finished task's training set for experience replay.

After each incremental task, a fixed number of training samples (images + ground-truth
labels, in the finished task's class-id space) is kept as replay data for the next tasks.

Class-id space invariant: head expansion keeps existing class ids and order, and appends
unseen classes after them (tools/expand_model_head.py). Each task's replay_dataset is kept
uniformly in that task's class-id space, matching its dataset.yaml names: carried-over
history labels are remapped by class name on carry-over, and the next task's prepare step
remaps the whole replay dataset again into the expanded model's class-id space with
tools/convert_dataset_class_ids.py (same conversion as the task training data).

Selection is pluggable through two registries:

    REPLAY_STRATEGIES: strategy(records, num, rng) -> list[ReplayRecord]
        Per-task strategies: pick ``num`` samples of this task, the carried-over
        history is appended afterwards, so the memory grows with every task.
        The default "random" strategy lives here.

    POOL_STRATEGIES: pool_strategy(pool, capacity, context) -> list[ReplayRecord]
        Memory strategies: pick a ``capacity``-sized subset of the whole pool
        (carried-over history plus this task's samples), so the memory stays
        bounded. "ocdm" implements Optimizing Class Distribution in Memory.

For a pool strategy ``--num`` is the memory capacity, not the per-task sample count.

Usage:
    $ python tools/select_replay_samples.py \
        --dataset data/VOC_15+5/task_1_cls_15/dataset.yaml \
        --output_dir runs/<run>/task-1/replay_dataset \
        --num 100 --strategy random --seed 0 \
        [--load_hist runs/<run>/task-0/replay_dataset]

    $ python tools/select_replay_samples.py \
        --dataset data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --output_dir runs/<run>/task-2/replay_dataset \
        --num 800 --strategy ocdm --ocdm_stats model \
        --ocdm_model runs/<run>/task-2/best.pt --ocdm_device 0 \
        [--load_hist runs/<run>/task-1/replay_dataset]
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from utils import convert_class_ids, normalize_names

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


# Per-task strategies: sample this task, then append the carried-over history.
REPLAY_STRATEGIES = {
    "random": select_random,
}


def _read_class_set(label_path, class_id_map=None):
    """Return the class ids present in one YOLO label file (each class counted once)."""
    if label_path is None:
        return frozenset()
    class_ids = set()
    for line_number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_ids.add(int(parts[0]))
        except (IndexError, ValueError):
            raise ValueError(f"Invalid label line at '{label_path}:{line_number}': {line!r}") from None
    if class_id_map is not None:
        try:
            class_ids = {class_id_map[class_id] for class_id in class_ids}
        except KeyError as error:
            raise KeyError(
                f"Class id {error.args[0]} of '{label_path}' is missing from the history class map"
            ) from None
    return frozenset(class_ids)


def _gt_class_sets(records, class_id_map=None):
    """Per-record GT class set, used as the OCDM statistics when no model is given."""
    return [_read_class_set(record.label_path, class_id_map) for record in records]


def _model_class_sets(records, model_path, names, confidence, imgsz, device, batch):
    """Per-record predicted class set of a checkpoint, the reference OCDM statistics.

    Args:
        records (list[ReplayRecord]): Candidate pool, in pool order.
        model_path (Path): Checkpoint that annotates the pool (the finished task's model).
        names (dict[int, str]): Class-id space of the replay dataset.
        confidence (float): Score threshold of the retained predictions.
        imgsz (int): Inference image size.
        device (str): Inference device.
        batch (int): Inference batch size.

    Returns:
        list[frozenset[int]]: Predicted class ids per record, in pool order.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names = {index: name for index, name in model.names.items()}
    id_by_name = {name: index for index, name in names.items()}
    unknown = sorted(set(model_names.values()) - set(id_by_name))
    if unknown:
        raise KeyError(f"Model classes missing from the replay class space: {unknown}")

    paths = [str(record.image_path) for record in records]
    predicted_sets = []
    for result in model.predict(
        source=paths, stream=True, conf=confidence, imgsz=imgsz, device=device, batch=batch, verbose=False
    ):
        predicted = set()
        if result.boxes is not None and len(result.boxes):
            for class_id in result.boxes.cls.tolist():
                predicted.add(id_by_name[model_names[int(class_id)]])
        predicted_sets.append(frozenset(predicted))
    if len(predicted_sets) != len(records):
        raise RuntimeError(f"Prediction pass returned {len(predicted_sets)} results for {len(records)} pool images")
    LOGGER.info(f"Annotated {len(records)} pool images with '{model_path}' at conf {confidence}")
    return predicted_sets


def _greedy_removal(class_sets, capacity, num_classes):
    """Return the positions to drop so the remaining class distribution is closest to uniform.

    Greedy OCDM: repeatedly drop the sample whose removal minimizes the cross
    entropy between the uniform target distribution and the class distribution of
    what remains. A class is counted once per image, as in multi-label detection.

    Args:
        class_sets (list[frozenset[int]]): Class ids present in each candidate sample.
        capacity (int): Number of samples to keep.
        num_classes (int): Size of the class-id space.

    Returns:
        list[int]: Positions (into ``class_sets``) to remove.
    """
    pool_size = len(class_sets)
    if pool_size <= capacity:
        return []
    counts = np.zeros((pool_size, num_classes), dtype=np.float64)
    for row, class_set in enumerate(class_sets):
        counts[row, list(class_set)] = 1.0
    totals = counts.sum(axis=0)
    target = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    removed = np.zeros(pool_size, dtype=bool)
    for _ in range(pool_size - capacity):
        remaining = totals - counts  # (pool_size, num_classes) distribution without each sample
        mass = remaining.sum(axis=1, keepdims=True)
        distribution = np.divide(remaining, mass, out=np.zeros_like(remaining), where=mass > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_entropy = -np.sum(target * np.log(np.maximum(distribution, 1e-12)), axis=1)
        cross_entropy[removed] = np.inf
        index = int(np.argmin(cross_entropy))
        removed[index] = True
        totals = remaining[index]
    return np.flatnonzero(removed).tolist()


def select_ocdm(pool, capacity, context):
    """Pick a ``capacity``-sized subset of the memory pool with OCDM.

    The pool (carried-over history plus this task's samples) is consumed in
    batches, as in the reference implementation: every batch is merged with the
    memory kept so far and the samples whose removal brings the class
    distribution closest to uniform are dropped until the capacity fits.

    Args:
        pool (list[ReplayRecord]): Memory pool, history samples first.
        capacity (int): Memory capacity (``--num``).
        context (dict): ``class_sets`` (per-pool-record class ids), ``num_classes``
            and ``batch_size``.

    Returns:
        list[ReplayRecord]: The selected replay samples.
    """
    class_sets = context["class_sets"]
    num_classes = context["num_classes"]
    batch_size = context["batch_size"]
    if len(class_sets) != len(pool):
        raise RuntimeError(f"OCDM got {len(class_sets)} class sets for {len(pool)} pool records")
    if capacity < 1:
        raise ValueError(f"OCDM capacity must be at least 1, got {capacity}")
    if batch_size < 1:
        raise ValueError(f"OCDM batch size must be at least 1, got {batch_size}")
    kept = []
    cursor = 0
    while cursor < len(pool):
        batch_end = min(cursor + batch_size, len(pool))
        candidate = kept + list(range(cursor, batch_end))
        cursor = batch_end
        if len(candidate) <= capacity:
            kept = candidate
            continue
        dropped = set(_greedy_removal([class_sets[index] for index in candidate], capacity, num_classes))
        kept = [index for position, index in enumerate(candidate) if position not in dropped]
    LOGGER.info(f"OCDM kept {len(kept)}/{len(pool)} pool samples with capacity {capacity}")
    return [pool[index] for index in kept]


# Pool strategies keep the memory bounded over the history + current-task pool.
POOL_STRATEGIES = {
    "ocdm": select_ocdm,
}

OCDM_STATS = ("model", "gt")


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
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Finished task dataset YAML (labels in that task's class-id space)"
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Replay dataset directory to create")
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="Replay samples to keep from this task (per-task strategies) or the memory capacity "
        "(pool strategies such as ocdm)",
    )
    parser.add_argument(
        "--strategy", type=str, default="random", choices=sorted({*REPLAY_STRATEGIES, *POOL_STRATEGIES})
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--load_hist",
        type=Path,
        default=None,
        help="Previous task's replay dataset directory; its samples are carried over (cumulative replay) "
        "and form the memory half of a pool strategy",
    )
    parser.add_argument(
        "--ocdm_stats",
        type=str,
        default="model",
        choices=OCDM_STATS,
        help="Class statistics of the ocdm strategy: model predictions (reference) or GT labels",
    )
    parser.add_argument(
        "--ocdm_model", type=Path, default=None, help="Checkpoint annotating the pool for --ocdm_stats model"
    )
    parser.add_argument("--ocdm_conf", type=float, default=0.5, help="Prediction score threshold of the statistics")
    parser.add_argument("--ocdm_imgsz", type=int, default=640, help="Prediction image size of the statistics")
    parser.add_argument("--ocdm_device", type=str, default="0", help="Prediction device of the statistics")
    parser.add_argument("--ocdm_infer_batch", type=int, default=16, help="Prediction batch size of the statistics")
    parser.add_argument(
        "--ocdm_batch_size",
        type=int,
        default=1000,
        help="Memory pool batch size of the greedy ocdm update",
    )
    args = parser.parse_args()
    if args.num < 1:
        parser.error("--num must be at least 1")
    if args.strategy == "ocdm" and args.ocdm_stats == "model" and args.ocdm_model is None:
        parser.error("--ocdm_model is required when --strategy ocdm uses --ocdm_stats model")
    if args.ocdm_batch_size < 1:
        parser.error("--ocdm_batch_size must be at least 1")
    return args


def _load_history(args, current_names):
    """Load the carried-over replay dataset and its history-to-current class id map."""
    if args.load_hist is None:
        return [], None
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
    return history_records, history_id_map


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

    history_records, history_id_map = _load_history(args, current_names)
    history_selection: list[ReplayRecord] = []
    current_selection: list[ReplayRecord] = []

    if args.strategy in POOL_STRATEGIES:
        pool = history_records + records
        if args.ocdm_stats == "model":
            class_sets = _model_class_sets(
                records=pool,
                model_path=args.ocdm_model,
                names=current_names,
                confidence=args.ocdm_conf,
                imgsz=args.ocdm_imgsz,
                device=args.ocdm_device,
                batch=args.ocdm_infer_batch,
            )
        else:
            class_sets = _gt_class_sets(history_records, class_id_map=history_id_map) + _gt_class_sets(records)
        selected = POOL_STRATEGIES[args.strategy](
            pool,
            args.num,
            {
                "class_sets": class_sets,
                "num_classes": len(current_names),
                "batch_size": args.ocdm_batch_size,
            },
        )
        history_pool = set(history_records)
        history_selection = [record for record in selected if record in history_pool]
        current_selection = [record for record in selected if record not in history_pool]
    else:
        rng = random.Random(args.seed)
        num = min(args.num, len(records))
        if num < args.num:
            LOGGER.warning(f"Requested {args.num} replay samples but only {len(records)} candidates exist; keeping all")
        current_selection = REPLAY_STRATEGIES[args.strategy](records, num, rng)
        history_selection = history_records
        LOGGER.info(f"Selected {len(current_selection)}/{len(records)} replay samples with strategy '{args.strategy}'")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    written = _materialize_records(history_selection, args.output_dir, class_id_map=history_id_map)
    written += _materialize_records(current_selection, args.output_dir)
    YAML().save(args.output_dir / "dataset.yaml", {"names": current_names, "train": "images/train"})
    LOGGER.info(f"Replay dataset with {written} samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
