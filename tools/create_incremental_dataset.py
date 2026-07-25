"""Create class-incremental YOLO datasets with parallel file processing."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

from ultralytics.utils import LOGGER, YAML


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp")
SUPPORTED_LABEL_EXTENSIONS = (".txt",)
_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class LabelRecord:
    """Parsed source annotation and its matching image."""

    label_path: Path
    image_path: Path
    labels: tuple[tuple[str, ...], ...]
    class_ids: frozenset[int]
    invalid_lines: tuple[int, ...]


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing one independent annotation file."""

    label_path: Path
    record: LabelRecord | None
    error: str | None


@dataclass(frozen=True)
class OutputJob:
    """One image/label pair to materialize."""

    image_path: Path
    label_path: Path
    destination_image: Path
    destination_label: Path
    class_id_map: Mapping[int, int]


@dataclass(frozen=True)
class RecordPlan:
    """All output writes generated from one source annotation."""

    source_label: Path
    jobs: tuple[OutputJob, ...]


def _bounded_map(
    executor: ThreadPoolExecutor,
    function: Callable[[_T], _R],
    items: Iterable[_T],
    max_pending: int,
) -> Iterator[_R]:
    """Map in submission order without creating one Future per dataset item."""
    if max_pending < 1:
        raise ValueError(f"max_pending must be positive, got {max_pending}")

    item_iterator = iter(items)
    pending: list[Future[_R]] = []
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(item_iterator)))
        except StopIteration:
            break

    while pending:
        future = pending.pop(0)
        yield future.result()
        try:
            pending.append(executor.submit(function, next(item_iterator)))
        except StopIteration:
            pass


def _resolve_dataset_path(path: str, source_config: Path) -> Path:
    """Resolve one dataset path relative to its YAML file."""
    candidate = Path(path)
    if not candidate.exists():
        candidate = source_config.parent / candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {candidate}")
    return candidate


def _discover_files(directories: Sequence[Path], extensions: Sequence[str]) -> list[Path]:
    """Return files in the same extension-major order used by the legacy tool."""
    files = []
    for directory in directories:
        for extension in extensions:
            files.extend(directory.glob(f"*{extension.lower()}"))
            files.extend(directory.glob(f"*{extension.upper()}"))
    return files


def _build_image_index(image_files: Sequence[Path]) -> dict[str, Path]:
    """Index image stems for constant-time label-to-image matching."""
    image_index: dict[str, Path] = {}
    for image_path in image_files:
        previous = image_index.setdefault(image_path.stem, image_path)
        if previous != image_path:
            raise ValueError(
                f"Duplicate image stem '{image_path.stem}' maps to both '{previous}' and '{image_path}'"
            )
    return image_index


def _validate_unique_label_stems(label_files: Sequence[Path]) -> None:
    """Reject labels that would race while writing the same destination."""
    labels_by_stem: dict[str, Path] = {}
    for label_path in label_files:
        previous = labels_by_stem.setdefault(label_path.stem, label_path)
        if previous != label_path:
            raise ValueError(
                f"Duplicate label stem '{label_path.stem}' maps to both '{previous}' and '{label_path}'"
            )


def _parse_label(label_path: Path, image_index: Mapping[str, Path]) -> ParseResult:
    """Parse one YOLO label file without mutating shared state."""
    image_path = image_index.get(label_path.stem)
    if image_path is None:
        return ParseResult(label_path, None, "no corresponding image file")

    labels = []
    class_ids = set()
    invalid_lines = []
    try:
        with label_path.open(encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                parts = tuple(line.strip().split())
                if len(parts) < 5:
                    invalid_lines.append(line_number)
                    continue
                class_id = int(parts[0])
                labels.append(parts)
                class_ids.add(class_id)
    except (OSError, ValueError) as error:
        return ParseResult(label_path, None, f"{type(error).__name__}: {error}")

    return ParseResult(
        label_path=label_path,
        record=LabelRecord(
            label_path=label_path,
            image_path=image_path,
            labels=tuple(labels),
            class_ids=frozenset(class_ids),
            invalid_lines=tuple(invalid_lines),
        ),
        error=None,
    )


def _task_dir(output_dir: Path, task_id: int, num_classes: int) -> Path:
    return output_dir / f"task_{task_id + 1}_cls_{num_classes}"


def _cumulative_task_dir(output_dir: Path, task_id: int, num_classes: int) -> Path:
    return output_dir / f"task_1-{task_id + 1}_cls_{num_classes}"


def _make_job(
    record: LabelRecord,
    destination_dir: Path,
    split: str,
    class_id_map: Mapping[int, int],
) -> OutputJob:
    return OutputJob(
        image_path=record.image_path,
        label_path=record.label_path,
        destination_image=destination_dir / "images" / split / record.image_path.name,
        destination_label=destination_dir / "labels" / split / record.label_path.name,
        class_id_map=class_id_map,
    )


def _plan_record(
    record: LabelRecord,
    mode: str,
    split: str,
    output_dir: Path,
    task_class_maps: Sequence[Mapping[int, int]],
    cumulative_class_maps: Sequence[Mapping[int, int]],
    task_image_counts: list[int],
) -> RecordPlan:
    """Plan destinations while preserving sequential full-split balancing."""
    jobs = []
    if mode == "full-split":
        candidate_tasks = [
            task_id
            for task_id, class_map in enumerate(task_class_maps)
            if record.class_ids.intersection(class_map)
        ]
        if candidate_tasks:
            assigned_task = min(
                candidate_tasks,
                key=lambda task_id: task_image_counts[task_id] / len(task_class_maps[task_id]),
            )
            jobs.append(
                _make_job(
                    record,
                    _task_dir(output_dir, assigned_task, len(task_class_maps[assigned_task])),
                    split,
                    task_class_maps[assigned_task],
                )
            )
            task_image_counts[assigned_task] += 1

            for end_task in range(max(assigned_task, 1), len(cumulative_class_maps)):
                cumulative_map = cumulative_class_maps[end_task]
                if record.class_ids.intersection(cumulative_map):
                    jobs.append(
                        _make_job(
                            record,
                            _cumulative_task_dir(output_dir, end_task, len(cumulative_map)),
                            split,
                            cumulative_map,
                        )
                    )
    elif mode == "sample-filter":
        for task_id, class_map in enumerate(task_class_maps):
            if record.class_ids.intersection(class_map):
                jobs.append(
                    _make_job(
                        record,
                        _task_dir(output_dir, task_id, len(class_map)),
                        split,
                        class_map,
                    )
                )
                task_image_counts[task_id] += 1

        for end_task in range(1, len(cumulative_class_maps)):
            cumulative_map = cumulative_class_maps[end_task]
            if record.class_ids.intersection(cumulative_map):
                jobs.append(
                    _make_job(
                        record,
                        _cumulative_task_dir(output_dir, end_task, len(cumulative_map)),
                        split,
                        cumulative_map,
                    )
                )
    elif mode == "label-filter":
        for task_id, class_map in enumerate(task_class_maps):
            jobs.append(
                _make_job(
                    record,
                    _task_dir(output_dir, task_id, len(class_map)),
                    split,
                    class_map,
                )
            )
            task_image_counts[task_id] += 1

        for end_task in range(1, len(cumulative_class_maps)):
            cumulative_map = cumulative_class_maps[end_task]
            jobs.append(
                _make_job(
                    record,
                    _cumulative_task_dir(output_dir, end_task, len(cumulative_map)),
                    split,
                    cumulative_map,
                )
            )
    else:
        raise ValueError(f"Unsupported dataset splitting mode: {mode}")

    return RecordPlan(source_label=record.label_path, jobs=tuple(jobs))


def _render_labels(record: LabelRecord, class_id_map: Mapping[int, int]) -> str:
    """Filter labels and convert source IDs to task-local IDs."""
    output_lines = []
    for parts in record.labels:
        source_class_id = int(parts[0])
        if source_class_id in class_id_map:
            converted = (str(class_id_map[source_class_id]), *parts[1:])
            output_lines.append(" ".join(converted))
    return "".join(f"{line}\n" for line in output_lines)


def _materialize_plan(plan_and_record: tuple[RecordPlan, LabelRecord]) -> None:
    """Copy all outputs for one source sample."""
    plan, record = plan_and_record
    try:
        for job in plan.jobs:
            shutil.copy2(job.image_path, job.destination_image)
            job.destination_label.write_text(
                _render_labels(record, job.class_id_map),
                encoding="utf-8",
            )
    except OSError as error:
        raise RuntimeError(f"Failed to write outputs for '{plan.source_label}': {error}") from error


def _build_class_splits(
    source_classes: Mapping[int, str],
    classes_per_task: Sequence[int],
) -> tuple[list[dict[int, str]], list[dict[int, int]], list[dict[int, str]], list[dict[int, int]]]:
    """Build task-local and cumulative class dictionaries."""
    sorted_source_ids = sorted(source_classes)
    requested_class_count = sum(classes_per_task)
    if requested_class_count > len(sorted_source_ids):
        raise ValueError(
            f"n_classes sums to {requested_class_count}, but source dataset has only "
            f"{len(sorted_source_ids)} classes"
        )
    if requested_class_count < len(sorted_source_ids):
        LOGGER.warning(
            f"Using the first {requested_class_count} of {len(sorted_source_ids)} source classes"
        )
    if any(class_count < 1 for class_count in classes_per_task):
        raise ValueError(f"Every task must contain at least one class, got {list(classes_per_task)}")

    task_classes = []
    task_class_maps = []
    offset = 0
    for class_count in classes_per_task:
        source_ids = sorted_source_ids[offset : offset + class_count]
        task_classes.append({task_id: source_classes[source_id] for task_id, source_id in enumerate(source_ids)})
        task_class_maps.append({source_id: task_id for task_id, source_id in enumerate(source_ids)})
        offset += class_count

    cumulative_classes = []
    cumulative_class_maps = []
    cumulative_source_ids: list[int] = []
    for task_map in task_class_maps:
        cumulative_source_ids.extend(task_map)
        cumulative_classes.append(
            {
                cumulative_id: source_classes[source_id]
                for cumulative_id, source_id in enumerate(cumulative_source_ids)
            }
        )
        cumulative_class_maps.append(
            {source_id: cumulative_id for cumulative_id, source_id in enumerate(cumulative_source_ids)}
        )
    return task_classes, task_class_maps, cumulative_classes, cumulative_class_maps


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create a clean output directory, prompting unless overwrite is enabled."""
    if output_dir.exists():
        remove_existing = overwrite
        if not overwrite:
            LOGGER.info(f"Output directory {output_dir} already exists, remove it? (Yes/No/Cancel)")
            answer = input().strip().lower()
            if answer in {"cancel", "c"}:
                LOGGER.info("Aborting...")
                raise SystemExit(1)
            remove_existing = answer in {"yes", "y"}
        if remove_existing:
            shutil.rmtree(output_dir)
            LOGGER.info(f"Output directory {output_dir} removed.")
    output_dir.mkdir(parents=True, exist_ok=True)


def _create_task_directories(
    output_dir: Path,
    splits: Sequence[str],
    task_classes: Sequence[Mapping[int, str]],
    cumulative_classes: Sequence[Mapping[int, str]],
) -> None:
    for task_id, classes in enumerate(task_classes):
        for split in splits:
            (_task_dir(output_dir, task_id, len(classes)) / "images" / split).mkdir(parents=True, exist_ok=True)
            (_task_dir(output_dir, task_id, len(classes)) / "labels" / split).mkdir(parents=True, exist_ok=True)
            if task_id > 0:
                cumulative_dir = _cumulative_task_dir(
                    output_dir, task_id, len(cumulative_classes[task_id])
                )
                (cumulative_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (cumulative_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _process_split(
    split: str,
    source_config: Path,
    split_config: str | Sequence[str],
    output_dir: Path,
    mode: str,
    task_class_maps: Sequence[Mapping[int, int]],
    cumulative_class_maps: Sequence[Mapping[int, int]],
    task_image_counts: list[int],
    workers: int,
) -> None:
    """Parse, plan, and materialize one dataset split."""
    image_entries = [split_config] if isinstance(split_config, str) else list(split_config)
    if not image_entries or not all(isinstance(entry, str) for entry in image_entries):
        raise TypeError(f"Split '{split}' must be a path or list of paths, got {split_config!r}")

    image_dirs = [_resolve_dataset_path(entry, source_config) for entry in image_entries]
    label_dirs = [
        _resolve_dataset_path(entry.replace("images", "labels"), source_config)
        for entry in image_entries
    ]
    image_files = _discover_files(image_dirs, SUPPORTED_IMAGE_EXTENSIONS)
    label_files = _discover_files(label_dirs, SUPPORTED_LABEL_EXTENSIONS)
    _validate_unique_label_stems(label_files)
    image_index = _build_image_index(image_files)

    parser = lambda label_path: _parse_label(label_path, image_index)
    max_pending = workers * 4
    pending_writes: list[Future[None]] = []
    with (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dataset-io") as executor,
        tqdm(total=len(label_files), desc=f"Processing {split} split") as progress,
    ):
        parse_results = _bounded_map(executor, parser, label_files, max_pending=workers * 4)
        for result in parse_results:
            if result.error is not None:
                LOGGER.warning(f"Skipping label '{result.label_path}': {result.error}")
                progress.update()
                continue
            record = result.record
            if record is None:
                raise RuntimeError(f"Label parser returned no record or error for '{result.label_path}'")
            if record.invalid_lines:
                LOGGER.warning(
                    f"Invalid YOLO labels in '{record.label_path}' at lines {list(record.invalid_lines)}; "
                    "those lines were skipped"
                )
            plan = _plan_record(
                record=record,
                mode=mode,
                split=split,
                output_dir=output_dir,
                task_class_maps=task_class_maps,
                cumulative_class_maps=cumulative_class_maps,
                task_image_counts=task_image_counts,
            )
            pending_writes.append(executor.submit(_materialize_plan, (plan, record)))
            if len(pending_writes) >= max_pending:
                pending_writes.pop(0).result()
                progress.update()

        for future in pending_writes:
            future.result()
            progress.update()


def _save_configs(
    output_dir: Path,
    splits: Sequence[str],
    task_classes: Sequence[Mapping[int, str]],
    cumulative_classes: Sequence[Mapping[int, str]],
) -> None:
    for task_id, classes in enumerate(task_classes):
        task_dir = _task_dir(output_dir, task_id, len(classes))
        task_config = {"names": dict(classes), **{split: f"images/{split}" for split in splits}}
        YAML().save(task_dir / "dataset.yaml", task_config)

        if task_id > 0:
            cumulative = cumulative_classes[task_id]
            cumulative_dir = _cumulative_task_dir(output_dir, task_id, len(cumulative))
            cumulative_config = {
                "names": dict(cumulative),
                **{split: f"images/{split}" for split in splits},
            }
            YAML().save(cumulative_dir / "dataset.yaml", cumulative_config)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cfg", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_classes", type=int, nargs="+", required=True)
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--mode",
        choices=["full-split", "sample-filter", "label-filter"],
        default="sample-filter",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of I/O worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory without prompting.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    """Create task-specific and cumulative incremental datasets."""
    args = parse_args()
    source_config = args.source_cfg
    source_dataset = YAML().load(source_config)
    if "names" not in source_dataset:
        raise KeyError(f"Source dataset config has no 'names': {source_config}")
    source_classes = source_dataset["names"]
    if isinstance(source_classes, list):
        source_classes = dict(enumerate(source_classes))
    if not isinstance(source_classes, dict):
        raise TypeError(f"Dataset 'names' must be a list or dict, got {type(source_classes)}")
    source_classes = {int(class_id): class_name for class_id, class_name in source_classes.items()}

    splits = []
    for split in args.split:
        if split in source_dataset:
            splits.append(split)
        else:
            LOGGER.warning(f"Source dataset config '{source_config}' has no '{split}' split; skipping")
    if not splits:
        raise ValueError(f"None of the requested splits exist in '{source_config}'")

    task_classes, task_maps, cumulative_classes, cumulative_maps = _build_class_splits(
        source_classes=source_classes,
        classes_per_task=args.n_classes,
    )
    _prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    _create_task_directories(
        output_dir=args.output_dir,
        splits=splits,
        task_classes=task_classes,
        cumulative_classes=cumulative_classes,
    )

    image_counts = {task_id: {split: 0 for split in splits} for task_id in range(len(task_classes))}
    for split in splits:
        split_counts = [image_counts[task_id][split] for task_id in range(len(task_classes))]
        _process_split(
            split=split,
            source_config=source_config,
            split_config=source_dataset[split],
            output_dir=args.output_dir,
            mode=args.mode,
            task_class_maps=task_maps,
            cumulative_class_maps=cumulative_maps,
            task_image_counts=split_counts,
            workers=args.workers,
        )
        for task_id, count in enumerate(split_counts):
            image_counts[task_id][split] = count

    _save_configs(
        output_dir=args.output_dir,
        splits=splits,
        task_classes=task_classes,
        cumulative_classes=cumulative_classes,
    )
    for task_id, classes in enumerate(task_classes):
        task_dir = _task_dir(args.output_dir, task_id, len(classes))
        LOGGER.info(f"Task {task_id + 1} completed: {len(classes)} classes")
        for split in splits:
            LOGGER.info(f"  {split}: {image_counts[task_id][split]} images")
        LOGGER.info(f"  Task config saved to: {task_dir / 'dataset.yaml'}")
        if task_id > 0:
            cumulative_dir = _cumulative_task_dir(
                args.output_dir, task_id, len(cumulative_classes[task_id])
            )
            LOGGER.info(f"  Cumulative config saved to: {cumulative_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Dataset creation interrupted")
        sys.exit(130)
