"""Align dataset class IDs with a model output space using parallel file I/O."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER, YAML

from utils import convert_class_ids, normalize_names


_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class LabelJob:
    """One label conversion with mirrored relative path."""

    source: Path
    destination: Path


@dataclass(frozen=True)
class ImageJob:
    """One image link or copy with mirrored relative path."""

    source: Path
    destination: Path


def _bounded_map(
    executor: ThreadPoolExecutor,
    function: Callable[[_T], _R],
    items: Iterable[_T],
    max_pending: int,
) -> Iterator[_R]:
    """Map in input order while bounding queued Future objects."""
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


def _build_class_mapping(
    source_names: Mapping[int, str],
    model_names: Mapping[int, str],
    keep_unrecognized: bool,
) -> tuple[dict[int, int], dict[int, str]]:
    """Map source class IDs into the model output space."""
    output_names = dict(model_names)
    model_ids_by_name = {}
    for class_id, class_name in model_names.items():
        model_ids_by_name.setdefault(class_name, class_id)

    class_id_map = {}
    for source_id, class_name in source_names.items():
        model_id = model_ids_by_name.get(class_name)
        if model_id is not None:
            class_id_map[source_id] = model_id
        elif keep_unrecognized:
            new_id = len(output_names)
            class_id_map[source_id] = new_id
            output_names[new_id] = class_name
            model_ids_by_name[class_name] = new_id
        else:
            LOGGER.warning(f"Class '{class_name}' was not found in model classes and will be skipped")
    return class_id_map, output_names


def _resolve_split_roots(
    dataset_path: Path,
    dataset_config: Mapping,
    split: str,
) -> tuple[list[Path], list[Path]]:
    """Resolve image and corresponding label roots for one split."""
    split_value = dataset_config[split]
    split_entries = [split_value] if isinstance(split_value, str) else split_value
    if not isinstance(split_entries, Sequence) or not split_entries:
        raise TypeError(f"Dataset split '{split}' must be a path or non-empty list of paths")

    configured_root = dataset_config.get("path")
    if configured_root is None:
        dataset_root = dataset_path.parent
    else:
        dataset_root = Path(configured_root)
        if not dataset_root.is_absolute():
            dataset_root = dataset_path.parent / dataset_root

    image_roots = []
    label_roots = []
    for entry in split_entries:
        if not isinstance(entry, str):
            raise TypeError(f"Dataset split '{split}' contains a non-path entry: {entry!r}")
        image_root = Path(entry)
        if not image_root.is_absolute():
            image_root = dataset_root / image_root
        image_root = image_root.resolve()
        label_root = Path(str(image_root).replace("images", "labels"))
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image directory for split '{split}' does not exist: {image_root}")
        if not label_root.is_dir():
            raise FileNotFoundError(f"Label directory for split '{split}' does not exist: {label_root}")
        image_roots.append(image_root)
        label_roots.append(label_root)
    return image_roots, label_roots


def _collect_mirrored_jobs(
    source_roots: Sequence[Path],
    destination_root: Path,
    *,
    suffix: str | None = None,
) -> list[tuple[Path, Path]]:
    """Enumerate source files once and validate destination uniqueness."""
    jobs = []
    destinations: dict[Path, Path] = {}
    for source_root in source_roots:
        for directory, _, file_names in os.walk(source_root):
            relative_directory = Path(directory).relative_to(source_root)
            for file_name in file_names:
                source = Path(directory) / file_name
                if suffix is not None and source.suffix.lower() != suffix:
                    continue
                destination = destination_root / relative_directory / file_name
                previous = destinations.setdefault(destination, source)
                if previous != source:
                    raise ValueError(
                        f"Multiple source files map to '{destination}': '{previous}' and '{source}'"
                    )
                jobs.append((source, destination))
    return jobs


def _convert_label(job: LabelJob, class_id_map: Mapping[int, int], task: str) -> None:
    """Convert and write one YOLO label file."""
    try:
        lines = job.source.read_text(encoding="utf-8").splitlines(keepends=True)
        converted_lines = convert_class_ids(lines, class_id_map, task=task)
        job.destination.write_text("".join(converted_lines), encoding="utf-8")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to convert label '{job.source}': {error}") from error


def _mirror_image(job: ImageJob, no_use_link: bool) -> None:
    """Create one image symlink, or copy it when explicitly requested."""
    try:
        if no_use_link:
            shutil.copy2(job.source, job.destination)
        else:
            job.destination.symlink_to(job.source)
    except OSError as error:
        action = "copy" if no_use_link else "symlink"
        raise RuntimeError(
            f"Failed to {action} image '{job.source}' to '{job.destination}': {error}"
        ) from error


def _run_parallel(
    jobs: Sequence[_T],
    function: Callable[[_T], None],
    workers: int,
    description: str,
) -> None:
    """Execute independent I/O jobs in chunks to reduce Future overhead."""
    if not jobs:
        return
    for parent in {job.destination.parent for job in jobs}:
        parent.mkdir(parents=True, exist_ok=True)

    chunk_size = max(16, min(256, (len(jobs) + workers * 8 - 1) // (workers * 8)))
    chunks = [jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size)]

    def process_chunk(chunk: Sequence[_T]) -> int:
        for job in chunk:
            function(job)
        return len(chunk)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="class-id-io") as executor:
        results = _bounded_map(executor, process_chunk, chunks, max_pending=workers * 2)
        with tqdm(total=len(jobs), desc=description) as progress:
            for processed_count in results:
                progress.update(processed_count)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Expanded model checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="Source dataset YAML")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--keep_unrecognized_classes",
        action="store_true",
        help="Append classes absent from the model output space instead of dropping them",
    )
    parser.add_argument(
        "--no-use-link",
        action="store_true",
        help="Copy image files instead of creating per-file symbolic links",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel label/link I/O workers",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    """Convert requested splits and save an aligned dataset config."""
    args = parse_args()
    model = YOLO(args.model)
    task = getattr(model, "task", None) or "detect"
    model_names = normalize_names(model.names, source=f"model '{args.model}'")

    dataset_config = YAML().load(args.dataset)
    if "names" not in dataset_config:
        raise KeyError(f"Dataset config has no 'names': {args.dataset}")
    source_names = normalize_names(
        dataset_config["names"],
        source=f"dataset '{args.dataset}'",
    )
    class_id_map, output_names = _build_class_mapping(
        source_names=source_names,
        model_names=model_names,
        keep_unrecognized=args.keep_unrecognized_classes,
    )

    if args.output_dir.exists() or args.output_dir.is_symlink():
        if args.output_dir.is_symlink() or args.output_dir.is_file():
            args.output_dir.unlink()
        else:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    output_config = {"names": output_names}
    for split in args.splits:
        if split not in dataset_config:
            LOGGER.warning(f"Split '{split}' was not found in '{args.dataset}'; skipping")
            continue

        image_roots, label_roots = _resolve_split_roots(
            dataset_path=args.dataset,
            dataset_config=dataset_config,
            split=split,
        )
        label_pairs = _collect_mirrored_jobs(
            label_roots,
            args.output_dir / "labels" / split,
            suffix=".txt",
        )
        label_jobs = [LabelJob(source, destination) for source, destination in label_pairs]
        _run_parallel(
            jobs=label_jobs,
            function=lambda job: _convert_label(job, class_id_map, task),
            workers=args.workers,
            description=f"Converting labels for {split}",
        )

        image_pairs = _collect_mirrored_jobs(
            image_roots,
            args.output_dir / "images" / split,
        )
        image_jobs = [ImageJob(source, destination) for source, destination in image_pairs]
        _run_parallel(
            jobs=image_jobs,
            function=lambda job: _mirror_image(job, args.no_use_link),
            workers=args.workers,
            description=f"Linking images for {split}" if not args.no_use_link else f"Copying images for {split}",
        )
        output_config[split] = f"images/{split}"

    YAML().save(args.output_dir / "dataset.yaml", output_config)


if __name__ == "__main__":
    main()
