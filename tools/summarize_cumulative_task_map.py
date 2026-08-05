"""Summarize per-task mAP from the final cumulative per-class evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


METRIC_ORDER = ("mAP50", "mAP75", "mAP50-95")


def _normalize_class_name(name: str) -> str:
    """Normalize a class name for exact cross-file matching."""
    return " ".join(name.strip().casefold().split())


def load_task_class_names(dataset_yaml: Path) -> list[str]:
    """Load the ordered class names from one task dataset YAML."""
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Task dataset YAML not found: {dataset_yaml}")
    with dataset_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"Dataset YAML must contain a names mapping or list: {dataset_yaml}")
    names = data["names"]
    if isinstance(names, list):
        ordered_names = names
    elif isinstance(names, dict):
        indexed_names = {int(index): name for index, name in names.items()}
        expected_indices = list(range(len(indexed_names)))
        if sorted(indexed_names) != expected_indices:
            raise ValueError(
                f"Class IDs in {dataset_yaml} must be contiguous from 0, got {sorted(indexed_names)}"
            )
        ordered_names = [indexed_names[index] for index in expected_indices]
    else:
        raise TypeError(f"names in {dataset_yaml} must be a list or mapping, got {type(names)}")
    if not ordered_names or not all(isinstance(name, str) and name.strip() for name in ordered_names):
        raise ValueError(f"Dataset YAML contains invalid or empty class names: {dataset_yaml}")
    return ordered_names


def load_per_class_metrics(evaluation_csv: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load per-class mAP metrics and the available metric columns."""
    if not evaluation_csv.is_file():
        raise FileNotFoundError(f"Final cumulative evaluation CSV not found: {evaluation_csv}")
    with evaluation_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if "Class" not in fieldnames:
            raise ValueError(f"Evaluation CSV has no Class column: {evaluation_csv}")
        metrics = [metric for metric in METRIC_ORDER if metric in fieldnames]
        if not metrics:
            raise ValueError(f"Evaluation CSV has no supported mAP columns: {evaluation_csv}")
        class_metrics = {}
        for row_number, row in enumerate(reader, start=2):
            class_name = row.get("Class", "")
            if not class_name or class_name.casefold() == "all":
                continue
            normalized_name = _normalize_class_name(class_name)
            if normalized_name in class_metrics:
                raise ValueError(f"Duplicate class '{class_name}' at CSV row {row_number}")
            values = {}
            for metric in metrics:
                raw_value = row.get(metric, "")
                if raw_value == "":
                    raise ValueError(f"Missing {metric} for class '{class_name}' at CSV row {row_number}")
                values[metric] = float(raw_value)
            class_metrics[normalized_name] = values
    if not class_metrics:
        raise ValueError(f"Evaluation CSV contains no per-class rows: {evaluation_csv}")
    return class_metrics, metrics


def summarize_task_metrics(
    class_metrics: dict[str, dict[str, float]],
    task_class_names: list[list[str]],
    metrics: list[str],
) -> list[dict[str, object]]:
    """Compute macro-average metrics for each task's class set.

    Classes absent from the evaluation CSV have no instances in the evaluated split; they are
    excluded from the macro-average with a warning, mirroring how the validator computes mAP.
    """
    summaries = []
    seen_classes = set()
    for task_index, class_names in enumerate(task_class_names, start=1):
        normalized_names = [_normalize_class_name(name) for name in class_names]
        duplicates = seen_classes.intersection(normalized_names)
        if duplicates:
            raise ValueError(f"Classes appear in multiple tasks: {sorted(duplicates)}")
        present = [name for name in normalized_names if name in class_metrics]
        missing = [name for name, normalized in zip(class_names, normalized_names) if normalized not in class_metrics]
        if missing:
            print(
                f"WARNING: Task {task_index} classes have no evaluation rows (no instances in the "
                f"evaluated split) and are excluded from the average: {missing}"
            )
        if not present:
            raise ValueError(f"Task {task_index} has no classes present in the cumulative CSV")
        seen_classes.update(normalized_names)
        row = {"Task": f"Task_{task_index}", "NumClasses": len(class_names)}
        for metric in metrics:
            row[metric] = sum(class_metrics[name][metric] for name in present) / len(present)
        summaries.append(row)
    all_row = {"Task": "All", "NumClasses": len(class_metrics)}
    for metric in metrics:
        all_row[metric] = sum(values[metric] for values in class_metrics.values()) / len(class_metrics)
    summaries.append(all_row)
    return summaries


def write_task_metrics(output_csv: Path, summaries: list[dict[str, object]], metrics: list[str]) -> None:
    """Write task-level mAP summaries."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Task", "NumClasses", *metrics])
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            for metric in metrics:
                row[metric] = f"{float(row[metric]):.5f}"
            writer.writerow(row)


def main() -> None:
    """Run task-level aggregation from a final cumulative evaluation CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_csv", type=Path, required=True, help="Final model cumulative per-class CSV")
    parser.add_argument(
        "--task_data",
        type=Path,
        nargs="+",
        required=True,
        help="Task-local dataset YAML files in task order",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output task-level mAP CSV")
    args = parser.parse_args()

    class_metrics, metrics = load_per_class_metrics(args.evaluation_csv)
    task_class_names = [load_task_class_names(path) for path in args.task_data]
    summaries = summarize_task_metrics(class_metrics, task_class_names, metrics)
    write_task_metrics(args.output, summaries, metrics)
    print(f"Task-level cumulative mAP saved to {args.output}")


if __name__ == "__main__":
    main()
