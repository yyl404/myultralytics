"""Per-stage mAP aggregation of evaluation CSVs, read from the models' own history.

For every evaluation CSV ``model_<k>_eval_<tag>.csv`` under ``--eval_dir`` (per-task
and cumulative cells alike), aggregate its per-class rows into the class space of
each incremental stage the model went through. The stage class spaces come from the
checkpoint itself (the ``incremental_history`` module attribute stamped by
tools/train.py and tools/expand_model_head.py), never from the eval-time task
yamls, so the breakdown stays correct when the eval datasets differ from the
training ones in order, kind, or number.

Outputs (under <eval_dir>):
    model_<k>_eval_<tag>_stage_mAP.csv   one table per evaluation cell: rows =
                                         incremental stages, columns = NumClasses +
                                         mAP metrics
    stage_mAP_sequence.csv               all of the above with Model_Task and Eval
                                         columns prepended

Usage:
    $ python tools/stage_task_map.py \
        --run_dir runs/<run> --eval_dir runs/<run>/evaluation_results
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from ultralytics import YOLO

METRIC_ORDER = ("mAP50", "mAP75", "mAP50-95")

EVAL_CSV_PATTERN = re.compile(r"^model_(\d+)_eval_(.+)\.csv$")
EXCLUDED_SUFFIXES = ("_stage_mAP", "_confusion_matrix")


def _normalize_class_name(name: str) -> str:
    """Normalize a class name for exact cross-file matching."""
    return " ".join(name.strip().casefold().split())


def load_per_class_metrics(evaluation_csv: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load per-class mAP metrics and the available metric columns.

    An evaluation cell whose classes are disjoint from the model's class space has
    no per-class rows; it returns an empty mapping and the caller skips it.
    """
    if not evaluation_csv.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {evaluation_csv}")
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
    return class_metrics, metrics


def load_incremental_history(model_path: Path) -> list[list[str]]:
    """Return the per-stage class-name lists recorded in the checkpoint.

    Fails fast when the checkpoint predates history stamping or when the recorded
    stages do not exactly tile the model's current class space.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = YOLO(str(model_path))
    history = getattr(model.model, "incremental_history", None)
    if not history:
        raise ValueError(
            f"{model_path} carries no incremental_history; retrain it with the current "
            f"tools/train.py and tools/expand_model_head.py"
        )
    model_names = [model.model.names[i] for i in sorted(model.model.names)]
    history_names = [name for stage in history for name in stage["names"]]
    if history_names != model_names:
        raise ValueError(
            f"incremental_history of {model_path} does not match its class space: "
            f"history covers {history_names}, model names are {model_names}"
        )
    return [stage["names"] for stage in history]


def summarize_task_metrics(
    class_metrics: dict[str, dict[str, float]],
    task_class_names: list[list[str]],
    metrics: list[str],
) -> list[dict[str, object]]:
    """Compute macro-average metrics for each stage's class set.

    Stages with no classes in the evaluation CSV are not covered by the evaluated
    dataset; they are omitted with a warning. Classes absent from the CSV have no
    instances in the evaluated split and are excluded from the macro-average,
    mirroring how the validator computes mAP.
    """
    summaries = []
    seen_classes = set()
    for task_index, class_names in enumerate(task_class_names, start=1):
        normalized_names = [_normalize_class_name(name) for name in class_names]
        duplicates = seen_classes.intersection(normalized_names)
        if duplicates:
            raise ValueError(f"Classes appear in multiple tasks: {sorted(duplicates)}")
        present = [name for name in normalized_names if name in class_metrics]
        if not present:
            print(f"WARNING: stage {task_index} classes are absent from the evaluation CSV; stage skipped")
            seen_classes.update(normalized_names)
            continue
        missing = [name for name, normalized in zip(class_names, normalized_names) if normalized not in class_metrics]
        if missing:
            print(
                f"WARNING: stage {task_index} classes have no evaluation rows (no instances in the "
                f"evaluated split) and are excluded from the average: {missing}"
            )
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
    """Write per-stage mAP summaries."""
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
    """Build per-stage mAP tables for every evaluation cell under the eval dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory holding task-k/best.pt")
    parser.add_argument("--eval_dir", type=Path, required=True, help="Directory with model_k_eval_*.csv files")
    args = parser.parse_args()

    evaluation_csvs = []
    for csv_path in sorted(args.eval_dir.glob("model_*_eval_*.csv")):
        match = EVAL_CSV_PATTERN.match(csv_path.name)
        if match is None or match.group(2).endswith(EXCLUDED_SUFFIXES):
            continue
        evaluation_csvs.append((csv_path, int(match.group(1)), match.group(2)))
    if not evaluation_csvs:
        raise ValueError(f"No model_<k>_eval_*.csv files found under {args.eval_dir}")

    histories: dict[int, list[list[str]]] = {}
    sequence_rows: list[dict[str, object]] = []
    metrics: list[str] | None = None
    for csv_path, model_task, eval_tag in evaluation_csvs:
        class_metrics, cell_metrics = load_per_class_metrics(csv_path)
        if not class_metrics:
            print(f"WARNING: skipping {csv_path.name}: no per-class rows (eval dataset disjoint from the model)")
            continue
        if metrics is None:
            metrics = cell_metrics
        elif cell_metrics != metrics:
            raise ValueError(f"Metric columns of {csv_path} differ from earlier cells: {cell_metrics} vs {metrics}")
        if model_task not in histories:
            histories[model_task] = load_incremental_history(args.run_dir / f"task-{model_task}" / "best.pt")
        summaries = summarize_task_metrics(class_metrics, histories[model_task], cell_metrics)
        output_csv = args.eval_dir / f"{csv_path.stem}_stage_mAP.csv"
        write_task_metrics(output_csv, summaries, cell_metrics)
        print(f"Stage-wise mAP for {csv_path.name} saved to {output_csv}")
        for summary in summaries:
            sequence_rows.append({"Model_Task": f"Model_{model_task}", "Eval": eval_tag, **summary})

    if not sequence_rows:
        raise ValueError(f"No per-stage tables produced under {args.eval_dir}")

    metric_columns = [metric for metric in METRIC_ORDER if metric in metrics]
    sequence_csv = args.eval_dir / "stage_mAP_sequence.csv"
    fieldnames = ["Model_Task", "Eval", "Task", "NumClasses", *metric_columns]
    with sequence_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sequence_rows:
            writer.writerow({**row, **{metric: f"{float(row[metric]):.5f}" for metric in metric_columns}})
    print(f"Stage mAP sequence saved to {sequence_csv}")


if __name__ == "__main__":
    main()
