"""Per-stage mAP matrices on the cumulative evaluation, read from the models' own history.

For every task-k checkpoint (<run_dir>/task-k/best.pt), aggregate the per-class rows of the
cumulative evaluation (<eval_dir>/model_k_eval_cumulative.csv) into the class space of each
incremental stage the model went through. The stage class spaces come from the checkpoint
itself (the ``incremental_history`` module attribute stamped by tools/train.py and
tools/expand_model_head.py), so the breakdown stays correct even when the eval-time task
datasets differ from the training ones.

Outputs (under <eval_dir>):
    model_k_eval_cumulative_stage_mAP.csv  one matrix per model: rows = incremental stages,
                                           columns = NumClasses + mAP metrics
    cumulative_stage_mAP_sequence.csv      the sequence of those matrices with a Model_Task
                                           column prepended

Usage:
    $ python tools/cumulative_stage_map.py \
        --run_dir runs/<run> --eval_dir runs/<run>/evaluation_results --num_tasks 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from summarize_cumulative_task_map import (
    METRIC_ORDER,
    load_per_class_metrics,
    summarize_task_metrics,
    write_task_metrics,
)


def load_incremental_history(model_path: Path) -> list[list[str]]:
    """Return the per-stage class-name lists recorded in the checkpoint.

    Fails fast when the checkpoint predates history stamping or when the recorded stages do
    not exactly tile the model's current class space.
    """
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


def main() -> None:
    """Build per-stage mAP matrices for every task model on its cumulative evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory holding task-k/best.pt")
    parser.add_argument("--eval_dir", type=Path, required=True, help="Directory with model_k_eval_cumulative.csv files")
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of incremental tasks")
    args = parser.parse_args()

    sequence_rows: list[dict[str, object]] = []
    metrics: list[str] | None = None
    for model_task in range(1, args.num_tasks + 1):
        model_path = args.run_dir / f"task-{model_task}" / "best.pt"
        evaluation_csv = args.eval_dir / f"model_{model_task}_eval_cumulative.csv"
        if not model_path.is_file() or not evaluation_csv.is_file():
            print(f"WARNING: skipping model task {model_task}: missing {model_path} or {evaluation_csv}")
            continue
        stage_class_names = load_incremental_history(model_path)
        class_metrics, model_metrics = load_per_class_metrics(evaluation_csv)
        if metrics is None:
            metrics = model_metrics
        elif model_metrics != metrics:
            raise ValueError(f"Metric columns of {evaluation_csv} differ from earlier models: {model_metrics} vs {metrics}")
        summaries = summarize_task_metrics(class_metrics, stage_class_names, model_metrics)
        output_csv = args.eval_dir / f"model_{model_task}_eval_cumulative_stage_mAP.csv"
        write_task_metrics(output_csv, summaries, model_metrics)
        print(f"Stage-wise cumulative mAP for model task {model_task} saved to {output_csv}")
        for summary in summaries:
            sequence_rows.append({"Model_Task": f"Model_{model_task}", **summary})

    if not sequence_rows:
        raise ValueError(f"No per-stage matrices produced under {args.eval_dir}")

    sequence_csv = args.eval_dir / "cumulative_stage_mAP_sequence.csv"
    fieldnames = ["Model_Task", "Task", "NumClasses", *[m for m in METRIC_ORDER if m in metrics]]
    with sequence_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sequence_rows:
            writer.writerow({**row, **{m: f"{float(row[m]):.5f}" for m in fieldnames[3:]}})
    print(f"Stage mAP matrix sequence saved to {sequence_csv}")


if __name__ == "__main__":
    main()
