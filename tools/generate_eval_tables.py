#!/usr/bin/env python3
"""
Summarize evaluation results of incremental learning and form csv tables.

This script reads the per-class evaluation CSVs of an incremental learning
experiment and generates up to two matrix tables, each strictly shaped by the
actual number of models and the actual number of evaluation datasets:

1. Individual dataset evaluation table: mAP for each model on each eval dataset
2. Cumulative dataset evaluation table: mAP for each model on each cumulative dataset

Usage:
    $ python tools/generate_eval_tables.py \
        --eval_dir <path/to/evaluation_results> \
        --model_tasks 1 2 \
        --num_eval_tasks 2 \
        [--num_cumulative 2] \
        --output_dir <path/to/output_directory>

Arguments:
    --eval_dir: Directory containing evaluation CSV results. The script expects CSV
        files named 'model_{model_task}_eval_task_{dataset_task}.csv' for individual
        dataset evaluations and 'model_{model_task}_eval_cumulative_{n}.csv' for
        cumulative dataset evaluations.
    --model_tasks: Task ids of the evaluated models (the task-k/best.pt found
        under the run directory).
    --num_eval_tasks: Number of per-task evaluation datasets.
    --num_cumulative: Number of cumulative evaluation datasets (0 = skip the
        cumulative table).
    --output_dir: Directory where the generated evaluation tables will be saved.
        The script will create:
        - individual_datasets_eval.csv: Table with mAP values for each model on each dataset
        - cumulative_datasets_eval.csv: Table with mAP values for each model on cumulative datasets

Examples:
    $ python tools/generate_eval_tables.py \
        --eval_dir runs/<run>/evaluation_results \
        --model_tasks 1 2 \
        --num_eval_tasks 2 \
        --num_cumulative 2 \
        --output_dir runs/<run>/evaluation_results
"""

import argparse
import csv
import os

from ultralytics.utils import LOGGER


METRIC_ORDER = ("mAP50", "mAP75", "mAP50-95")


def extract_metrics_from_csv(csv_path):
    """Extract available mAP metrics, averaging per-class rows when no all row exists."""
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metric_columns = [metric for metric in METRIC_ORDER if metric in (reader.fieldnames or [])]
        values = {metric: [] for metric in metric_columns}
        for row in reader:
            if row.get('Class', '').lower() == 'all':
                return {
                    metric: float(row[metric])
                    for metric in metric_columns
                    if row.get(metric, '') != ''
                }
            for metric in metric_columns:
                raw_value = row.get(metric, '')
                if raw_value != '':
                    values[metric].append(float(raw_value))

    return {
        metric: sum(metric_values) / len(metric_values)
        for metric, metric_values in values.items()
        if metric_values
    }


def _available_metrics(results):
    """Return metric columns present in at least one result, in stable order."""
    present = set()
    for result in results:
        present.update(result)
    return [metric for metric in METRIC_ORDER if metric in present]


def generate_matrix_table(eval_results, column_prefix, output_path):
    """Write one models x datasets mAP matrix.

    eval_results: dict of {model_task: {dataset_task: {metric_name: value}}};
    missing cells (no evaluation CSV or no per-class rows) are written as N/A.
    """
    model_tasks = sorted(eval_results.keys())
    dataset_tasks = set()
    for model_results in eval_results.values():
        dataset_tasks.update(model_results.keys())
    dataset_tasks = sorted(dataset_tasks)

    metrics = _available_metrics(
        metrics for model_results in eval_results.values() for metrics in model_results.values()
    )
    fieldnames = ['Model_Task']
    for dataset_task in dataset_tasks:
        for metric in metrics:
            fieldnames.append(f'{column_prefix}_{dataset_task}_{metric}')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_task in model_tasks:
            row = {'Model_Task': f'Model_{model_task}'}
            for dataset_task in dataset_tasks:
                if dataset_task not in eval_results[model_task]:
                    LOGGER.warning(f"No evaluation results found for Model Task {model_task} on Dataset Task {dataset_task}.")
                result = eval_results[model_task].get(dataset_task) or {}
                for metric in metrics:
                    value = result.get(metric)
                    row[f'{column_prefix}_{dataset_task}_{metric}'] = f'{value:.4f}' if value is not None else 'N/A'
            writer.writerow(row)

    print(f"Evaluation table saved to {output_path}")


def collect_matrix(eval_dir, model_tasks, num_datasets, name_template):
    """Read the per-class CSVs of one matrix into {model_task: {dataset_task: metrics}}."""
    results = {}
    for model_task in model_tasks:
        results[model_task] = {}
        for dataset_task in range(1, num_datasets + 1):
            csv_path = os.path.join(eval_dir, name_template.format(model=model_task, dataset=dataset_task))
            results[model_task][dataset_task] = extract_metrics_from_csv(csv_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tables from CSV results")
    parser.add_argument("--eval_dir", type=str, required=True,
                       help="Directory containing evaluation results")
    parser.add_argument("--model_tasks", type=int, nargs="+", required=True,
                       help="Task ids of the evaluated models")
    parser.add_argument("--num_eval_tasks", type=int, required=True,
                       help="Number of per-task evaluation datasets")
    parser.add_argument("--num_cumulative", type=int, default=0,
                       help="Number of cumulative evaluation datasets (0 = skip cumulative table)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    individual_results = collect_matrix(
        args.eval_dir, args.model_tasks, args.num_eval_tasks,
        "model_{model}_eval_task_{dataset}.csv",
    )
    generate_matrix_table(
        individual_results, "Task",
        os.path.join(args.output_dir, "individual_datasets_eval.csv"),
    )

    if args.num_cumulative > 0:
        cumulative_results = collect_matrix(
            args.eval_dir, args.model_tasks, args.num_cumulative,
            "model_{model}_eval_cumulative_{dataset}.csv",
        )
        generate_matrix_table(
            cumulative_results, "Cumulative",
            os.path.join(args.output_dir, "cumulative_datasets_eval.csv"),
        )

    print("Evaluation tables generated successfully!")


if __name__ == "__main__":
    main()
