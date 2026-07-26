#!/usr/bin/env python3
"""
Summarize evaluation results of incremental learning and form a csv table.

This script reads all evaluation CSV format results of an incremental learning experiment and generates two tables:
1. Individual dataset evaluation table: mAP for each model on each dataset
2. Cumulative dataset evaluation table: mAP for each model on cumulative datasets

Usage:
    $ python tools/generate_eval_tables.py \
        --eval_dir <path/to/evaluation_results> \
        --num_tasks <number_of_tasks> \
        --output_dir <path/to/output_directory> \
        [--zero_shot]

Arguments:
    --eval_dir: Directory containing evaluation CSV results. The script expects CSV files
        named as 'model_{model_task}_eval_task_{dataset_task}.csv' for individual
        dataset evaluations and 'model_{model_task}_eval_cumulative.csv' for cumulative
        dataset evaluations.
    --num_tasks: Number of tasks in the incremental learning setup. This determines
        how many model tasks to process (from task 1 to num_tasks). For zero-shot,
        this is the number of datasets to evaluate.
    --output_dir: Directory where the generated evaluation tables will be saved.
        The script will create two CSV files:
        - individual_datasets_eval.csv: Table with mAP values for each model on each dataset
        - cumulative_datasets_eval.csv: Table with mAP values for each model on cumulative datasets
    --zero_shot: Flag to indicate zero-shot evaluation scenario. In zero-shot mode,
        only model_1 is processed, and all datasets (from 1 to num_tasks) are evaluated.

Examples:
    $ python tools/generate_eval_tables.py \
        --eval_dir runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label/evaluation_results \
        --num_tasks 2 \
        --output_dir runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label/evaluation_results
    
    $ python tools/generate_eval_tables.py \
        --eval_dir ./eval_results \
        --num_tasks 3 \
        --output_dir ./eval_tables
    
    $ python tools/generate_eval_tables.py \
        --eval_dir runs/yoloev8l_4-domain_zero-shot/evaluation_results \
        --num_tasks 4 \
        --output_dir runs/yoloev8l_4-domain_zero-shot/evaluation_results \
        --zero_shot
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


def generate_individual_table(eval_results, output_path):
    """
    Generate table for individual dataset evaluations.
    
    eval_results: dict of {model_task: {dataset_task: {metric_name: value}}}
    """
    # Get all unique model tasks and dataset tasks
    model_tasks = sorted(eval_results.keys())
    dataset_tasks = set()
    for model_results in eval_results.values():
        dataset_tasks.update(model_results.keys())
    dataset_tasks = sorted(dataset_tasks)
    
    # Build fieldnames
    metrics = _available_metrics(
        metrics for model_results in eval_results.values() for metrics in model_results.values()
    )
    fieldnames = ['Model_Task']
    for dataset_task in dataset_tasks:
        for metric in metrics:
            fieldnames.append(f'Task_{dataset_task}_{metric}')
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_task in model_tasks:
            row = {'Model_Task': f'Task_{model_task}'}
            for dataset_task in dataset_tasks:
                if dataset_task in eval_results[model_task]:
                    result = eval_results[model_task][dataset_task]
                    if result:
                        for metric in metrics:
                            value = result.get(metric)
                            row[f'Task_{dataset_task}_{metric}'] = f'{value:.4f}' if value is not None else 'N/A'
                    else:
                        LOGGER.warning(f"Failed to extract mAP values for Model Task {model_task} on Dataset Task {dataset_task}. CSV file may be empty or invalid.")
                        for metric in metrics:
                            row[f'Task_{dataset_task}_{metric}'] = 'N/A'
                else:
                    LOGGER.warning(f"No evaluation results found for Model Task {model_task} on Dataset Task {dataset_task}.")
                    for metric in metrics:
                        row[f'Task_{dataset_task}_{metric}'] = 'N/A'
            writer.writerow(row)
    
    print(f"Individual dataset evaluation table saved to {output_path}")


def generate_cumulative_table(cumulative_results, output_path):
    """
    Generate table for cumulative dataset evaluations.
    
    cumulative_results: dict of {model_task: {metric_name: value}}
    """
    metrics = _available_metrics(cumulative_results.values())
    fieldnames = ['Model', *metrics]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_task in sorted(cumulative_results.keys()):
            result = cumulative_results[model_task]
            if result:
                row = {'Model': f'Model_{model_task}'}
                for metric in metrics:
                    value = result.get(metric)
                    row[metric] = f'{value:.4f}' if value is not None else 'N/A'
                writer.writerow(row)
            else:
                LOGGER.warning(f"Failed to extract mAP values for Model Task {model_task} on cumulative dataset. CSV file may be empty or invalid.")
                writer.writerow({'Model': f'Model_{model_task}', **{metric: 'N/A' for metric in metrics}})
    
    print(f"Cumulative dataset evaluation table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tables from CSV results")
    parser.add_argument("--eval_dir", type=str, required=True,
                       help="Directory containing evaluation results")
    parser.add_argument("--num_tasks", type=int, required=True,
                       help="Number of tasks")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for tables")
    parser.add_argument("--zero_shot", action="store_true",
                       help="Zero-shot evaluation mode: only process model_1 and evaluate on all datasets")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect individual dataset evaluation results
    individual_results = {}
    cumulative_results = {}
    
    if args.zero_shot:
        # Zero-shot mode: only process model_1, evaluate on all datasets
        model_task = 1
        individual_results[model_task] = {}
        
        # Evaluate on all datasets (from 1 to num_tasks)
        for dataset_task in range(1, args.num_tasks + 1):
            csv_path = os.path.join(args.eval_dir, 
                                   f"model_{model_task}_eval_task_{dataset_task}.csv")
            individual_results[model_task][dataset_task] = extract_metrics_from_csv(csv_path)
        
        # Check for cumulative dataset (optional for zero-shot)
        csv_path = os.path.join(args.eval_dir,
                               f"model_{model_task}_eval_cumulative.csv")
        if os.path.exists(csv_path):
            cumulative_results[model_task] = extract_metrics_from_csv(csv_path)
    else:
        # Incremental learning mode: process all model tasks
        for model_task in range(1, args.num_tasks + 1):
            individual_results[model_task] = {}
            
            # Evaluate on each dataset seen so far
            for dataset_task in range(1, model_task + 1):
                csv_path = os.path.join(args.eval_dir, 
                                       f"model_{model_task}_eval_task_{dataset_task}.csv")
                individual_results[model_task][dataset_task] = extract_metrics_from_csv(csv_path)
            
            # Evaluate on cumulative dataset
            csv_path = os.path.join(args.eval_dir,
                                   f"model_{model_task}_eval_cumulative.csv")
            if os.path.exists(csv_path):
                cumulative_results[model_task] = extract_metrics_from_csv(csv_path)
    
    # Generate tables
    individual_table_path = os.path.join(args.output_dir, "individual_datasets_eval.csv")
    cumulative_table_path = os.path.join(args.output_dir, "cumulative_datasets_eval.csv")
    
    generate_individual_table(individual_results, individual_table_path)
    if len(cumulative_results) > 0:
        generate_cumulative_table(cumulative_results, cumulative_table_path)
    
    print("Evaluation tables generated successfully!")


if __name__ == "__main__":
    main()

