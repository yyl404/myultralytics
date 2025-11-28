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


def extract_map_from_csv(csv_path):
    """
    Extract mAP50 and mAP50-95 from evaluation CSV file.
    Returns the average mAP across all classes.
    """
    if not os.path.exists(csv_path):
        return None, None
    
    map50_values = []
    map50_95_values = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip 'all' row if exists (we'll handle it separately)
            if row.get('Class', '').lower() == 'all':
                map50 = float(row.get('mAP50', 0))
                map50_95 = float(row.get('mAP50-95', 0))
                return map50, map50_95
            
            # Collect mAP values
            map50_str = row.get('mAP50', '')
            map50_95_str = row.get('mAP50-95', '')
            
            if map50_str and map50_95_str:
                map50_val = float(map50_str)
                map50_95_val = float(map50_95_str)
                map50_values.append(map50_val)
                map50_95_values.append(map50_95_val)
    
    # Calculate mean if no 'all' row found
    if map50_values:
        map50 = sum(map50_values) / len(map50_values)
        map50_95 = sum(map50_95_values) / len(map50_95_values)
        return map50, map50_95
    else:
        return None, None


def generate_individual_table(eval_results, output_path):
    """
    Generate table for individual dataset evaluations.
    
    eval_results: dict of {model_task: {dataset_task: (map50, map50_95)}}
    """
    # Get all unique model tasks and dataset tasks
    model_tasks = sorted(eval_results.keys())
    dataset_tasks = set()
    for model_results in eval_results.values():
        dataset_tasks.update(model_results.keys())
    dataset_tasks = sorted(dataset_tasks)
    
    # Build fieldnames
    fieldnames = ['Model_Task']
    for dataset_task in dataset_tasks:
        fieldnames.append(f'Task_{dataset_task}_mAP50')
        fieldnames.append(f'Task_{dataset_task}_mAP50-95')
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_task in model_tasks:
            row = {'Model_Task': f'Task_{model_task}'}
            for dataset_task in dataset_tasks:
                if dataset_task in eval_results[model_task]:
                    map50, map50_95 = eval_results[model_task][dataset_task]
                    if map50 is not None:
                        row[f'Task_{dataset_task}_mAP50'] = f'{map50:.4f}'
                        row[f'Task_{dataset_task}_mAP50-95'] = f'{map50_95:.4f}'
                    else:
                        LOGGER.warning(f"Failed to extract mAP values for Model Task {model_task} on Dataset Task {dataset_task}. CSV file may be empty or invalid.")
                        row[f'Task_{dataset_task}_mAP50'] = 'N/A'
                        row[f'Task_{dataset_task}_mAP50-95'] = 'N/A'
                else:
                    LOGGER.warning(f"No evaluation results found for Model Task {model_task} on Dataset Task {dataset_task}.")
                    row[f'Task_{dataset_task}_mAP50'] = 'N/A'
                    row[f'Task_{dataset_task}_mAP50-95'] = 'N/A'
            writer.writerow(row)
    
    print(f"Individual dataset evaluation table saved to {output_path}")


def generate_cumulative_table(cumulative_results, output_path):
    """
    Generate table for cumulative dataset evaluations.
    
    cumulative_results: dict of {model_task: (map50, map50_95)}
    """
    fieldnames = ['Model', 'mAP50', 'mAP50-95']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_task in sorted(cumulative_results.keys()):
            map50, map50_95 = cumulative_results[model_task]
            if map50 is not None:
                writer.writerow({
                    'Model': f'Model_{model_task}',
                    'mAP50': f'{map50:.4f}',
                    'mAP50-95': f'{map50_95:.4f}'
                })
            else:
                LOGGER.warning(f"Failed to extract mAP values for Model Task {model_task} on cumulative dataset. CSV file may be empty or invalid.")
                writer.writerow({
                    'Model': f'Model_{model_task}',
                    'mAP50': 'N/A',
                    'mAP50-95': 'N/A'
                })
    
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
            map50, map50_95 = extract_map_from_csv(csv_path)
            individual_results[model_task][dataset_task] = (map50, map50_95)
        
        # Check for cumulative dataset (optional for zero-shot)
        csv_path = os.path.join(args.eval_dir,
                               f"model_{model_task}_eval_cumulative.csv")
        if os.path.exists(csv_path):
            map50, map50_95 = extract_map_from_csv(csv_path)
            cumulative_results[model_task] = (map50, map50_95)
    else:
        # Incremental learning mode: process all model tasks
        for model_task in range(1, args.num_tasks + 1):
            individual_results[model_task] = {}
            
            # Evaluate on each dataset seen so far
            for dataset_task in range(1, model_task + 1):
                csv_path = os.path.join(args.eval_dir, 
                                       f"model_{model_task}_eval_task_{dataset_task}.csv")
                map50, map50_95 = extract_map_from_csv(csv_path)
                individual_results[model_task][dataset_task] = (map50, map50_95)
            
            # Evaluate on cumulative dataset
            csv_path = os.path.join(args.eval_dir,
                                   f"model_{model_task}_eval_cumulative.csv")
            if os.path.exists(csv_path):
                map50, map50_95 = extract_map_from_csv(csv_path)
                cumulative_results[model_task] = (map50, map50_95)
    
    # Generate tables
    individual_table_path = os.path.join(args.output_dir, "individual_datasets_eval.csv")
    cumulative_table_path = os.path.join(args.output_dir, "cumulative_datasets_eval.csv")
    
    generate_individual_table(individual_results, individual_table_path)
    if len(cumulative_results) > 0:
        generate_cumulative_table(cumulative_results, cumulative_table_path)
    
    print("Evaluation tables generated successfully!")


if __name__ == "__main__":
    main()

