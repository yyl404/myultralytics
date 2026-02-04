#!/usr/bin/env python3
"""
Extract and summarize NSGP+pseudo_label evaluation results for different VOC splits.

This script extracts mAP50 and mAP50-95 results for old classes, new classes, and all classes
from the evaluation_results folders of 10+10, 15+5, and 19+1 splits.
"""

import csv
import os
import sys

# VOC class names in order (0-19)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Class splits for each configuration
CLASS_SPLITS = {
    '10_10': {
        'old': VOC_CLASSES[0:10],  # 0-9: aeroplane to cow
        'new': VOC_CLASSES[10:20]  # 10-19: diningtable to tvmonitor
    },
    '15_5': {
        'old': VOC_CLASSES[0:15],  # 0-14: aeroplane to person
        'new': VOC_CLASSES[15:20]  # 15-19: pottedplant to tvmonitor
    },
    '19_1': {
        'old': VOC_CLASSES[0:19],  # 0-18: aeroplane to train
        'new': VOC_CLASSES[19:20]  # 19: tvmonitor
    }
}


def extract_results_from_csv(csv_path, old_classes, new_classes):
    """
    Extract mAP50 and mAP50-95 from evaluation CSV file.
    Returns (old_map50, old_map50_95, new_map50, new_map50_95, all_map50, all_map50_95)
    """
    if not os.path.exists(csv_path):
        return None, None, None, None, None, None
    
    old_map50_values = []
    old_map50_95_values = []
    new_map50_values = []
    new_map50_95_values = []
    all_map50_values = []
    all_map50_95_values = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row.get('Class', '').strip()
            if not class_name:
                continue
            
            map50_str = row.get('mAP50', '')
            map50_95_str = row.get('mAP50-95', '')
            
            if not map50_str or not map50_95_str:
                continue
            
            try:
                map50_val = float(map50_str)
                map50_95_val = float(map50_95_str)
                
                all_map50_values.append(map50_val)
                all_map50_95_values.append(map50_95_val)
                
                if class_name in old_classes:
                    old_map50_values.append(map50_val)
                    old_map50_95_values.append(map50_95_val)
                elif class_name in new_classes:
                    new_map50_values.append(map50_val)
                    new_map50_95_values.append(map50_95_val)
            except ValueError:
                continue
    
    # Calculate averages
    old_map50 = sum(old_map50_values) / len(old_map50_values) if old_map50_values else None
    old_map50_95 = sum(old_map50_95_values) / len(old_map50_95_values) if old_map50_95_values else None
    new_map50 = sum(new_map50_values) / len(new_map50_values) if new_map50_values else None
    new_map50_95 = sum(new_map50_95_values) / len(new_map50_95_values) if new_map50_95_values else None
    all_map50 = sum(all_map50_values) / len(all_map50_values) if all_map50_values else None
    all_map50_95 = sum(all_map50_95_values) / len(all_map50_95_values) if all_map50_95_values else None
    
    return old_map50, old_map50_95, new_map50, new_map50_95, all_map50, all_map50_95


def main():
    base_dir = '/root/myultralytics/runs'
    
    # Results storage
    results = {}
    
    # Process each split
    for split_name in ['10_10', '15_5', '19_1']:
        eval_dir = os.path.join(base_dir, f'yolov8l_voc_{split_name}_fromscratch_nsgp+pseudo_label/evaluation_results')
        csv_path = os.path.join(eval_dir, 'model_2_eval_cumulative.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {split_name}")
            continue
        
        old_classes = CLASS_SPLITS[split_name]['old']
        new_classes = CLASS_SPLITS[split_name]['new']
        
        old_map50, old_map50_95, new_map50, new_map50_95, all_map50, all_map50_95 = \
            extract_results_from_csv(csv_path, old_classes, new_classes)
        
        results[split_name] = {
            'old': (old_map50, old_map50_95),
            'new': (new_map50, new_map50_95),
            'all': (all_map50, all_map50_95)
        }
    
    # Generate table
    print("\n" + "="*80)
    print("NSGP+Pseudo Label Results Summary")
    print("="*80)
    print("\nTable: mAP50 and mAP50-95 for Old Classes, New Classes, and All Classes")
    print("-"*80)
    
    # Header
    header = f"{'Split':<10} {'Metric':<15} {'Old Classes':<15} {'New Classes':<15} {'All Classes':<15}"
    print(header)
    print("-"*80)
    
    # Data rows
    for split_name in ['10_10', '15_5', '19_1']:
        if split_name not in results:
            continue
        
        split_display = split_name.replace('_', '+')
        old_map50, old_map50_95 = results[split_name]['old']
        new_map50, new_map50_95 = results[split_name]['new']
        all_map50, all_map50_95 = results[split_name]['all']
        
        # mAP50 row
        if old_map50 is not None and new_map50 is not None and all_map50 is not None:
            print(f"{split_display:<10} {'mAP50':<15} {old_map50:<15.4f} {new_map50:<15.4f} {all_map50:<15.4f}")
        else:
            print(f"{split_display:<10} {'mAP50':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # mAP50-95 row
        if old_map50_95 is not None and new_map50_95 is not None and all_map50_95 is not None:
            print(f"{'':<10} {'mAP50-95':<15} {old_map50_95:<15.4f} {new_map50_95:<15.4f} {all_map50_95:<15.4f}")
        else:
            print(f"{'':<10} {'mAP50-95':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        print("-"*80)
    
    # Generate CSV output
    csv_output_path = '/root/myultralytics/nsgp_pseudo_label_results.csv'
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Split', 'Metric', 'Old Classes', 'New Classes', 'All Classes'])
        
        for split_name in ['10_10', '15_5', '19_1']:
            if split_name not in results:
                continue
            
            split_display = split_name.replace('_', '+')
            old_map50, old_map50_95 = results[split_name]['old']
            new_map50, new_map50_95 = results[split_name]['new']
            all_map50, all_map50_95 = results[split_name]['all']
            
            # mAP50 row
            if old_map50 is not None and new_map50 is not None and all_map50 is not None:
                writer.writerow([split_display, 'mAP50', f'{old_map50:.4f}', f'{new_map50:.4f}', f'{all_map50:.4f}'])
            else:
                writer.writerow([split_display, 'mAP50', 'N/A', 'N/A', 'N/A'])
            
            # mAP50-95 row
            if old_map50_95 is not None and new_map50_95 is not None and all_map50_95 is not None:
                writer.writerow([split_display, 'mAP50-95', f'{old_map50_95:.4f}', f'{new_map50_95:.4f}', f'{all_map50_95:.4f}'])
            else:
                writer.writerow([split_display, 'mAP50-95', 'N/A', 'N/A', 'N/A'])
    
    print(f"\nResults saved to: {csv_output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

