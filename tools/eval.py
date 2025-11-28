"""Evaluate given model on given dataset and save the results to a CSV file.

This script evaluates a trained YOLO model on a specified dataset and saves the evaluation
results (including mAP metrics, precision, recall, F1 scores) and confusion matrix to CSV files.

Usage:
    $ python tools/eval.py \
        --model <path/to/model.pt> \
        --data <path/to/dataset.yaml> \
        [--weight <path/to/weight.pt>] \
        [--device <device>] \
        [--project <project_dir>] \
        [--save_path <results.csv>] \
        [--confusion_matrix_path <confusion_matrix.csv>] \
        [--<additional_args> ...]

Arguments:
    --model: Path to the model checkpoint file (.pt) or model configuration file (.yaml).
        Required argument.
    --data: Path to the dataset configuration file (.yaml). Required argument.
    --weight: Path to the model weight file to load (.pt). Optional argument.
    --device: Device to use for evaluation (e.g., 'cuda', 'cpu', '0', '1'). 
        Default: 'cuda'.
    --project: Project directory where evaluation logs and outputs will be saved.
        Default: 'runs/detect'.
    --save_path: Path to save the evaluation results CSV file. The CSV contains metrics
        for each class including: Class, Instances, Box-P (Precision), Box-R (Recall),
        Box-F1, mAP50, mAP50-95. Default: 'results.csv'.
    --confusion_matrix_path: Path to save the confusion matrix CSV file.
        Default: 'confusion_matrix.csv'.
    --<additional_args>: Additional dynamic arguments can be passed to the model.val()
        method. These will be automatically parsed and passed through. Examples include
        --imgsz, --conf, --iou, --batch, etc.

Examples:
    $ python tools/eval.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label/task-1/best.pt \
        --data data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --device 0 \
        --save_path eval_results.csv
    
    $ python tools/eval.py \
        --model best.pt \
        --data dataset.yaml \
        --device cuda \
        --project runs/evaluation \
        --save_path results.csv \
        --confusion_matrix_path cm.csv \
        --imgsz 640 \
        --batch 16
"""

import argparse
import csv

from ultralytics import YOLO


def _coerce_value(raw: str):
    # Try to convert strings to bool/int/float when possible; fallback to string
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def parse_dynamic_named_args(tokens):
    """
    Parse arbitrary named CLI args into a dict.

    Supports forms:
    - --key value
    - --key=value
    - --key [value1, value2, ...]
    - --flag (boolean True)
    """
    extra = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" in token:
                key, raw_val = token[2:].split("=", 1)
                extra[key] = _coerce_value(raw_val)
                i += 1
            else:
                key = token[2:]
                # Check if next token is a list starting with '['
                if i + 1 < len(tokens) and tokens[i + 1].startswith("["):
                    # Parse list: [value1, value2, ...]
                    list_values = []
                    i += 1  # Move to the '[' token
                    list_token = tokens[i]
                    
                    # Handle cases where list is in a single token: [value1,value2] or [value1, value2]
                    if list_token.startswith("[") and list_token.endswith("]"):
                        list_str = list_token[1:-1]  # Remove '[' and ']'
                        if list_str.strip():
                            # Split by comma and clean up each value
                            list_values = [v.strip() for v in list_str.split(",") if v.strip()]
                    else:
                        # List spans multiple tokens: [ value1 value2 ] or [ value1, value2 ]
                        # Extract content from opening bracket if present
                        if list_token.startswith("["):
                            first_val = list_token[1:].rstrip(",")
                            if first_val.strip():
                                list_values.append(first_val)
                        
                        # Collect all values until we find the closing ']'
                        i += 1
                        while i < len(tokens):
                            token = tokens[i]
                            if token.endswith("]"):
                                # Last token, extract value before ']'
                                last_val = token.rstrip("]").rstrip(",")
                                if last_val.strip():
                                    list_values.append(last_val)
                                break
                            else:
                                # Regular value token, remove trailing comma if present
                                val = token.rstrip(",")
                                if val.strip():
                                    list_values.append(val)
                            i += 1
                    
                    # Coerce each value and add to dict
                    extra[key] = [_coerce_value(v) for v in list_values if v]
                    i += 1
                # If next token exists and is not another flag, treat it as value; else flag=True
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    extra[key] = _coerce_value(tokens[i + 1])
                    i += 2
                else:
                    extra[key] = True
                    i += 1
        else:
            # Positional or stray token; skip
            i += 1

    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to evaluate(.pt/.yaml)")
    parser.add_argument("--weight", type=str, default=None, help="Model weight to load(.pt)")
    parser.add_argument("--data", type=str, help="Data config path(.yaml)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project name(where to save logs)")
    parser.add_argument("--save_path", type=str, default="results.csv", help="Path to save results")
    parser.add_argument("--confusion_matrix_path", type=str, default="confusion_matrix.csv", help="Path to save confusion matrix")
    args, unknown = parser.parse_known_args()
    dynamic_kwargs = parse_dynamic_named_args(unknown) # Other dynamic arguments

    model = YOLO(args.model)
    if args.weight is not None:
        # This is for loading weights of heterogeneous models while preserving the architecture of the originally initialized model
        model.load(args.weight)
    results = model.val(data=args.data, device=args.device, project=args.project, **dynamic_kwargs)
    summary = results.summary()
    confusion_matrix = results.confusion_matrix.summary()
    
    # Write results to CSV file
    with open(args.save_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Class', 'Instances', 'Box-P', 'Box-R', 'Box-F1', 'mAP50', 'mAP50-95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in summary:
            writer.writerow({
                'Class': item["Class"],
                'Instances': item["Instances"],
                'Box-P': item["Box-P"],
                'Box-R': item["Box-R"],
                'Box-F1': item["Box-F1"],
                'mAP50': item["mAP50"],
                'mAP50-95': item["mAP50-95"]
            })
    
    print(f"Results saved to {args.save_path}")

    # Write confusion matrix to CSV file
    with open(args.confusion_matrix_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = confusion_matrix[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in confusion_matrix:
            writer.writerow({
                **{key: item[key] for key in fieldnames},
            })

if __name__ == "__main__":
    main()