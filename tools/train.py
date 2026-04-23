"""Train a YOLO model on a given dataset and save the trained model.

This script trains a YOLO model on a specified dataset and saves the trained model checkpoint.
It supports both standard detection training and anti-forgetting training for incremental learning.

Usage:
    $ python tools/train.py \
        --model <path/to/model.pt> \
        --data <path/to/dataset.yaml> \
        --save_path <path/to/save/model.pt> \
        [--epochs <num_epochs>] \
        [--batch_size <batch_size>] \
        [--workers <num_workers>] \
        [--device <device>] \
        [--project <project_dir>] \
        [--trainer <trainer_type>] \
        [--<additional_args> ...]

Arguments:
    --model: Path to the model checkpoint file (.pt) for fine-tuning or model configuration
        file (.yaml) for training from scratch. Required argument.
    --data: Path to the dataset configuration file (.yaml). Required argument.
    --save_path: Path where the trained model will be saved. Required argument.
    --epochs: Number of training epochs. Default: 100.
    --batch_size: Batch size for training. Default: 16.
    --workers: Number of worker threads for data loading. Default: 8.
    --device: Device to use for training (e.g., 'cuda', 'cpu', '0', '1').
        Default: 'cuda'.
    --project: Project directory where training logs and outputs will be saved.
        Default: 'runs/detect'.
    --trainer: Type of trainer to use. Options:
        - None or not specified: Use default DetectionTrainer
        - 'antiforget': Use AntiForgetDetectionTrainer for incremental learning
        Default: None.
    --<additional_args>: Additional dynamic arguments can be passed to the model.train()
        method. These will be automatically parsed and passed through. Examples include
        --imgsz, --lr0, --lrf, --momentum, --weight_decay, etc.

Examples:
    $ python tools/train.py \
        --model yolov8l.yaml \
        --data data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --save_path runs/task1/best.pt \
        --epochs 100 \
        --batch_size 16 \
        --device 0
    
    $ python tools/train.py \
        --model runs/task1/best.pt \
        --data data/VOC_inc_10_10/task_2_cls_10/dataset.yaml \
        --save_path runs/task2/best.pt \
        --epochs 100 \
        --batch_size 16 \
        --workers 8 \
        --device cuda \
        --trainer antiforget \
        --imgsz 640 \
        --lr0 0.01
"""

import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, AntiForgetDetectionTrainer, ABRDetectionTrainer
from ultralytics.models.yolo.obb import AntiForgetOBBTrainer


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
    parser.add_argument("--model", type=str, help="Model to train(.pt/.yaml)")
    parser.add_argument("--weight", type=str, default=None, help="Pretrained weight to load(.pt)")
    parser.add_argument("--data", type=str, help="Data config path(.yaml)")
    parser.add_argument("--save_path", type=str, help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project name(where to save logs)")
    parser.add_argument("--trainer", type=str, default=None, help="Trainer to use, default is None, which means use the default trainer")
    args, unknown = parser.parse_known_args()
    dynamic_kwargs = parse_dynamic_named_args(unknown) # Other dynamic arguments

    model = YOLO(args.model)
    if args.weight is not None:
        # This is for loading weights of heterogeneous models while preserving the architecture of the originally initialized model
        model.load(args.weight)

    # Select trainer by model task and user-specified trainer type.
    # When trainer is None, do not pass trainer so the model uses its task-specific default (OBBTrainer for obb, etc.).
    # When trainer is "antiforget", use the task-appropriate anti-forget trainer.
    task = getattr(model, "task", "detect")
    if args.trainer == "antiforget":
        trainer = AntiForgetOBBTrainer if task == "obb" else AntiForgetDetectionTrainer
    elif args.trainer == "abr":
        trainer = ABRDetectionTrainer
    else:
        trainer = None  # use model's default (OBBTrainer for obb, DetectionTrainer for detect, etc.)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch_size, workers=args.workers,
                device=args.device, project=args.project, trainer=trainer, **dynamic_kwargs)
    model.save(args.save_path)


if __name__ == "__main__":
    main()