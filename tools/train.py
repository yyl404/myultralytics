import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, AntiForgetDetectionTrainer


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
    
    if args.trainer == "antiforget":
        trainer = AntiForgetDetectionTrainer
    else:
        trainer = DetectionTrainer
    model.train(data=args.data, epochs=args.epochs, batch=args.batch_size, workers=args.workers,
                device=args.device, project=args.project, trainer=trainer, **dynamic_kwargs)
    model.save(args.save_path)


if __name__ == "__main__":
    main()