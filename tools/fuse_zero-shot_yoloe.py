"""
Fuse the class names' embedding to YOLOE classification layer weights.

This script converts YOLOE model weights (e.g., yoloe-v8l-seg.pt) to YOLOv8 architecture
by fusing text embeddings of class names into the classification layer weights.

The conversion process:
1. Loads YOLOE model weights and extracts class names
2. Generates text embeddings for class names using get_text_pe() method
3. Creates a YOLO detection model with the specified configuration
4. Copies YOLOE pretrained weights to YOLO detection (excluding classification head)
5. Fuses text embeddings into the classification layer's last convolution weights
6. Saves the converted YOLO detection model weights

Usage:
    python tools/fuse_zero-shot_yoloe.py \\
        --input yoloe-v8l-seg.pt \\
        --output yolov8l_fused.pt \\
        --model_cfg yolov8l.yaml \\
        --class_names class1 class2 class3  # optional \\
        --data dataset.yaml # optional
"""
import argparse
import os.path as OSP

from ultralytics import YOLO, YOLOE
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import LOGGER, YAML


def fuse_zero_shot_yoloe(
    input_weights: str,
    output_weights: str,
    model_cfg: str = "yolov8l.yaml",
    class_names: list = None
):
    """
    Convert YOLOE model weights to YOLO detection architecture with fused class embeddings.
    
    Args:
        input_weights (str): Path to input YOLOE model weights (e.g., yoloe-v8l-seg.pt)
        output_weights (str): Path to save output YOLO detection model weights
        model_cfg (str): YOLO detection model configuration file (default: yolov8l.yaml)
        class_names (list, optional): List of class names. If None, will try to extract from model.
    """
    LOGGER.info(f"Loading YOLOE model from {input_weights}")
    
    # Load YOLOE model
    yoloe_model = YOLOE(input_weights)
    yoloe_model.eval()

    # Embed class names
    if class_names is None:
        if hasattr(yoloe_model, "names"):
            class_names = [yoloe_model.names[k] for k in sorted(yoloe_model.names.keys())]
        else:
            LOGGER.error("Class names not found in YOLOE model, must be provided manually")

    # Initialize YOLO model
    ## 1. Convert config file to correct class num
    model_name = model_cfg.split(".")[0]
    model_cfg = yaml_model_load(model_cfg)
    model_cfg["nc"] = len(class_names)
    save_dir = OSP.dirname(output_weights)
    YAML.save(data=model_cfg, file=OSP.join(save_dir, f"{model_name}-nc{len(class_names)}.yaml"))
    ## 2. Load converted config file
    yolo_model = YOLO(OSP.join(save_dir, f"{model_name}-nc{len(class_names)}.yaml"))
    yolo_model.eval()

    # fuse text embeddings to classify head
    tpe = yoloe_model.get_text_pe(class_names)
    yoloe_model.model.model[-1].fuse(tpe)

    # Copy yoloe weight to yolo model
    yolo_model.model.load_state_dict(yoloe_model.model.state_dict(), strict=False)
    yolo_model.model.names = {k: v for k, v in enumerate(class_names)}

    # Save model
    yolo_model.save(output_weights)


def main():
    parser = argparse.ArgumentParser(description="Fuse YOLOE class embeddings to YOLO classification layer")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input YOLOE model weights (e.g., yoloe-v8l-seg.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output YOLO model weights",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="yolov8l.yaml",
        help="YOLO model configuration file (default: yolov8l.yaml)",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="List of class names (optional, will try to extract from model if not provided)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data set's .yaml path (extract class names from it)",
    )
    
    args = parser.parse_args()

    if args.data is not None:
        yaml_names = YAML.load(args.data)["names"]
        args.class_names = [yaml_names[k] for k in sorted(yaml_names)]
    
    fuse_zero_shot_yoloe(
        input_weights=args.input,
        output_weights=args.output,
        model_cfg=args.model_cfg,
        class_names=args.class_names
    )


if __name__ == "__main__":
    main()
