"""Dump per-image detections, visualize boxes, and split images by GT matching.

This script runs a trained YOLO detector on a dataset split, writes YOLO-format
prediction txt files plus images with boxes and confidence scores, then compares
each image against ground truth and copies results into three folders:

    - correct: no false alarms and no missed GT
    - missed: at least one unmatched GT box
    - false_alarm: at least one unmatched prediction

An image that has both missed boxes and false alarms is copied into both folders.

Usage:
    $ python tools/detect.py \
        --model <path/to/model.pt> \
        --data <path/to/dataset.yaml> \
        [--weight <path/to/weight.pt>] \
        [--device <device>] \
        [--project <project_dir>] \
        [--save_path <output_dir>] \
        [--iou_threshold <0.50>] \
        [--split <val|test|train>] \
        [--categories <name_or_id ...>] \
        [--<additional_args> ...]

Arguments:
    --model: Path to the model checkpoint file (.pt) or model configuration file (.yaml).
        Required argument.
    --data: Path to the dataset configuration file (.yaml). Required argument.
    --weight: Path to the model weight file to load (.pt). Optional argument.
    --device: Device to use for inference (e.g., 'cuda', 'cpu', '0', '1').
        Default: 'cuda'.
    --project: Project directory forwarded to YOLO predict logs.
        Default: 'runs/detect'.
    --save_path: Output directory for txt files, visualized images, and the three
        matched folders. Default: 'detection_results'.
    --split: Dataset split to run on ('val', 'test', or 'train'). Default: 'val'.
    --iou_threshold: IoU threshold used to match predictions with GT. This does
        not change NMS IoU (pass --iou for NMS). Default: 0.5.
    --categories: Optional class names or ids to keep. Images without these
        classes in GT or predictions are skipped. If omitted, all classes are used.
    --<additional_args>: Additional dynamic arguments passed to model.predict().
        Examples include --imgsz, --conf, --iou, --batch, etc.

Examples:
    $ python tools/detect.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_pseudo_label/task-1/best.pt \
        --data data/VOC_inc_10_10/task_1_cls_10/dataset.yaml \
        --device 0 \
        --save_path det_vis \
        --categories person car

    $ python tools/detect.py \
        --model best.pt \
        --data dataset.yaml \
        --split test \
        --iou_threshold 0.5 \
        --categories 14 \
        --conf 0.25 \
        --imgsz 640
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator

COLOR_TP = (46, 184, 64)
COLOR_FP = (0, 0, 255)
COLOR_FN = (0, 200, 255)
FOLDER_ALL = "all"
FOLDER_CORRECT = "correct"
FOLDER_MISSED = "missed"
FOLDER_FALSE_ALARM = "false_alarm"


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


def _tokenize_categories(raw: Optional[list]) -> list[str]:
    """Split category CLI tokens into class names or id strings."""
    if not raw:
        return []
    tokens = []
    for item in raw:
        text = str(item).strip().strip("[]")
        for part in text.split(","):
            token = part.strip().strip("'\"")
            if token:
                tokens.append(token)
    return tokens


def resolve_categories(raw: Optional[list], names: dict[int, str]) -> Optional[list[int]]:
    """Map category names or ids onto dataset class ids."""
    tokens = _tokenize_categories(raw)
    if not tokens:
        return None
    name_to_id = {str(name).lower(): int(class_id) for class_id, name in names.items()}
    class_ids = []
    unknown = []
    for token in tokens:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            class_id = int(token)
            if class_id not in names:
                unknown.append(token)
            else:
                class_ids.append(class_id)
        elif token.lower() in name_to_id:
            class_ids.append(name_to_id[token.lower()])
        else:
            unknown.append(token)
    if unknown:
        available = ", ".join(f"{class_id}:{name}" for class_id, name in names.items())
        raise ValueError(f"Unknown categories {unknown}. Available classes: {available}")
    return sorted(set(class_ids))


def list_images(source: str) -> list[str]:
    """Collect image files from a YOLO split path (directory or txt list)."""
    path = Path(source)
    if path.is_dir():
        return sorted(
            str(file)
            for file in path.rglob("*")
            if file.is_file() and file.suffix[1:].lower() in IMG_FORMATS
        )
    if path.is_file() and path.suffix.lower() in {".txt", ".csv"}:
        parent = str(path.parent) + "/"
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item:
                continue
            resolved = Path(item.replace("./", parent, 1) if item.startswith("./") else item)
            if resolved.suffix[1:].lower() in IMG_FORMATS:
                images.append(str(resolved))
        return images
    raise FileNotFoundError(f"Split path does not exist or is not an image source: {source}")


def load_gt_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_ids: Optional[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Load YOLO GT boxes as pixel xyxy and class ids, optionally filtered by class."""
    xywhn = []
    classes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError:
                continue
            if class_ids is not None and class_id not in class_ids:
                continue
            classes.append(class_id)
            xywhn.append(box)
    if not classes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    xyxy = xywhn2xyxy(np.asarray(xywhn, dtype=np.float32), w=image_width, h=image_height)
    return np.asarray(xyxy, dtype=np.float32), np.asarray(classes, dtype=np.int32)


def extract_predictions(result, class_ids: Optional[list[int]]):
    """Return prediction xyxy, xywhn, class ids, and confidences as numpy arrays."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, empty.copy(), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    xywhn = boxes.xywhn.cpu().numpy().astype(np.float32)
    classes = boxes.cls.cpu().numpy().astype(np.int32)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    if class_ids is not None:
        keep = np.isin(classes, class_ids)
        xyxy, xywhn, classes, confs = xyxy[keep], xywhn[keep], classes[keep], confs[keep]
    return xyxy, xywhn, classes, confs


def pairwise_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a.astype(np.float32, copy=False)
    b = boxes_b.astype(np.float32, copy=False)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(right_bottom - left_top, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def match_predictions(
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy per-class matching of predictions to GT. Returns TP, FP, FN masks."""
    n_pred = len(pred_xyxy)
    n_gt = len(gt_xyxy)
    tp = np.zeros(n_pred, dtype=bool)
    fp = np.ones(n_pred, dtype=bool)
    fn = np.ones(n_gt, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return tp, fp, fn

    gt_used = np.zeros(n_gt, dtype=bool)
    for pred_index in np.argsort(-pred_conf):
        class_id = pred_cls[pred_index]
        candidates = np.where((gt_cls == class_id) & (~gt_used))[0]
        if len(candidates) == 0:
            continue
        ious = pairwise_iou_xyxy(pred_xyxy[pred_index : pred_index + 1], gt_xyxy[candidates])[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            gt_index = int(candidates[best])
            gt_used[gt_index] = True
            tp[pred_index] = True
            fp[pred_index] = False
            fn[gt_index] = False
    return tp, fp, fn


def prediction_txt_lines(xywhn: np.ndarray, classes: np.ndarray, confs: np.ndarray) -> str:
    """Format YOLO detection lines with confidence: class x y w h conf."""
    lines = [
        f"{int(class_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {conf:.6f}"
        for class_id, box, conf in zip(classes, xywhn, confs)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def save_pair(image: np.ndarray, txt_content: str, folder: Path, image_name: str, label_name: str) -> tuple[Path, Path]:
    """Write one visualized image and its detection txt into a class folder."""
    image_path = folder / "images" / image_name
    label_path = folder / "labels" / label_name
    write_image(image_path, image)
    write_text(label_path, txt_content)
    return image_path, label_path


def copy_pair(src_image: Path, src_label: Path, folder: Path, image_name: str, label_name: str) -> None:
    """Copy one visualized image and its detection txt into a class folder."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, folder / "images" / image_name)
    shutil.copy2(src_label, folder / "labels" / label_name)


def class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_detections(image: np.ndarray, xyxy: np.ndarray, classes: np.ndarray, confs: np.ndarray, names: dict) -> np.ndarray:
    """Draw predicted boxes and confidence scores on a copy of the image."""
    annotator = Annotator(image.copy(), example=names)
    for box, class_id, conf in zip(xyxy, classes, confs):
        label = f"{class_name(names, int(class_id))} {conf:.2f}"
        annotator.box_label(box, label=label)
    return annotator.result()


def draw_comparison(
    image: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    names: dict,
) -> np.ndarray:
    """Draw TP (green), FP (red), and missed GT (yellow) with a legend bar."""
    annotator = Annotator(image.copy(), example=names)
    for box, class_id, conf, is_tp, is_fp in zip(pred_xyxy, pred_cls, pred_conf, tp, fp):
        name = class_name(names, int(class_id))
        if is_tp:
            annotator.box_label(box, label=f"{name} {conf:.2f}", color=COLOR_TP)
        elif is_fp:
            annotator.box_label(box, label=f"{name} {conf:.2f}", color=COLOR_FP)
    for box, class_id, is_fn in zip(gt_xyxy, gt_cls, fn):
        if is_fn:
            name = class_name(names, int(class_id))
            annotator.box_label(box, label=f"{name} missed", color=COLOR_FN)
    canvas = annotator.result()
    bar_h = max(28, canvas.shape[0] // 40)
    bar = np.full((bar_h, canvas.shape[1], 3), 30, dtype=np.uint8)
    legend = "green: correct (TP) | yellow: missed (FN) | red: false alarm (FP)"
    cv2.putText(bar, legend, (8, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, bar_h / 42), (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, canvas], axis=0)


def prepare_output_dirs(save_path: Path) -> dict[str, Path]:
    folders = {
        FOLDER_ALL: save_path / FOLDER_ALL,
        FOLDER_CORRECT: save_path / FOLDER_CORRECT,
        FOLDER_MISSED: save_path / FOLDER_MISSED,
        FOLDER_FALSE_ALARM: save_path / FOLDER_FALSE_ALARM,
    }
    for folder in folders.values():
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(parents=True, exist_ok=True)
    return folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to evaluate(.pt/.yaml)")
    parser.add_argument("--weight", type=str, default=None, help="Model weight to load(.pt)")
    parser.add_argument("--data", type=str, help="Data config path(.yaml)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project name(where to save logs)")
    parser.add_argument("--save_path", type=str, default="detection_results", help="Directory to save detection dumps")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to run on (val/test/train)")
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions with GT; this does not change NMS IoU",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Class names or ids to keep, e.g. --categories person car or --categories 14",
    )
    args, unknown = parser.parse_known_args()
    dynamic_kwargs = parse_dynamic_named_args(unknown)

    if not args.model or not args.data:
        parser.error("--model and --data are required")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou_threshold must be in (0, 1]")

    data = check_det_dataset(args.data)
    names = data["names"]
    class_ids = resolve_categories(args.categories, names)
    split = dynamic_kwargs.pop("split", args.split)
    if split not in data or not data[split]:
        available = [key for key in ("train", "val", "test") if data.get(key)]
        raise FileNotFoundError(f"Split '{split}' not found in {args.data}. Available: {available}")

    image_files = list_images(data[split])
    if not image_files:
        raise FileNotFoundError(f"No images found in split '{split}': {data[split]}")

    save_path = Path(args.save_path)
    folders = prepare_output_dirs(save_path)

    model = YOLO(args.model)
    if args.weight is not None:
        # This is for loading weights of heterogeneous models while preserving the architecture of the originally initialized model
        model.load(args.weight)

    predict_kwargs = {
        "device": args.device,
        "project": args.project,
        **dynamic_kwargs,
        "stream": True,
        "save": False,
        "save_txt": False,
        "verbose": dynamic_kwargs.get("verbose", False),
    }
    if class_ids is None and "classes" in dynamic_kwargs:
        extra_classes = dynamic_kwargs["classes"]
        class_ids = [extra_classes] if isinstance(extra_classes, int) else list(extra_classes)
    if class_ids is not None:
        predict_kwargs["classes"] = class_ids

    counts = {"all": 0, "correct": 0, "missed": 0, "false_alarm": 0, "skipped": 0}
    # Pass the split directory, not a Python list of paths. A list is treated as
    # in-memory images and renamed to image0.jpg, image1.jpg, which breaks GT lookup.
    results = model.predict(source=data[split], **predict_kwargs)
    for result in TQDM(results, total=len(image_files), desc="Dumping detections"):
        image_path = Path(result.path)
        height, width = result.orig_shape[:2]
        orig_img = result.orig_img
        if orig_img is None:
            orig_img = cv2.imread(str(image_path))
            if orig_img is None:
                LOGGER.warning(f"Skipping unreadable image: {image_path}")
                counts["skipped"] += 1
                continue

        pred_xyxy, pred_xywhn, pred_cls, pred_conf = extract_predictions(result, class_ids)
        gt_path = Path(img2label_paths([str(image_path)])[0])
        gt_xyxy, gt_cls = load_gt_boxes(gt_path, width, height, class_ids)

        if class_ids is not None and len(pred_cls) == 0 and len(gt_cls) == 0:
            counts["skipped"] += 1
            continue

        tp, fp, fn = match_predictions(pred_xyxy, pred_cls, pred_conf, gt_xyxy, gt_cls, args.iou_threshold)
        has_fp = bool(fp.any())
        has_fn = bool(fn.any())
        vis_names = result.names if getattr(result, "names", None) else names

        stem = image_path.stem
        image_name = image_path.name
        label_name = f"{stem}.txt"
        txt_content = prediction_txt_lines(pred_xywhn, pred_cls, pred_conf)

        pred_vis = draw_detections(orig_img, pred_xyxy, pred_cls, pred_conf, vis_names)
        save_pair(pred_vis, txt_content, folders[FOLDER_ALL], image_name, label_name)
        counts["all"] += 1

        targets = []
        if has_fn:
            targets.append(FOLDER_MISSED)
            counts["missed"] += 1
        if has_fp:
            targets.append(FOLDER_FALSE_ALARM)
            counts["false_alarm"] += 1
        if not has_fn and not has_fp and (len(gt_cls) > 0 or len(pred_cls) > 0):
            targets.append(FOLDER_CORRECT)
            counts["correct"] += 1

        if targets:
            comparison = draw_comparison(
                orig_img, pred_xyxy, pred_cls, pred_conf, gt_xyxy, gt_cls, tp, fp, fn, vis_names
            )
            first_image, first_label = save_pair(
                comparison, txt_content, folders[targets[0]], image_name, label_name
            )
            for folder_name in targets[1:]:
                copy_pair(first_image, first_label, folders[folder_name], image_name, label_name)

    LOGGER.info(
        "Detection dump saved to "
        f"{save_path} | all={counts['all']} correct={counts['correct']} "
        f"missed={counts['missed']} false_alarm={counts['false_alarm']} skipped={counts['skipped']}"
    )
    print(f"Results saved to {save_path}")
    print(
        f"  {FOLDER_ALL}: {counts['all']}\n"
        f"  {FOLDER_CORRECT}: {counts['correct']}\n"
        f"  {FOLDER_MISSED}: {counts['missed']}\n"
        f"  {FOLDER_FALSE_ALARM}: {counts['false_alarm']}"
    )


if __name__ == "__main__":
    main()
