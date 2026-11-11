"""Teacher high-confidence error analysis for class-incremental detection.

A teacher trained on the history stages is run on the training images of a later
stage. Because it has never seen those classes, it reports them -- often with
high confidence -- as history classes. This tool measures, as a function of the
reported confidence threshold:

- type 1: background -> history false positive (a box whose best GT IoU is not above the threshold)
- type 2: target-stage -> history error (a box whose best GT IoU is above the
  threshold and whose GT is a target-stage object)
- type 3: later-stage -> history error (same, for a GT object of a later stage)
- miss: history-class GT object with no box whose IoU is above the threshold

Each reported box is classified by the GT box with which it has the largest IoU;
a box with no GT above the IoU threshold is a type-1 background false positive.
The false-positive rates use the total number of reported boxes at the threshold
as denominator; the miss rate uses the total number of history-class GT objects.

When ``--vis_dir`` is given, the tool also renders typical examples of every
error type at ``--vis_conf`` (default 0.25, the Ultralytics inference default):
GT boxes are gray, history-matched boxes green, type-1 boxes blue, type-2 boxes
red, and type-3 boxes magenta.

Usage:
    $ python tools/teacher_error_analysis.py \
        --predictions runs/<run>/teacher_errors/task_2/predictions \
        --model runs/<run>/task-1/best.pt \
        --stages data/VOC_10+10/task_1_cls_10/dataset.yaml \
                 data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --target_stage 2 \
        --source_cfg data/VOC-YOLO/VOC.yaml \
        --split train \
        --vis_dir runs/<run>/teacher_errors/task_2/examples \
        --output_dir runs/<run>/teacher_errors/task_2

Arguments:
    --predictions: Dump root from tools/predict.py; its ``all/labels`` directory
        holds ``class x y w h confidence`` lines per image.
    --model: Teacher checkpoint whose class names label the predictions.
    --stages: Ordered per-stage dataset yamls; classes of stages before the
        target stage are the history classes, stages after it are the future.
    --target_stage: 1-based stage whose training images the teacher sees.
    --source_cfg: Dataset yaml holding the full class-space GT annotation.
    --split: Split of the source dataset providing the GT (default: train).
    --iou_threshold: IoU above which a box counts as matched (default: 0.5).
    --conf_min / --conf_max / --conf_step: Reported-confidence sweep.
    --vis_dir: Optional directory for example visualizations (one montage plus
        the individual images per error type) and ``examples.csv``.
    --vis_conf: Confidence threshold used to pick and draw the examples.
    --vis_per_type: Number of examples rendered per error type.

Outputs:
    error_rates.csv / .md   per-threshold counts and rates
    error_rates.png         rates vs confidence threshold
    summary.txt             protocol, class groups and denominators
    examples/               optional typical examples per error type
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.plotting import Annotator

from utils import (
    index_label_files,
    normalize_names,
    pairwise_iou_xyxy,
    parse_label_line,
    resolve_split_roots,
    stage_class_names,
)


KIND_ORDER = ("type1", "type2", "type3", "miss")
KIND_COLORS = {
    "type1": (255, 80, 0),  # BGR blue
    "type2": (0, 0, 255),  # BGR red
    "type3": (255, 0, 255),  # BGR magenta
    "history": (0, 170, 0),  # BGR green
}
GT_COLOR = (185, 185, 185)


def _load_predictions(label_dir, model_names):
    """Read prediction dumps into per-image class names, xyxy boxes and confidences.

    Returns:
        dict[str, tuple[list[str], np.ndarray, np.ndarray]]: Image stem ->
        (class names, boxes with shape (N, 4) in normalized xyxy, confidences
        with shape (N,)).
    """
    predictions = {}
    for label_path in sorted(Path(label_dir).glob("*.txt")):
        names = []
        boxes = []
        confs = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"Prediction '{label_path}:{line_number}' must be 'class x y w h confidence', got {line!r}"
                )
            class_id = int(parts[0])
            if class_id not in model_names:
                raise ValueError(f"Prediction class id {class_id} at '{label_path}:{line_number}' is not a model class")
            x, y, w, h = (float(value) for value in parts[1:5])
            names.append(model_names[class_id])
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            confs.append(float(parts[5]))
        predictions[label_path.stem] = (
            names,
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(confs, dtype=np.float32),
        )
    if not predictions:
        raise FileNotFoundError(f"No prediction txt files found under: {label_dir}")
    return predictions


def _load_gt(label_path, source_names, class_groups):
    """Read full class-space GT into class names, normalized xyxy boxes and groups."""
    names = []
    boxes = []
    groups = []
    for line_number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            raise ValueError(f"Invalid GT label at '{label_path}:{line_number}': {line!r}")
        class_id, x, y, w, h = parsed
        if class_id not in source_names:
            raise ValueError(
                f"GT class id {class_id} at '{label_path}:{line_number}' is outside the source class space"
            )
        class_name = source_names[class_id]
        if class_name not in class_groups:
            raise ValueError(f"GT class '{class_name}' at '{label_path}:{line_number}' belongs to no stage")
        names.append(class_name)
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        groups.append(class_groups[class_name])
    return names, np.asarray(boxes, dtype=np.float32).reshape(-1, 4), groups


def _classify_boxes(pred_confs, gt_groups, iou_matrix, threshold, iou_threshold):
    """Classify reported boxes of one image at a confidence threshold.

    Args:
        pred_confs (np.ndarray): Prediction confidences with shape (N,).
        gt_groups (Sequence[str]): Stage group of every GT box.
        iou_matrix (np.ndarray): Prediction-to-GT IoU matrix with shape (N, M).
        threshold (float): Reported-confidence threshold.
        iou_threshold (float): IoU above which a box counts as matched.

    Returns:
        tuple[np.ndarray, list[str], list[int | None]]: Reported prediction
        indices, one kind per reported box (``type1``/``type2``/``type3``/
        ``history``), and the matched GT index per box (``None`` for type 1).
    """
    reported = np.where(pred_confs >= threshold)[0]
    if len(reported) == 0:
        return reported, [], []
    if len(gt_groups) == 0:
        return reported, ["type1"] * len(reported), [None] * len(reported)

    ious = iou_matrix[reported]  # (num_reported, num_gt)
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(len(reported)), best_gt]
    group_kind = {"history": "history", "target": "type2", "future": "type3"}
    kinds = []
    gt_refs = []
    for row, gt_index in enumerate(best_gt):
        if best_iou[row] <= iou_threshold:
            kinds.append("type1")
            gt_refs.append(None)
            continue
        kinds.append(group_kind[gt_groups[gt_index]])
        gt_refs.append(int(gt_index))
    return reported, kinds, gt_refs


def _count_image(pred_confs, gt_groups, iou_matrix, threshold, iou_threshold):
    """Count reported boxes by error type and missed GT objects for one image."""
    counts = defaultdict(int)
    reported, kinds, _ = _classify_boxes(pred_confs, gt_groups, iou_matrix, threshold, iou_threshold)
    counts["pred"] = len(reported)
    for kind in kinds:
        counts[kind] += 1
    if len(reported) == 0:
        missed = np.ones(len(gt_groups), dtype=bool)
    else:
        missed = iou_matrix[reported].max(axis=0) <= iou_threshold
    counts["miss_history"] = sum(
        1 for gt_index, group in enumerate(gt_groups) if missed[gt_index] and group == "history"
    )
    return counts


def _safe_rate(numerator, denominator):
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    return float(numerator) / denominator if denominator else 0.0


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows):
    columns = [
        ("confidence_threshold", "Conf"),
        ("n_pred", "Reported"),
        ("type1_rate", "Type 1 rate"),
        ("type2_rate", "Type 2 rate"),
        ("type3_rate", "Type 3 rate"),
        ("miss_rate", "History miss rate"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join(["---:"] * len(columns)) + "|",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            cells.append(str(value) if key in {"confidence_threshold", "n_pred"} else f"{value:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path, rows, target_stage, has_future):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["confidence_threshold"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(thresholds, [row["type1_rate"] for row in rows], marker="o", label="Type 1 background -> history")
    axis.plot(
        thresholds, [row["type2_rate"] for row in rows], marker="s", label=f"Type 2 stage {target_stage} -> history"
    )
    if has_future:
        axis.plot(thresholds, [row["type3_rate"] for row in rows], marker="D", label="Type 3 later stages -> history")
    axis.plot(thresholds, [row["miss_rate"] for row in rows], marker="^", label="History miss")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Rate")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _index_images(image_dirs):
    """Index source images by stem, rejecting ambiguous duplicate stems."""
    index = {}
    for image_dir in image_dirs:
        for image_path in sorted(Path(image_dir).iterdir()):
            if not image_path.is_file() or image_path.suffix[1:].lower() not in IMG_FORMATS:
                continue
            previous = index.setdefault(image_path.stem, image_path)
            if previous != image_path:
                raise ValueError(f"Duplicate image stem '{image_path.stem}': '{previous}' and '{image_path}'")
    return index


def _history_missed_indices(gt_groups, missed):
    """Return the GT indices of missed history-class objects."""
    return [gt_index for gt_index, group in enumerate(gt_groups) if missed[gt_index] and group == "history"]


def _missed_class_summary(gt_names, missed_indices):
    """Return a deterministic ``class xcount`` summary over the given GT indices."""
    counts = defaultdict(int)
    for gt_index in missed_indices:
        counts[gt_names[gt_index]] += 1
    return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_examples(entries, limit):
    """Pick the highest-scoring images, one entry per image."""
    best = {}
    for score, stem in entries:
        if stem not in best or score > best[stem][0]:
            best[stem] = (score, stem)
    return sorted(best.values(), key=lambda item: (-item[0], item[1]))[:limit]


def _render_example(
    image_path,
    output_path,
    kind,
    pred_names,
    pred_boxes,
    pred_confs,
    gt_names,
    gt_boxes,
    gt_groups,
    iou_matrix,
    threshold,
    iou_threshold,
):
    """Draw GT boxes, reported boxes and the featured error type on one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]

    reported, kinds, gt_refs = _classify_boxes(pred_confs, gt_groups, iou_matrix, threshold, iou_threshold)
    missed = np.ones(len(gt_names), dtype=bool)
    if len(reported):
        missed = iou_matrix[reported].max(axis=0) <= iou_threshold

    featured = [(row, gt_refs[row]) for row in range(len(reported)) if kinds[row] == kind]
    highlighted_gt = None
    if kind in {"type2", "type3"} and featured:
        best_row = max(featured, key=lambda item: pred_confs[reported[item[0]]])[0]
        highlighted_gt = gt_refs[best_row]

    for gt_index, gt_box in enumerate(gt_boxes):
        x1, y1 = int(gt_box[0] * width), int(gt_box[1] * height)
        x2, y2 = int(gt_box[2] * width), int(gt_box[3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), GT_COLOR, 1)
        if gt_index != highlighted_gt:
            continue
        label = f"GT {gt_names[gt_index]}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_x = max(0, min(width - label_width, x2 - label_width))
        text_y = y1 - 4 if y1 - 4 > label_height else y1 + label_height + 2
        cv2.rectangle(
            image,
            (text_x, text_y - label_height - 2),
            (text_x + label_width, text_y + 2),
            GT_COLOR,
            -1,
        )
        cv2.putText(
            image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA
        )

    annotator = Annotator(image, line_width=2)
    for row, pred_index in enumerate(reported):
        box = pred_boxes[pred_index]
        pixel_box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        label = f"{pred_names[pred_index]} {pred_confs[pred_index]:.2f} [{kinds[row]}]"
        annotator.box_label(pixel_box, label, color=KIND_COLORS[kinds[row]])
    result = annotator.result()

    if kind == "miss":
        history_missed = _history_missed_indices(gt_groups, missed)
        counts = defaultdict(int)
        for gt_index in history_missed:
            counts[gt_names[gt_index]] += 1
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        line_one = f"missed history GT: {len(history_missed)}"
        line_two = ", ".join(f"{name} x{count}" for name, count in top)
        if len(top) < len(counts):
            line_two += ", ..."
        strip_height = 52
        strip = np.full((strip_height, result.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(strip, line_one, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(strip, line_two, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        result = np.concatenate([result, strip], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write example image: {output_path}")


def _montage(paths, output_path, height=360):
    """Stack example images horizontally under one shared color legend."""
    panels = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        scale = height / image.shape[0]
        panels.append(cv2.resize(image, (max(1, int(image.shape[1] * scale)), height)))
    gap = 8
    width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    legend_height = 36
    canvas = np.full((height + legend_height, width, 3), 255, dtype=np.uint8)
    legend = "GT gray | type1 blue | type2 red | type3 magenta | history green"
    cv2.putText(
        canvas, legend, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA
    )
    left = 0
    for panel in panels:
        canvas[legend_height:, left : left + panel.shape[1]] = panel
        left += panel.shape[1] + gap
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write montage: {output_path}")


def _save_examples(
    vis_dir,
    candidates,
    predictions,
    source_labels,
    source_images,
    source_names,
    class_groups,
    args,
):
    """Render typical examples per error type and write their manifest."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(vis_dir.glob("*.png")) + [vis_dir / "examples.csv"]:
        if stale.exists():
            stale.unlink()

    rows = []
    for kind in KIND_ORDER:
        selected = _top_examples(candidates[kind], args.vis_per_type)
        rendered = []
        for rank, (score, stem) in enumerate(selected, start=1):
            if stem not in source_images:
                raise FileNotFoundError(f"No source image for prediction stem '{stem}'")
            pred_names, pred_boxes, pred_confs = predictions[stem]
            gt_names, gt_boxes, gt_groups = _load_gt(source_labels[stem], source_names, class_groups)
            iou_matrix = pairwise_iou_xyxy(pred_boxes, gt_boxes)
            reported, kinds, _ = _classify_boxes(pred_confs, gt_groups, iou_matrix, args.vis_conf, args.iou_threshold)
            missed = np.ones(len(gt_names), dtype=bool)
            if len(reported):
                missed = iou_matrix[reported].max(axis=0) <= args.iou_threshold

            output_path = vis_dir / f"{kind}_{rank:02d}.png"
            _render_example(
                image_path=source_images[stem],
                output_path=output_path,
                kind=kind,
                pred_names=pred_names,
                pred_boxes=pred_boxes,
                pred_confs=pred_confs,
                gt_names=gt_names,
                gt_boxes=gt_boxes,
                gt_groups=gt_groups,
                iou_matrix=iou_matrix,
                threshold=args.vis_conf,
                iou_threshold=args.iou_threshold,
            )
            rendered.append(output_path)

            history_missed = _history_missed_indices(gt_groups, missed)
            # Describe the example by its most confident instance of the selected type.
            if kind == "miss":
                rows.append(
                    {
                        "type": kind,
                        "image": stem,
                        "confidence": "",
                        "pred_class": "",
                        "gt_class": "",
                        "n_miss_history": len(history_missed),
                        "missed_history_classes": _missed_class_summary(gt_names, history_missed),
                    }
                )
                continue
            boxes_of_kind = [
                (float(pred_confs[index]), index) for row, index in enumerate(reported) if kinds[row] == kind
            ]
            if not boxes_of_kind:
                raise RuntimeError(f"Selected {kind} example '{stem}' no longer contains that error at {args.vis_conf}")
            confidence, pred_index = max(boxes_of_kind)
            gt_class = ""
            if kind in {"type2", "type3"}:
                gt_class = gt_names[int(iou_matrix[pred_index].argmax())]
            rows.append(
                {
                    "type": kind,
                    "image": stem,
                    "confidence": f"{confidence:.4f}",
                    "pred_class": pred_names[pred_index],
                    "gt_class": gt_class,
                    "n_miss_history": len(history_missed),
                    "missed_history_classes": _missed_class_summary(gt_names, history_missed),
                }
            )
        if rendered:
            _montage(rendered, vis_dir / f"{kind}_examples.png")

    if rows:
        _write_csv(vis_dir / "examples.csv", rows)
    LOGGER.info(f"Example visualizations saved to {vis_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction dump root (all/labels).")
    parser.add_argument("--model", type=Path, required=True, help="Teacher checkpoint labeling the predictions.")
    parser.add_argument("--stages", type=Path, nargs="+", required=True, help="Ordered per-stage dataset yamls.")
    parser.add_argument("--target_stage", type=int, required=True, help="1-based stage the teacher is applied to.")
    parser.add_argument("--source_cfg", type=Path, required=True, help="Dataset yaml with the full class space.")
    parser.add_argument("--split", default="train", help="Source split providing the GT.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for box matching.")
    parser.add_argument("--conf_min", type=float, default=0.05, help="First confidence threshold.")
    parser.add_argument("--conf_max", type=float, default=0.95, help="Last confidence threshold.")
    parser.add_argument("--conf_step", type=float, default=0.05, help="Confidence threshold step.")
    parser.add_argument("--vis_dir", type=Path, default=None, help="Directory for example visualizations.")
    parser.add_argument(
        "--vis_conf",
        type=float,
        default=0.25,
        help="Confidence threshold used to pick and draw examples (Ultralytics default: 0.25).",
    )
    parser.add_argument("--vis_per_type", type=int, default=3, help="Number of examples per error type.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for the output tables and plot.")
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou_threshold must be in (0, 1]")
    if not 0.0 < args.conf_min <= args.conf_max < 1.0:
        parser.error("--conf_min / --conf_max must satisfy 0 < min <= max < 1")
    if args.conf_step <= 0:
        parser.error("--conf_step must be positive")
    if not 0.0 < args.vis_conf < 1.0:
        parser.error("--vis_conf must be in (0, 1)")
    if args.vis_per_type < 1:
        parser.error("--vis_per_type must be positive")
    return args


def main() -> None:
    """Run the teacher error analysis."""
    args = parse_args()

    stage_names = stage_class_names(args.stages)
    if not 1 <= args.target_stage <= len(stage_names):
        raise ValueError(f"--target_stage must be in [1, {len(stage_names)}], got {args.target_stage}")
    target_index = args.target_stage - 1
    history_names = {name for names in stage_names[:target_index] for name in names.values()}
    target_names = set(stage_names[target_index].values())
    future_names = {name for names in stage_names[target_index + 1 :] for name in names.values()}
    if not history_names:
        raise ValueError(f"Teacher for stage {args.target_stage} has no history stage to transfer from")
    class_groups = {name: "history" for name in history_names}
    class_groups.update({name: "target" for name in target_names})
    class_groups.update({name: "future" for name in future_names})

    source_config = YAML().load(args.source_cfg)
    if "names" not in source_config:
        raise KeyError(f"Source dataset config has no 'names': {args.source_cfg}")
    source_names = normalize_names(source_config["names"], source=f"source dataset '{args.source_cfg}'")
    source_image_dirs, source_label_dirs = resolve_split_roots(args.source_cfg, source_config, args.split)
    source_labels = index_label_files(source_label_dirs)
    source_images = _index_images(source_image_dirs) if args.vis_dir is not None else {}

    model = YOLO(args.model)
    model_names = normalize_names(model.names, source=f"model '{args.model}'")
    model_class_names = set(model_names.values())
    if model_class_names != history_names:
        raise ValueError(
            "Teacher classes must equal the history stages exactly: "
            f"missing={sorted(history_names - model_class_names)}, "
            f"unexpected={sorted(model_class_names - history_names)}"
        )

    predictions = _load_predictions(Path(args.predictions) / "all" / "labels", model_names)
    missing = [stem for stem in predictions if stem not in source_labels]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} prediction images have no full-space '{args.split}' label, e.g. '{missing[0]}'"
        )

    thresholds = []
    value = args.conf_min
    while value <= args.conf_max + 1e-9:
        thresholds.append(round(value, 6))
        value += args.conf_step
    # Always report the Ultralytics default inference threshold when it is in range.
    if args.conf_min <= 0.25 <= args.conf_max and not any(abs(value - 0.25) < 1e-9 for value in thresholds):
        thresholds = sorted([*thresholds, 0.25])

    totals = defaultdict(int)
    candidates = defaultdict(list)
    per_threshold = {threshold: defaultdict(int) for threshold in thresholds}
    for index, (stem, (pred_names, pred_boxes, pred_confs)) in enumerate(sorted(predictions.items()), start=1):
        gt_names, gt_boxes, gt_groups = _load_gt(source_labels[stem], source_names, class_groups)
        iou_matrix = pairwise_iou_xyxy(pred_boxes, gt_boxes)
        totals["images"] += 1
        totals["gt"] += len(gt_names)
        for group in ("history", "target", "future"):
            totals[f"gt_{group}"] += gt_groups.count(group)
        for threshold in thresholds:
            counts = _count_image(pred_confs, gt_groups, iou_matrix, threshold, args.iou_threshold)
            for key, count in counts.items():
                per_threshold[threshold][key] += count
        if args.vis_dir is not None:
            reported, kinds, _ = _classify_boxes(
                pred_confs, gt_groups, iou_matrix, args.vis_conf, args.iou_threshold
            )
            for row, pred_index in enumerate(reported):
                kind = kinds[row]
                if kind not in {"type1", "type2", "type3"}:
                    continue
                if kind == "type1" and iou_matrix.shape[1] and iou_matrix[pred_index].max() > 0.0:
                    # Prefer examples that do not touch any GT box at all.
                    continue
                candidates[kind].append((float(pred_confs[pred_index]), stem))
            if len(reported):
                missed = iou_matrix[reported].max(axis=0) <= args.iou_threshold
            else:
                missed = np.ones(len(gt_names), dtype=bool)
            missed_history = len(_history_missed_indices(gt_groups, missed))
            if missed_history:
                candidates["miss"].append((float(missed_history), stem))
        if index % 1000 == 0:
            LOGGER.info(f"Processed {index}/{len(predictions)} images")

    rows = []
    for threshold in thresholds:
        counts = per_threshold[threshold]
        classified = counts["type1"] + counts["type2"] + counts["type3"] + counts["history"]
        if classified != counts["pred"]:
            raise RuntimeError(
                f"Threshold {threshold}: {classified} classified boxes do not match {counts['pred']} reported boxes"
            )
        rows.append(
            {
                "confidence_threshold": threshold,
                "n_images": totals["images"],
                "n_gt": totals["gt"],
                "n_history_gt": totals["gt_history"],
                "n_pred": counts["pred"],
                "n_type1_background_to_history": counts["type1"],
                "n_type2_target_to_history": counts["type2"],
                "n_type3_future_to_history": counts["type3"],
                "n_miss_history": counts["miss_history"],
                "type1_rate": _safe_rate(counts["type1"], counts["pred"]),
                "type2_rate": _safe_rate(counts["type2"], counts["pred"]),
                "type3_rate": _safe_rate(counts["type3"], counts["pred"]),
                "miss_rate": _safe_rate(counts["miss_history"], totals["gt_history"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "error_rates.csv", rows)
    _write_markdown(args.output_dir / "error_rates.md", rows)
    _plot(args.output_dir / "error_rates.png", rows, args.target_stage, totals["gt_future"] > 0)

    if args.vis_dir is not None:
        _save_examples(
            vis_dir=args.vis_dir,
            candidates=candidates,
            predictions=predictions,
            source_labels=source_labels,
            source_images=source_images,
            source_names=source_names,
            class_groups=class_groups,
            args=args,
        )

    summary = [
        "Teacher high-confidence error analysis",
        f"  predictions : {args.predictions}",
        f"  model       : {args.model}",
        f"  source      : {args.source_cfg} (split={args.split})",
        f"  target stage: {args.target_stage} of {len(stage_names)}",
        f"  history     : {len(history_names)} classes",
        f"  target      : {len(target_names)} classes",
        f"  future      : {len(future_names)} classes",
        f"  IoU         : {args.iou_threshold}",
        f"  confidence  : {thresholds[0]}..{thresholds[-1]} step {args.conf_step}",
        f"  images      : {totals['images']}",
        (
            f"  GT objects  : {totals['gt']} (history {totals['gt_history']}, "
            f"target {totals['gt_target']}, future {totals['gt_future']})"
        ),
        "",
        "Definitions:",
        "  type 1 = reported box whose largest GT IoU is not above the threshold (background -> history)",
        "  type 2 = reported box whose largest GT IoU is above the threshold and that GT is a target-stage object",
        "  type 3 = reported box whose largest GT IoU is above the threshold and that GT is a later-stage object",
        "  miss   = history-class GT object with no reported box above the IoU threshold",
        "  type1_rate = type 1 boxes / reported boxes",
        "  type2_rate = type 2 boxes / reported boxes",
        "  type3_rate = type 3 boxes / reported boxes",
        "  miss_rate = missed history-class GT objects / history-class GT objects",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
