"""Incremental-model high-confidence error analysis for class-incremental detection.

The model produced by an incremental stage is run on the images of an earlier
(history) stage, which keeps the full class-space ground truth. Because the model
has not seen that image distribution while learning the increment, it reports
history-task content in the wrong class space. As a function of the reported
confidence threshold, this tool measures:

- type 1: background -> incremental false positive (best GT IoU not above the threshold)
- type 2: background -> history false positive (best GT IoU not above the threshold)
- type 3: history object reported as an incremental class (best GT IoU above the
  threshold and that GT is a history-stage object)
- type 4: incremental object reported as a history class (same, for an
  incremental-stage GT object)
- miss history: history-class GT object with no box above the IoU threshold
- miss incremental: incremental-class GT object with no box above the IoU threshold

Every reported box is classified by the GT box it overlaps most; a box with no GT
above the IoU threshold is a background false positive, split by whether it names
a history or an incremental class. The false-positive rates use the number of
reported boxes at the threshold as denominator; the miss rates use the number of
GT objects of their own group.

When ``--vis_dir`` is given, the tool also renders typical examples of every error
type at ``--vis_conf`` (default 0.25, the Ultralytics inference default): GT boxes
are gray, matched boxes green, and the four false-positive types keep the colors
listed in the montage legend.

Usage:
    $ python tools/incremental_error_analysis.py \
        --predictions runs/<run>/incremental_errors/task-2-on-task-1/predictions \
        --model runs/<run>/task-2/best.pt \
        --stages data/VOC_10+10/task_1_cls_10/dataset.yaml \
                 data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --history_stages 1 \
        --source_cfg data/VOC-YOLO/VOC.yaml \
        --split test \
        --vis_dir runs/<run>/incremental_errors/task-2-on-task-1/examples \
        --output_dir runs/<run>/incremental_errors/task-2-on-task-1

Arguments:
    --predictions: Dump root from tools/predict.py; its ``all/labels`` directory
        holds ``class x y w h confidence`` lines per image.
    --model: Incremental checkpoint whose class names label the predictions.
    --stages: Ordered per-stage dataset yamls.
    --history_stages: 1-based stages whose classes count as history (default: 1);
        the remaining stages provide the incremental classes.
    --source_cfg: Dataset yaml holding the full class-space GT annotation.
    --split: Split of the source dataset providing the GT (default: test).
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
from utils import (
    index_label_files,
    normalize_names,
    pairwise_iou_xyxy,
    parse_label_line,
    resolve_split_roots,
    stage_class_names,
)

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.plotting import Annotator

KIND_ORDER = ("type1", "type2", "type3", "type4", "type5", "miss_history", "miss_incremental")
KIND_COLORS = {
    "type1": (255, 80, 0),  # BGR blue: background -> incremental
    "type2": (0, 0, 255),  # BGR red: background -> history
    "type3": (255, 0, 255),  # BGR magenta: history object -> incremental class
    "type4": (0, 165, 255),  # BGR orange: incremental object -> history class
    "type5": (0, 255, 255),  # BGR yellow: future-stage object -> history class
    "forward": (128, 128, 128),  # BGR gray: future-stage object -> incremental class
    "matched": (0, 170, 0),  # BGR green
}
GT_COLOR = (185, 185, 185)
KIND_LABELS = {
    "type1": "background -> incremental",
    "type2": "background -> history",
    "type3": "history -> incremental class",
    "type4": "incremental -> history class",
    "type5": "future stage -> history class",
    "miss_history": "missed history object",
    "miss_incremental": "missed incremental object",
}


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
            raise ValueError(f"GT class '{class_name}' at '{label_path}:{line_number}' belongs to no stage group")
        names.append(class_name)
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        groups.append(class_groups[class_name])
    return names, np.asarray(boxes, dtype=np.float32).reshape(-1, 4), groups


def _classify_boxes(pred_names, pred_confs, gt_groups, iou_matrix, threshold, iou_threshold, class_groups):
    """Classify the reported boxes of one image at a confidence threshold.

    The history/target split follows the predicted class: a box that matches no GT
    is a type-1 false positive when it names an incremental class and a type-2 one
    when it names a history class. A box that does match a GT is classified by the
    GT/prediction group pair, so ``type5`` (a future-stage object reported as a
    history class) only appears when ``--future_stages`` is given.

    Returns:
        tuple[np.ndarray, list[str], list[int | None]]: Reported prediction
        indices, one kind per reported box (``type1``..``type5``, ``forward`` or
        ``matched``), and the matched GT index per box (``None`` for background
        false positives).
    """
    reported = np.where(pred_confs >= threshold)[0]
    if len(reported) == 0:
        return reported, [], []
    if len(gt_groups) == 0:
        background_kind = {"incremental": "type1", "history": "type2", "future": "forward"}
        return (
            reported,
            [background_kind[class_groups[pred_names[index]]] for index in reported],
            [None] * len(reported),
        )

    ious = iou_matrix[reported]  # (num_reported, num_gt)
    best_gt = ious.argmax(axis=1)
    best_iou = ious[np.arange(len(reported)), best_gt]
    kinds = []
    gt_refs = []
    for row, gt_index in enumerate(best_gt):
        pred_group = class_groups[pred_names[reported[row]]]
        if best_iou[row] <= iou_threshold:
            kinds.append({"history": "type2", "incremental": "type1", "future": "forward"}[pred_group])
            gt_refs.append(None)
            continue
        gt_group = gt_groups[gt_index]
        if pred_group == gt_group:
            kinds.append("matched")
        elif gt_group == "history":
            kinds.append("type3")  # history object reported in a later stage's class space
        elif gt_group == "incremental":
            kinds.append("type4" if pred_group == "history" else "forward")
        else:  # future-stage GT: only "reported as a history class" is an error against history
            kinds.append("type5" if pred_group == "history" else "forward")
        gt_refs.append(int(gt_index))
    return reported, kinds, gt_refs


def _classify_gt_objects(pred_names, pred_confs, gt_groups, iou_matrix, threshold, iou_threshold, class_groups):
    """Classify every GT object by its highest-IoU reported box at a threshold.

    A GT object with no reported box above the IoU threshold is missed; otherwise
    its outcome is ``correct`` when the best box names a class of the GT's group,
    or ``<gt_group>_to_<pred_group>`` when it names another group. These counts
    therefore sum to the number of GT objects, unlike the box-level ``type*``
    counts where several boxes may match the same object.
    """
    counts = defaultdict(int)
    if not gt_groups:
        return counts
    reported = np.where(pred_confs >= threshold)[0]
    if len(reported) == 0:
        for group in gt_groups:
            counts[f"obj_miss_{group}"] += 1
        return counts

    ious = iou_matrix[reported]  # (num_reported, num_gt)
    best_reported = ious.argmax(axis=0)  # (num_gt,)
    best_iou = ious[best_reported, np.arange(len(gt_groups))]
    for gt_index, gt_group in enumerate(gt_groups):
        if best_iou[gt_index] <= iou_threshold:
            counts[f"obj_miss_{gt_group}"] += 1
            continue
        pred_group = class_groups[pred_names[reported[best_reported[gt_index]]]]
        if pred_group == gt_group:
            counts[f"obj_correct_{gt_group}"] += 1
        else:
            counts[f"obj_{gt_group}_to_{pred_group}"] += 1
    return counts


def _count_image(pred_names, pred_confs, gt_groups, iou_matrix, threshold, iou_threshold, class_groups):
    """Count reported boxes by error type and classify the GT objects of one image."""
    counts = defaultdict(int)
    reported, kinds, _ = _classify_boxes(
        pred_names, pred_confs, gt_groups, iou_matrix, threshold, iou_threshold, class_groups
    )
    counts["pred"] = len(reported)
    for kind in kinds:
        counts[kind] += 1
    for key, count in _classify_gt_objects(
        pred_names, pred_confs, gt_groups, iou_matrix, threshold, iou_threshold, class_groups
    ).items():
        counts[key] += count
    if len(reported) == 0:
        missed = np.ones(len(gt_groups), dtype=bool)
    else:
        missed = iou_matrix[reported].max(axis=0) <= iou_threshold
    for gt_index, group in enumerate(gt_groups):
        if missed[gt_index]:
            counts[f"miss_{group}"] += 1
    return counts


def _safe_rate(numerator, denominator):
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    return float(numerator) / denominator if denominator else 0.0


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows, has_future=False):
    columns = [
        ("confidence_threshold", "Conf"),
        ("n_pred", "Reported"),
        ("type1_rate", "Type 1 bg->inc"),
        ("type2_rate", "Type 2 bg->hist"),
        ("type3_rate", "Type 3 hist->inc"),
        ("type4_rate", "Type 4 inc->hist"),
        ("type5_rate", "Type 5 future->hist"),
        ("miss_history_rate", "Miss hist"),
        ("miss_incremental_rate", "Miss inc"),
        ("miss_future_rate", "Miss future"),
    ]
    if not has_future:
        columns = [column for column in columns if not column[0].endswith("future_rate")]
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


def _plot(path, rows, has_future=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["confidence_threshold"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    styles = {
        "type1_rate": ("o", "Type 1 background -> incremental"),
        "type2_rate": ("s", "Type 2 background -> history"),
        "type3_rate": ("D", "Type 3 history -> incremental class"),
        "type4_rate": ("v", "Type 4 incremental -> history class"),
        "miss_history_rate": ("^", "History miss"),
        "miss_incremental_rate": ("P", "Incremental miss"),
    }
    if has_future:
        styles = {
            "type1_rate": ("o", "Type 1 background -> incremental"),
            "type2_rate": ("s", "Type 2 background -> history"),
            "type3_rate": ("D", "Type 3 history -> incremental class"),
            "type5_rate": ("X", "Type 5 future stage -> history class"),
            "miss_history_rate": ("^", "History miss"),
            "miss_incremental_rate": ("P", "Incremental miss"),
            "miss_future_rate": ("*", "Future miss"),
        }
    for key, (marker, label) in styles.items():
        axis.plot(thresholds, [row[key] for row in rows], marker=marker, label=label)
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


def _top_examples(entries, limit):
    """Pick the highest-scoring images, one entry per image."""
    best = {}
    for score, stem in entries:
        if stem not in best or score > best[stem][0]:
            best[stem] = (score, stem)
    return sorted(best.values(), key=lambda item: (-item[0], item[1]))[:limit]


def _missed_summary(gt_names, missed_indices):
    """Return a deterministic ``class xcount`` summary over the given GT indices."""
    counts = defaultdict(int)
    for gt_index in missed_indices:
        counts[gt_names[gt_index]] += 1
    return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _render_example(
    image_path,
    output_path,
    gt_names,
    gt_boxes,
    pred_names,
    pred_boxes,
    pred_confs,
    boxes,
    highlight_gt=None,
    missed_gt=None,
):
    """Draw GT boxes, the selected reported boxes and the missed-GT strip of one image.

    Args:
        image_path (Path): Source image.
        output_path (Path): Rendered example destination.
        gt_names (list[str]): Class name per GT box.
        gt_boxes (np.ndarray): GT boxes with shape (M, 4) in normalized xyxy.
        pred_names (list[str]): Class name per reported box.
        pred_boxes (np.ndarray): Reported boxes with shape (N, 4) in normalized xyxy.
        pred_confs (np.ndarray): Reported confidences with shape (N,).
        boxes (list[tuple[int, str]]): Reported box index and kind to draw.
        highlight_gt (int | None): GT box labelled with its class name.
        missed_gt (list[int] | None): Missed GT indices summarized in the bottom strip.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]

    for gt_index, gt_box in enumerate(gt_boxes):
        x1, y1 = int(gt_box[0] * width), int(gt_box[1] * height)
        x2, y2 = int(gt_box[2] * width), int(gt_box[3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), GT_COLOR, 1)
        if gt_index != highlight_gt:
            continue
        label = f"GT {gt_names[gt_index]}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_x = max(0, min(width - label_width, x2 - label_width))
        text_y = y1 - 4 if y1 - 4 > label_height else y1 + label_height + 2
        cv2.rectangle(image, (text_x, text_y - label_height - 2), (text_x + label_width, text_y + 2), GT_COLOR, -1)
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    annotator = Annotator(image, line_width=2)
    for pred_index, kind in boxes:
        box = pred_boxes[pred_index]
        pixel_box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        label = f"{pred_names[pred_index]} {pred_confs[pred_index]:.2f} [{kind}]"
        annotator.box_label(pixel_box, label, color=KIND_COLORS[kind])
    result = annotator.result()

    if missed_gt is not None:
        strip_height = 52
        strip = np.full((strip_height, result.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(
            strip,
            f"missed GT: {len(missed_gt)}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            strip,
            f"classes: {_missed_summary(gt_names, missed_gt)}",
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        result = np.concatenate([result, strip], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write example image: {output_path}")


def _montage(paths, output_path, kind, height=360):
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
    legend = (
        f"{kind}: {KIND_LABELS[kind]} | GT gray | boxes: green matched, "
        "blue type1, red type2, magenta type3, orange type4"
    )
    cv2.putText(canvas, legend, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    left = 0
    for panel in panels:
        canvas[legend_height:, left : left + panel.shape[1]] = panel
        left += panel.shape[1] + gap
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write montage: {output_path}")


def _save_examples(vis_dir, candidates, predictions, source_labels, source_images, source_names, class_groups, args):
    """Render typical examples per error type and write their manifest."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(vis_dir.glob("*.png")) + [vis_dir / "examples.csv"]:
        if stale.exists():
            stale.unlink()

    rows = []
    for kind in KIND_ORDER:
        rendered = []
        for rank, (_, stem) in enumerate(_top_examples(candidates[kind], args.vis_per_type), start=1):
            if stem not in source_images:
                raise FileNotFoundError(f"No source image for prediction stem '{stem}'")
            pred_names, pred_boxes, pred_confs = predictions[stem]
            gt_names, gt_boxes, gt_groups = _load_gt(source_labels[stem], source_names, class_groups)
            iou_matrix = pairwise_iou_xyxy(pred_boxes, gt_boxes)
            reported, kinds, gt_refs = _classify_boxes(
                pred_names, pred_confs, gt_groups, iou_matrix, args.vis_conf, args.iou_threshold, class_groups
            )
            missed = (
                iou_matrix[reported].max(axis=0) <= args.iou_threshold
                if len(reported)
                else np.ones(len(gt_names), dtype=bool)
            )
            missed_indices = [
                gt_index
                for gt_index, flag in enumerate(missed)
                if flag and gt_groups[gt_index] == ("incremental" if kind == "miss_incremental" else "history")
            ]

            if kind.startswith("miss"):
                if not missed_indices:
                    raise RuntimeError(f"Selected {kind} example '{stem}' no longer has that error")
                boxes = [(pred_index, kinds[row]) for row, pred_index in enumerate(reported) if kinds[row] != "matched"]
                rows.append(
                    {
                        "type": kind,
                        "image": stem,
                        "confidence": "",
                        "pred_class": "",
                        "gt_class": "",
                        "n_missed": len(missed_indices),
                        "missed_classes": _missed_summary(gt_names, missed_indices),
                    }
                )
                highlight_gt = None
            else:
                rows_of_kind = [(row, pred_index) for row, pred_index in enumerate(reported) if kinds[row] == kind]
                if not rows_of_kind:
                    raise RuntimeError(
                        f"Selected {kind} example '{stem}' no longer contains that error at {args.vis_conf}"
                    )
                best_row, pred_index = max(rows_of_kind, key=lambda item: float(pred_confs[item[1]]))
                boxes = [(index, kinds[row]) for row, index in enumerate(reported)]
                highlight_gt = gt_refs[best_row]
                rows.append(
                    {
                        "type": kind,
                        "image": stem,
                        "confidence": f"{float(pred_confs[pred_index]):.4f}",
                        "pred_class": pred_names[pred_index],
                        "gt_class": gt_names[highlight_gt] if highlight_gt is not None else "",
                        "n_missed": len(missed_indices),
                        "missed_classes": _missed_summary(gt_names, missed_indices),
                    }
                )

            output_path = vis_dir / f"{kind}_{rank:02d}.png"
            _render_example(
                image_path=source_images[stem],
                output_path=output_path,
                gt_names=gt_names,
                gt_boxes=gt_boxes,
                pred_names=pred_names,
                pred_boxes=pred_boxes,
                pred_confs=pred_confs,
                boxes=boxes,
                highlight_gt=highlight_gt,
                missed_gt=missed_indices if kind.startswith("miss") else None,
            )
            rendered.append(output_path)
        if rendered:
            _montage(rendered, vis_dir / f"{kind}_examples.png", kind)

    if rows:
        _write_csv(vis_dir / "examples.csv", rows)
    LOGGER.info(f"Example visualizations saved to {vis_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction dump root (all/labels).")
    parser.add_argument("--model", type=Path, required=True, help="Incremental checkpoint labeling the predictions.")
    parser.add_argument("--stages", type=Path, nargs="+", required=True, help="Ordered per-stage dataset yamls.")
    parser.add_argument(
        "--history_stages",
        type=int,
        nargs="+",
        default=[1],
        help="1-based stages whose classes count as history (default: 1).",
    )
    parser.add_argument(
        "--future_stages",
        type=int,
        nargs="+",
        default=[],
        help="1-based stages treated as future (GT only): their objects reported as history classes form "
        "type 5, the multi-stage teacher-style error.",
    )
    parser.add_argument("--source_cfg", type=Path, required=True, help="Dataset yaml with the full class space.")
    parser.add_argument("--split", default="test", help="Source split providing the GT.")
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
    """Run the incremental-model error analysis."""
    args = parse_args()

    stage_names = stage_class_names(args.stages)
    history_ids = sorted(set(args.history_stages))
    if not history_ids:
        raise ValueError("--history_stages must name at least one stage")
    if history_ids[0] < 1 or history_ids[-1] > len(stage_names):
        raise ValueError(f"--history_stages must be within [1, {len(stage_names)}], got {history_ids}")
    future_ids = sorted(set(args.future_stages))
    if future_ids and (future_ids[0] < 1 or future_ids[-1] > len(stage_names)):
        raise ValueError(f"--future_stages must be within [1, {len(stage_names)}], got {future_ids}")
    if set(future_ids) & set(history_ids):
        raise ValueError(f"--future_stages {future_ids} overlap --history_stages {history_ids}")
    incremental_ids = [
        index for index in range(1, len(stage_names) + 1) if index not in history_ids and index not in future_ids
    ]
    if not incremental_ids:
        raise ValueError("--history_stages covers every stage, so no incremental class is left")

    history_classes = {name for stage in history_ids for name in stage_names[stage - 1].values()}
    incremental_classes = {name for stage in incremental_ids for name in stage_names[stage - 1].values()}
    future_classes = {name for stage in future_ids for name in stage_names[stage - 1].values()}
    class_groups = {name: "history" for name in history_classes}
    class_groups.update({name: "incremental" for name in incremental_classes})
    class_groups.update({name: "future" for name in future_classes})

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
    known = history_classes | incremental_classes | future_classes
    missing = sorted(history_classes - model_class_names)
    unexpected = sorted(model_class_names - known)
    if missing or unexpected:
        raise ValueError(
            "Model classes must cover every history class and stay inside the stage class space: "
            f"missing={missing}, unexpected={unexpected}"
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
        for group in ("history", "incremental", "future"):
            totals[f"gt_{group}"] += gt_groups.count(group)
        for threshold in thresholds:
            counts = _count_image(
                pred_names, pred_confs, gt_groups, iou_matrix, threshold, args.iou_threshold, class_groups
            )
            for key, count in counts.items():
                per_threshold[threshold][key] += count
        if args.vis_dir is not None:
            reported, kinds, _gt_refs = _classify_boxes(
                pred_names, pred_confs, gt_groups, iou_matrix, args.vis_conf, args.iou_threshold, class_groups
            )
            for row, pred_index in enumerate(reported):
                kind = kinds[row]
                if kind == "matched":
                    continue
                if kind in {"type1", "type2"} and iou_matrix.shape[1] and iou_matrix[pred_index].max() > 0.0:
                    # Prefer examples that do not touch any GT box at all.
                    continue
                candidates[kind].append((float(pred_confs[pred_index]), stem))
            missed = (
                iou_matrix[reported].max(axis=0) <= args.iou_threshold
                if len(reported)
                else np.ones(len(gt_names), dtype=bool)
            )
            for group in ("history", "incremental", "future"):
                missed_group = sum(
                    1 for gt_index, gt_group in enumerate(gt_groups) if missed[gt_index] and gt_group == group
                )
                if missed_group:
                    candidates[f"miss_{group}"].append((float(missed_group), stem))
        if index % 500 == 0:
            LOGGER.info(f"Processed {index}/{len(predictions)} images")

    rows = []
    for threshold in thresholds:
        counts = per_threshold[threshold]
        classified = (
            counts["type1"]
            + counts["type2"]
            + counts["type3"]
            + counts["type4"]
            + counts["type5"]
            + counts["forward"]
            + counts["matched"]
        )
        if classified != counts["pred"]:
            raise RuntimeError(
                f"Threshold {threshold}: {classified} classified boxes do not match {counts['pred']} reported boxes"
            )
        rows.append(
            {
                "confidence_threshold": threshold,
                "n_images": totals["images"],
                "n_gt": totals["gt"],
                "n_gt_history": totals["gt_history"],
                "n_gt_incremental": totals["gt_incremental"],
                "n_gt_future": totals["gt_future"],
                "n_pred": counts["pred"],
                "n_type1_background_to_incremental": counts["type1"],
                "n_type2_background_to_history": counts["type2"],
                "n_type3_history_to_incremental": counts["type3"],
                "n_type4_incremental_to_history": counts["type4"],
                "n_type5_future_to_history": counts["type5"],
                "n_forward": counts["forward"],
                "n_matched": counts["matched"],
                "n_miss_history": counts["miss_history"],
                "n_miss_incremental": counts["miss_incremental"],
                "n_miss_future": counts["miss_future"],
                "n_obj_correct_history": counts["obj_correct_history"],
                "n_obj_history_to_incremental": counts["obj_history_to_incremental"],
                "n_obj_history_to_future": counts["obj_history_to_future"],
                "n_obj_miss_history": counts["obj_miss_history"],
                "n_obj_correct_incremental": counts["obj_correct_incremental"],
                "n_obj_incremental_to_history": counts["obj_incremental_to_history"],
                "n_obj_incremental_to_future": counts["obj_incremental_to_future"],
                "n_obj_miss_incremental": counts["obj_miss_incremental"],
                "n_obj_correct_future": counts["obj_correct_future"],
                "n_obj_future_to_history": counts["obj_future_to_history"],
                "n_obj_future_to_incremental": counts["obj_future_to_incremental"],
                "n_obj_miss_future": counts["obj_miss_future"],
                "type1_rate": _safe_rate(counts["type1"], counts["pred"]),
                "type2_rate": _safe_rate(counts["type2"], counts["pred"]),
                "type3_rate": _safe_rate(counts["type3"], counts["pred"]),
                "type4_rate": _safe_rate(counts["type4"], counts["pred"]),
                "type5_rate": _safe_rate(counts["type5"], counts["pred"]),
                "type3_object_rate": _safe_rate(counts["type3"], totals["gt_history"]),
                "type4_object_rate": _safe_rate(counts["type4"], totals["gt_incremental"]),
                "type5_object_rate": _safe_rate(counts["type5"], totals["gt_future"]),
                "miss_history_rate": _safe_rate(counts["miss_history"], totals["gt_history"]),
                "miss_incremental_rate": _safe_rate(counts["miss_incremental"], totals["gt_incremental"]),
                "miss_future_rate": _safe_rate(counts["miss_future"], totals["gt_future"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "error_rates.csv", rows)
    has_future = totals["gt_future"] > 0
    _write_markdown(args.output_dir / "error_rates.md", rows, has_future=has_future)
    _plot(args.output_dir / "error_rates.png", rows, has_future=has_future)

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
        "Incremental-model high-confidence error analysis",
        f"  predictions     : {args.predictions}",
        f"  model           : {args.model}",
        f"  source          : {args.source_cfg} (split={args.split})",
        f"  stages          : {len(stage_names)}",
        f"  history stages  : {history_ids}",
        f"  incremental     : {incremental_ids}",
        f"  future stages   : {future_ids if future_ids else 'none'}",
        f"  history classes : {len(history_classes)}",
        f"  incremental cls : {len(incremental_classes)}",
        f"  IoU             : {args.iou_threshold}",
        f"  confidence      : {thresholds[0]}..{thresholds[-1]} step {args.conf_step}",
        f"  images          : {totals['images']}",
        (
            f"  GT objects      : {totals['gt']} (history {totals['gt_history']}, "
            f"incremental {totals['gt_incremental']}, future {totals['gt_future']})"
        ),
        "",
        "Definitions:",
        "  type 1 = background box named as an incremental class (largest GT IoU not above the threshold)",
        "  type 2 = background box named as a history class (largest GT IoU not above the threshold)",
        "  type 3 = box matched to a history GT object but named as an incremental class",
        "  type 4 = box matched to an incremental GT object but named as a history class",
        "  type 5 = box matched to a future-stage GT object but named as a history class",
        "  forward = box matched to a future-stage GT object but named as an incremental class",
        "  matched = box matched to a GT object of its own class group",
        "  miss history = history-class GT object with no reported box above the IoU threshold",
        "  miss incremental = incremental-class GT object with no reported box above the IoU threshold",
        "  miss future = future-stage GT object with no reported box above the IoU threshold",
        "  type1..4_rate = boxes of the type / reported boxes (the *object* rates in the CSV use the",
        "                  corresponding GT object count as denominator)",
        "  miss_*_rate = missed GT objects of the group / GT objects of the group",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
