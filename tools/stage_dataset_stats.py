"""Per-stage class distribution and sample overlap of a class-incremental dataset.

The stage datasets are built by absorbing every sample that contains at least one
class of the current stage, so one sample can belong to several stages. This tool
reports two tables over the created stage datasets:

1. ``stage_class_counts``: object count of every class in every stage, counted
   on the full class-space annotation (all GT boxes of an image, including boxes
   of classes owned by other stages).
2. ``stage_sample_overlap``: number of training samples of each stage on the
   diagonal and the number of samples shared by two stages off the diagonal.

Usage:
    $ python tools/stage_dataset_stats.py \
        --stages data/VOC_10+10/task_1_cls_10/dataset.yaml \
                 data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --source_cfg data/VOC-YOLO/VOC.yaml \
        --split train \
        --output_dir runs/<run>/motivation/stage_stats

Arguments:
    --stages: Ordered per-stage dataset yamls (the incremental train sequence).
    --source_cfg: Dataset yaml holding the full class-space annotation.
    --split: Split of every stage dataset to read (default: train).
    --output_dir: Directory for the CSV / Markdown tables.

Outputs:
    stage_class_counts.csv / .md
    stage_sample_overlap.csv / .md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, YAML

from utils import index_label_files, normalize_names, parse_label_line, resolve_split_roots, stage_class_names


def _list_image_stems(image_dirs):
    """Return image stems of one split, rejecting duplicate stems across roots."""
    stems = []
    seen = {}
    for image_dir in image_dirs:
        for image_path in sorted(Path(image_dir).iterdir()):
            if not image_path.is_file() or image_path.suffix[1:].lower() not in IMG_FORMATS:
                continue
            previous = seen.setdefault(image_path.stem, image_path)
            if previous != image_path:
                raise ValueError(f"Duplicate image stem '{image_path.stem}': '{previous}' and '{image_path}'")
            stems.append(image_path.stem)
    return stems


def _count_objects(label_path, class_names):
    """Count GT objects per class id in one label file."""
    counts = defaultdict(int)
    for line_number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            raise ValueError(f"Invalid YOLO label at '{label_path}:{line_number}': {line!r}")
        class_id = parsed[0]
        if class_id not in class_names:
            raise ValueError(
                f"Class id {class_id} at '{label_path}:{line_number}' is outside the source class space "
                f"{sorted(class_names)}"
            )
        counts[class_id] += 1
    return counts


def _align_classes(stage_names, source_names):
    """Map stage classes onto source class ids, returning per-stage source ids and row order."""
    source_id_by_name = {}
    for source_id, class_name in source_names.items():
        previous = source_id_by_name.setdefault(class_name, source_id)
        if previous != source_id:
            raise ValueError(f"Source class name '{class_name}' is used by ids {previous} and {source_id}")

    stage_source_ids = []
    for stage_index, names in enumerate(stage_names, start=1):
        stage_ids = []
        for local_id, class_name in names.items():
            if class_name not in source_id_by_name:
                raise ValueError(
                    f"Stage {stage_index} class '{class_name}' (local id {local_id}) is not in the source dataset"
                )
            stage_ids.append(source_id_by_name[class_name])
        stage_source_ids.append(stage_ids)

    assigned = {source_id for ids in stage_source_ids for source_id in ids}
    unassigned = sorted(set(source_names) - assigned)
    if unassigned:
        raise ValueError(
            f"Source classes {unassigned} belong to no stage; a class-incremental split must cover them"
        )
    return stage_source_ids


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _write_class_table_markdown(path, class_names, stage_source_ids, counts, num_stages):
    """Write the class-count table with one block per stage."""
    header = ["Class", *(f"Stage {stage}" for stage in range(1, num_stages + 1))]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * num_stages) + "|",
    ]
    for stage_index, source_ids in enumerate(stage_source_ids, start=1):
        lines.append("| " + " | ".join([f"**Stage {stage_index}**", *[""] * num_stages]) + " |")
        for source_id in source_ids:
            cells = [str(counts[stage][source_id]) for stage in range(num_stages)]
            lines.append("| " + " | ".join([class_names[source_id], *cells]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overlap_markdown(path, split, num_stages, totals, overlaps):
    header = ["Stage", f"Total {split} samples", *(f"Stage {stage}" for stage in range(1, num_stages + 1))]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---", "---:"] + ["---:"] * num_stages) + "|",
    ]
    for stage in range(num_stages):
        cells = [str(overlaps[stage][other]) for other in range(num_stages)]
        lines.append("| " + " | ".join([f"Stage {stage + 1}", str(totals[stage]), *cells]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stages",
        type=Path,
        nargs="+",
        required=True,
        help="Ordered per-stage dataset yamls (incremental train sequence).",
    )
    parser.add_argument("--source_cfg", type=Path, required=True, help="Dataset yaml with the full class space.")
    parser.add_argument("--split", default="train", help="Split of every stage dataset to read.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for the output tables.")
    return parser.parse_args()


def main() -> None:
    """Build the class-count and sample-overlap tables."""
    args = parse_args()

    source_config = YAML().load(args.source_cfg)
    if "names" not in source_config:
        raise KeyError(f"Source dataset config has no 'names': {args.source_cfg}")
    source_names = normalize_names(source_config["names"], source=f"source dataset '{args.source_cfg}'")
    _, source_label_dirs = resolve_split_roots(args.source_cfg, source_config, args.split)
    source_labels = index_label_files(source_label_dirs)

    stage_names = stage_class_names(args.stages)
    stage_source_ids = _align_classes(stage_names, source_names)
    num_stages = len(stage_names)

    stage_samples = []
    stage_stems = []
    for stage_index, stage_yaml in enumerate(args.stages, start=1):
        stage_config = YAML().load(stage_yaml)
        image_dirs, _ = resolve_split_roots(stage_yaml, stage_config, args.split)
        stems = _list_image_stems(image_dirs)
        if not stems:
            raise ValueError(f"Stage {stage_index} dataset '{stage_yaml}' has no '{args.split}' images")
        missing = [stem for stem in stems if stem not in source_labels]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} stage {stage_index} '{args.split}' images have no full-space label in "
                f"{[str(path) for path in source_label_dirs]}, e.g. '{missing[0]}'"
            )
        stage_stems.append(stems)
        LOGGER.info(f"Stage {stage_index}: {len(stems)} '{args.split}' images")

    counts = [[0] * len(source_names) for _ in range(num_stages)]
    for stage_index, stems in enumerate(stage_stems):
        for stem in stems:
            for class_id, value in _count_objects(source_labels[stem], source_names).items():
                counts[stage_index][class_id] += value

    overlaps = [[0] * num_stages for _ in range(num_stages)]
    stem_sets = [set(stems) for stems in stage_stems]
    for left in range(num_stages):
        for right in range(num_stages):
            overlaps[left][right] = len(stem_sets[left] & stem_sets[right])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_rows = []
    for stage_index, source_ids in enumerate(stage_source_ids, start=1):
        for source_id in source_ids:
            class_rows.append(
                [
                    source_id,
                    source_names[source_id],
                    f"stage_{stage_index}",
                    *(counts[stage][source_id] for stage in range(num_stages)),
                ]
            )
    class_header = ["class_id", "class", "stage", *(f"stage_{stage}" for stage in range(1, num_stages + 1))]
    _write_csv(args.output_dir / "stage_class_counts.csv", class_header, class_rows)
    _write_class_table_markdown(
        args.output_dir / "stage_class_counts.md",
        source_names,
        stage_source_ids,
        counts,
        num_stages,
    )

    overlap_rows = []
    for stage in range(num_stages):
        overlap_rows.append(
            [
                f"stage_{stage + 1}",
                len(stage_stems[stage]),
                *(overlaps[stage][other] for other in range(num_stages)),
            ]
        )
    overlap_header = ["stage", "total_samples", *(f"stage_{other}" for other in range(1, num_stages + 1))]
    _write_csv(args.output_dir / "stage_sample_overlap.csv", overlap_header, overlap_rows)
    _write_overlap_markdown(
        args.output_dir / "stage_sample_overlap.md",
        args.split,
        num_stages,
        [len(stems) for stems in stage_stems],
        overlaps,
    )

    LOGGER.info(f"Stage class counts saved to {args.output_dir / 'stage_class_counts.csv'}")
    LOGGER.info(f"Stage sample overlap saved to {args.output_dir / 'stage_sample_overlap.csv'}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
