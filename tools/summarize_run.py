"""Summarize one incremental run into a single plain-text report.

The report collects everything needed to read a finished run offline: the per-task and
cumulative evaluation matrices, the final-model Old/New/All figures, the per-stage
aggregation, the patch-prototype PCA reconstruction deviations of every stage sample
set, and the file inventory of the underlying CSVs.

Usage:
    python tools/summarize_run.py --run runs/<run> --output runs/<run>/summary.txt
"""

from __future__ import annotations

import argparse
import csv
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from ultralytics.utils import YAML

MATRIX_KINDS = (("individual", "Task"), ("cumulative", "Cumulative"))


def read_csv(path):
    """Return the rows of a CSV file, or None when it does not exist."""
    if not path.exists():
        return None
    with open(path) as handle:
        return list(csv.DictReader(handle))


def read_matrix(run_dir, kind):
    """Return ``{model_task: {eval_task: mAP50-95}}`` for one evaluation matrix."""
    rows = read_csv(run_dir / "evaluation_results" / f"{kind}_datasets_eval.csv")
    if rows is None:
        return None
    matrix = {}
    for row in rows:
        model = int(row["Model_Task"].split("_")[1])
        values = {}
        for key, value in row.items():
            if key.startswith(("Task_", "Cumulative_")) and "mAP50-95" in key:
                values[int(key.split("_")[1])] = None if value in ("N/A", "") else float(value)
        matrix[model] = values
    return matrix


def mean(values):
    """Return the mean over the non-empty values, or None."""
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def cell(value):
    """Format one metric as a percentage with one decimal."""
    return "  N/A" if value is None else f"{value * 100:5.1f}"


def format_matrix(matrix, column_prefix, tasks):
    """Format one matrix as a fixed-width table plus the final model's Old/New/All."""
    lines = []
    header = f"{'model':>7} |" + "".join(f"{column_prefix}{index:>3} " for index in range(1, tasks + 1))
    lines.append(header)
    lines.append("-" * len(header))
    for model in sorted(matrix):
        cells = "".join(f"{cell(matrix[model].get(index))} " for index in range(1, tasks + 1))
        lines.append(f"{model:>7} |{cells}")
    final = max(matrix)
    per_task = [matrix[final].get(index) for index in range(1, tasks + 1)]
    old = mean(per_task[:-1])
    lines.append("")
    lines.append(
        f"final model (task {final}): "
        + " ".join(f"T{index}={cell(value)}" for index, value in enumerate(per_task, start=1))
        + f" | Old={cell(old)} New={cell(per_task[-1])} All={cell(mean(per_task))}"
    )
    return lines


def stage_table(run_dir):
    """Read the per-stage aggregation produced by tools/stage_task_map.py."""
    path = run_dir / "evaluation_results" / "stage_mAP_sequence.csv"
    rows = read_csv(path)
    if not rows:
        return []
    header = list(rows[0])
    lines = [" | ".join(header), "-" * 80]
    for row in rows:
        lines.append(" | ".join(str(row[key]) for key in header))
    return lines


def reconstruction_table(run_dir):
    """Aggregate the per-stage PCA reconstruction CSVs over classes."""
    directory = run_dir / "pca_reconstruction"
    if not directory.is_dir():
        return []
    lines = [
        f"{'stage':<22}{'level':>6}{'dim':>6}{'n':>6}{'E_feat':>10}{'E_cls':>10}{'E_reg':>10}",
        "-" * 70,
    ]
    for path in sorted(directory.glob("*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        aggregated = {}
        for row in rows:
            if row["class"] != "all":
                continue
            key = (int(row["level"]), int(row["dim"]))
            aggregated[key] = (int(row["n"]), float(row["E_feat"]), float(row["E_cls"]), float(row["E_reg"]))
        for (level, dim), (count, e_feat, e_cls, e_reg) in sorted(aggregated.items()):
            lines.append(
                f"{path.stem:<22}{level:>6}{dim:>6}{count:>6}{e_feat:>10.4f}{e_cls:>10.5f}{e_reg:>10.5f}"
            )
    return lines


def similarity_table(run_dir):
    """Return the prototype-similarity log lines if the run kept one."""
    path = run_dir / "pca_reconstruction" / "similarity.txt"
    if not path.exists():
        return []
    return path.read_text().strip().splitlines()


def gpu_name():
    """Return the first GPU name, or ``unknown``."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def training_protocol(run_dir):
    """Return the training settings of the last task of the run."""
    tasks = sorted((path for path in run_dir.glob("task-*") if path.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not tasks:
        return []
    args_path = tasks[-1] / "train" / "args.yaml"
    if not args_path.exists():
        return []
    args = YAML.load(args_path)
    keys = (
        "epochs",
        "batch",
        "imgsz",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "mosaic",
        "close_mosaic",
        "end2end",
        "agnostic_nms",
        "espreg",
        "espreg_weight_mode",
        "espreg_form",
        "espreg_scope",
        "espreg_loss_weight",
        "prad_cls_loss_weight",
        "prad_reg_loss_weight",
        "prad_bg_weight",
        "prad_proto_threshold",
        "prad_fg_threshold",
        "prd",
        "prd_samples",
        "prd_loss_weight",
        "prd_reg_loss_weight",
        "patch_prototypes",
        "lwf",
        "lwf_mode",
    )
    lines = [f"{key:<24}= {args.get(key)}" for key in keys if key in args]
    prototype = args.get("patch_prototypes")
    if prototype and Path(prototype).exists():
        artifact = Path(prototype)
        lines.append(f"{'patch_prototypes_size':<24}= {artifact.stat().st_size / 1e6:.1f} MB")
    return lines


def inventory(run_dir):
    """List the CSV artifacts of the run."""
    lines = []
    for path in sorted(run_dir.rglob("*.csv")):
        if "train" in path.parts:
            continue
        lines.append(f"  {path.relative_to(run_dir)}  ({path.stat().st_size} bytes)")
    return lines


def main():
    """Write the summary of one run."""
    parser = argparse.ArgumentParser(description="Summarize one incremental run into a text report")
    parser.add_argument("--run", required=True, help="Run directory holding task-k/best.pt")
    parser.add_argument("--output", default=None, help="Output text path (default runs/<run>/summary.txt)")
    parser.add_argument("--tasks", type=int, default=None, help="Number of tasks (default: max model task)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    output = Path(args.output) if args.output else run_dir / "summary.txt"

    individual = read_matrix(run_dir, "individual")
    tasks = args.tasks or (max(individual) if individual else 0)
    lines = [
        "=" * 78,
        "INCREMENTAL RUN SUMMARY",
        "=" * 78,
        f"run          : {run_dir}",
        f"generated    : {datetime.now().isoformat(timespec='seconds')}",
        f"host         : {socket.gethostname()} ({platform.platform()})",
        f"gpu          : {gpu_name()}",
        f"tasks        : {tasks}",
        "",
        "--- training protocol (last task) ---",
    ]
    lines += training_protocol(run_dir) or ["  (no args.yaml found)"]

    for kind, prefix in MATRIX_KINDS:
        matrix = read_matrix(run_dir, kind)
        lines.append("")
        lines.append(f"--- {kind} mAP50-95 matrix ---")
        if matrix is None:
            lines.append(f"  N/A (evaluation_results/{kind}_datasets_eval.csv missing)")
            continue
        lines += format_matrix(matrix, prefix, tasks)

    lines.append("")
    lines.append("--- per-stage mAP (stage_task_map) ---")
    lines += stage_table(run_dir) or ["  N/A"]

    lines.append("")
    lines.append("--- patch PCA reconstruction deviation per stage sample set ---")
    lines.append("  E_feat = relative patch L2 error, E_cls = sum |dp| over classes, E_reg = SmoothL1")
    lines += reconstruction_table(run_dir) or ["  N/A (no pca_reconstruction/*.csv)"]

    similarity = similarity_table(run_dir)
    if similarity:
        lines.append("")
        lines.append("--- prototype similarity of in-distribution patches ---")
        lines += similarity

    lines.append("")
    lines.append("--- artifact inventory ---")
    lines += inventory(run_dir) or ["  (no CSV found)"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
