#!/usr/bin/env python3
"""417-ir incremental campaign: GPU launch plan and result summary.

The experiment matrix lives here once and is consumed by both sides of the
campaign:

* ``plan`` prints the run queue of one scheduler (scripts/417-ir/run_*.sh)
* ``report`` prints the CIL and DIL result tables on the terminal
  (scripts/417-ir/get_results.sh)

Run directory layout (``--root``, default ``runs/417-ir``)::

    <root>/CIL/<experiment>/<method>/
    <root>/DIL/<experiment>/<method>/

where ``<experiment>`` is the experiment directory under
``scripts/417-ir/{CIL,DIL}/``. CIL runs ``naive`` and
``pseudo_label+espreg+replay``; DIL runs ``naive`` and ``espreg+replay``.

mAP50 comes from the per-cell CSVs written by ``scripts/eval.sh``
(``evaluation_results/model_<k>_eval_<tag>.csv``), using the class-aware
protocol. Per-stage training hours come from each task's
``train/results.csv``: the cumulative ``time`` of the epoch whose
``metrics/mAP50(B)`` is highest, i.e. the epoch that produced ``best.pt``
(``Metric.fitness`` in this working tree weights mAP50).
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "runs/417-ir"

NAIVE_LABEL = "微调"
INCREMENTAL_LABEL = "增量"

CIL_METHODS = ("naive", "pseudo_label+espreg+replay")
DIL_METHODS = ("naive", "espreg+replay")


@dataclass(frozen=True)
class CILSetting:
    """One CIL experiment directory and the labels of its two training stages."""

    name: str
    stage_1: str
    stage_2: str
    queue: str


# The 0-3 queue owns the hf-family experiments and the 4-7 queue the
# joint-family ones, so the two schedulers never train on the same second-stage
# dataset at the same time. Stage 1 (bk+hm) is shared by design.
CIL_SETTINGS = (
    CILSetting("bk+hm->hf", "驱逐舰、航母", "两栖", "cil-0-3"),
    CILSetting("bk+hm->hf-400", "驱逐舰、航母", "两栖（少样本）", "cil-0-3"),
    CILSetting("bk+hm->hf-pruned-400", "驱逐舰、航母", "两栖（最优样本）", "cil-0-3"),
    CILSetting("bk+hm-400->hf-400", "驱逐舰、航母（少样本）", "两栖（少样本）", "cil-0-3"),
    CILSetting("bk+hm->joint", "驱逐舰、航母", "驱逐舰、航母、两栖", "cil-4-7"),
    CILSetting("bk+hm->joint-hf-400", "驱逐舰、航母", "驱逐舰、航母、两栖（两栖为少样本）", "cil-4-7"),
    CILSetting("bk+hm->joint-hf-pruned-400", "驱逐舰、航母", "驱逐舰、航母、两栖（两栖为最优样本）", "cil-4-7"),
)

DIL_SETTING = "simulation->sea->land->low-contrast->heavy-cover"
DIL_STAGES = ("simulation", "sea", "land", "low-contrast", "heavy-cover")
DIL_LABELS = {
    "simulation": "纯仿真",
    "sea": "海态",
    "land": "陆态",
    "low-contrast": "低对比",
    "heavy-cover": "高遮挡",
}

QUEUES = ("cil-0-3", "cil-4-7", "dil-0-7")


# --------------------------------------------------------------------------- #
# Launch plan
# --------------------------------------------------------------------------- #
def queue_entries(queue: str, root: str) -> list[tuple[str, str, str, int]]:
    """Return (run_dir, method, scripts_dir, stage_count) of one queue, in order."""
    entries = []
    if queue in ("cil-0-3", "cil-4-7"):
        for setting in CIL_SETTINGS:
            if setting.queue != queue:
                continue
            for method in CIL_METHODS:
                entries.append(
                    (
                        str(Path(root) / "CIL" / setting.name / method),
                        method,
                        f"scripts/417-ir/CIL/{setting.name}",
                        2,
                    )
                )
    elif queue == "dil-0-7":
        for method in DIL_METHODS:
            entries.append(
                (
                    str(Path(root) / "DIL" / DIL_SETTING / method),
                    method,
                    f"scripts/417-ir/DIL/{DIL_SETTING}",
                    len(DIL_STAGES),
                )
            )
    else:
        raise ValueError(f"Unknown queue: {queue}")
    return entries


def command_plan(args: argparse.Namespace) -> int:
    """Print the queue as ``run_dir<TAB>method<TAB>scripts_dir<TAB>stage_count`` lines."""
    for run_dir, method, scripts_dir, stage_count in queue_entries(args.queue, args.root):
        print(f"{run_dir}\t{method}\t{scripts_dir}\t{stage_count}")
    return 0


# --------------------------------------------------------------------------- #
# Result reading
# --------------------------------------------------------------------------- #
def _to_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_map50(csv_path: Path) -> float | None:
    """Read mAP50 of one eval cell, averaging per-class rows when no all row exists."""
    if not csv_path.is_file():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "mAP50" not in (reader.fieldnames or []):
            return None
        rows = list(reader)
    if not rows:
        return None
    for row in rows:
        if (row.get("Class") or "").strip().casefold() == "all":
            return _to_float(row.get("mAP50"))
    values = [value for value in (_to_float(row.get("mAP50")) for row in rows) if value is not None]
    return sum(values) / len(values) if values else None


def stage_hours(task_dir: Path) -> float | None:
    """Hours from the first epoch to the epoch that produced best.pt."""
    csv_path = task_dir / "train" / "results.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        # best.pt is tracked by Metric.fitness, which this working tree weights as
        # mAP50 (w = [0, 0, 1, 0] in ultralytics/utils/metrics.py); fall back to
        # mAP50-95 when the CSV has no mAP50 column.
        metric = next((name for name in fieldnames if name.startswith("metrics/mAP50(")), None)
        if metric is None:
            metric = next((name for name in fieldnames if name.startswith("metrics/mAP50-95")), None)
        if metric is None or "time" not in fieldnames:
            return None
        best_score = None
        best_time = None
        for row in reader:
            score = _to_float(row.get(metric))
            elapsed = _to_float(row.get("time"))
            if score is None or elapsed is None:
                continue
            # Strict improvement keeps the first epoch on ties; the final
            # post-training validation row repeats the best metrics and must
            # never win on its later timestamp.
            if best_score is None or score > best_score:
                best_score, best_time = score, elapsed
    return None if best_time is None else best_time / 3600.0


def eval_cell(run_dir: Path, model_task: int, kind: str, dataset_task: int, suffix: str = "") -> float | None:
    """mAP50 of ``model-<model_task>`` on one eval cell of the eval sequence."""
    return read_map50(
        run_dir / "evaluation_results" / f"model_{model_task}_eval_{kind}_{dataset_task}{suffix}.csv"
    )


def stage_times(run_dir: Path, count: int) -> list[float | None]:
    """Per-stage training hours of the first ``count`` tasks of a run."""
    return [stage_hours(run_dir / f"task-{index}") for index in range(1, count + 1)]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _display_width(text))


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a left-aligned ASCII table that stays aligned with CJK text."""
    widths = [max(_display_width(str(row[column])) for row in (headers, *rows)) for column in range(len(headers))]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [separator, "| " + " | ".join(_pad(head, width) for head, width in zip(headers, widths)) + " |", separator]
    lines += ["| " + " | ".join(_pad(cell, width) for cell, width in zip(row, widths)) + " |" for row in rows]
    lines.append(separator)
    return "\n".join(lines)


def pct(value: float | None) -> str:
    return "缺失" if value is None else f"{value * 100:.2f}"


def durations(values: list[float | None]) -> str:
    return "+".join("缺失" if value is None else f"{value:.2f}h" for value in values)


# --------------------------------------------------------------------------- #
# CIL report
# --------------------------------------------------------------------------- #
def cil_label(setting: CILSetting, method: str) -> str:
    suffix = NAIVE_LABEL if method == "naive" else INCREMENTAL_LABEL
    return f"{setting.stage_1}训练+{setting.stage_2}{suffix}"


def cil_rows(root: Path) -> list[list[str]]:
    rows = []
    for setting in CIL_SETTINGS:
        for method in CIL_METHODS:
            run_dir = root / "CIL" / setting.name / method
            stage_1 = f"旧 {pct(eval_cell(run_dir, 1, 'task', 1))}"
            stage_2 = (
                f"旧 {pct(eval_cell(run_dir, 2, 'task', 1))} / "
                f"新 {pct(eval_cell(run_dir, 2, 'task', 2))} / "
                f"全部 {pct(eval_cell(run_dir, 2, 'cumulative', 2))}"
            )
            rows.append([cil_label(setting, method), stage_1, stage_2, durations(stage_times(run_dir, 2))])
    return rows


def objectness_row(root: Path) -> list[str]:
    """Stage-1 (bk+hm) model scored on hf under the class-agnostic protocol."""
    candidates = [root / "CIL" / "bk+hm->hf" / method for method in CIL_METHODS]
    stage_1_run = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "evaluation_results" / "model_1_eval_task_2_object.csv").is_file()
        ),
        candidates[0],
    )
    value = eval_cell(stage_1_run, 1, "task", 2, suffix="_object")
    return [
        "驱逐舰、航母训练（在两栖数据集上按照objectness评估）",
        f"objectness {pct(value)}",
        "—",
        durations(stage_times(stage_1_run, 1)),
    ]


# --------------------------------------------------------------------------- #
# DIL report
# --------------------------------------------------------------------------- #
def dil_chain_label(stage_count: int, kind: str) -> str:
    """Chain label from simulation through the first ``stage_count`` stages."""
    label = f"{DIL_LABELS[DIL_STAGES[0]]}训练"
    return label + "".join(f" + {DIL_LABELS[name]}{kind}" for name in DIL_STAGES[1:stage_count])


def dil_stage_rows(root: Path, stage: int) -> list[list[str]]:
    """The three statistics of stage ``stage`` (1-based, >= 2), each over tasks 1..stage.

    1. fine-tune the first stage-1 stages in order (naive)
    2. fine-tune the first stage stages in order (naive)
    3. train the first stage stages in order under the incremental module
    """
    naive_run = root / "DIL" / DIL_SETTING / "naive"
    inc_run = root / "DIL" / DIL_SETTING / DIL_METHODS[1]
    records = (
        (dil_chain_label(stage - 1, NAIVE_LABEL), naive_run, stage - 1),
        (dil_chain_label(stage, NAIVE_LABEL), naive_run, stage),
        (dil_chain_label(stage, INCREMENTAL_LABEL), inc_run, stage),
    )
    rows = []
    for label, run_dir, model_task in records:
        cells = [pct(eval_cell(run_dir, model_task, "task", task)) for task in range(1, stage + 1)]
        rows.append([label, *cells, durations(stage_times(run_dir, model_task))])
    return rows


# --------------------------------------------------------------------------- #
# Missing result summary
# --------------------------------------------------------------------------- #
def run_issues(run_dir: Path, stage_count: int) -> list[str]:
    issues = [
        f"task-{index} 训练时长" for index in range(1, stage_count + 1) if stage_hours(run_dir / f"task-{index}") is None
    ]
    if not (run_dir / "evaluation_results").is_dir():
        issues.append("评估结果")
    return issues


def missing_report(root: Path) -> list[str]:
    entries: list[tuple[str, Path, int]] = []
    for setting in CIL_SETTINGS:
        for method in CIL_METHODS:
            entries.append((f"CIL/{setting.name}/{method}", root / "CIL" / setting.name / method, 2))
    for method in DIL_METHODS:
        entries.append((f"DIL/{DIL_SETTING}/{method}", root / "DIL" / DIL_SETTING / method, len(DIL_STAGES)))
    lines = []
    for name, run_dir, stage_count in entries:
        if not run_dir.is_dir():
            lines.append(f"- {name}：未开始（无运行目录）")
            continue
        issues = run_issues(run_dir, stage_count)
        if issues:
            lines.append(f"- {name}：缺少 {'、'.join(issues)}")
    return lines


# --------------------------------------------------------------------------- #
# Report command
# --------------------------------------------------------------------------- #
def command_report(args: argparse.Namespace) -> int:
    root = Path(args.root)
    print("=" * 100)
    print("417-ir 增量实验结果汇总")
    print(f"  运行目录 : {root}")
    print("  mAP50(%) : 旧 = 驱逐舰+航母（bk+hm）| 新 = 两栖（hf）| 全部 = 联合（joint）")
    print("  训练时长 : 从最初 epoch 到 best.pt 产生（小时）")
    print("=" * 100)

    print("\n## 类增量（CIL）")
    print(f"说明：微调 = naive，增量 = {CIL_METHODS[1]}；一阶段给出旧类，二阶段给出旧/新/全部。\n")
    rows = cil_rows(root)
    rows.append(objectness_row(root))
    print(render_table(["结果记录名称", "一阶段评估结果", "二阶段评估结果", "各阶段训练时长"], rows))

    print("\n## 域增量（DIL）")
    print(
        "说明：第 k 个阶段统计（1）前 k-1 个阶段顺序微调、（2）前 k 个阶段顺序微调、"
        f"（3）前 k 个阶段顺序增量训练；每个统计项展示前 k 个任务数据集的 mAP50。增量 = {DIL_METHODS[1]}。"
    )
    for stage in range(2, len(DIL_STAGES) + 1):
        headers = ["记录项", *[DIL_LABELS[name] for name in DIL_STAGES[:stage]], "各阶段训练时长"]
        print(f"\n### 第 {stage} 阶段（前 {stage} 个任务）")
        print(render_table(headers, dil_stage_rows(root, stage)))

    issues = missing_report(root)
    print("\n## 缺失结果")
    if not issues:
        print("全部结果齐全。")
    else:
        print("以下结果缺失（表中已标记“缺失”并跳过计算）：")
        print("\n".join(issues))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="print one scheduler queue")
    plan.add_argument("--queue", required=True, choices=QUEUES)
    plan.add_argument("--root", default=DEFAULT_ROOT, help=f"run root (default: {DEFAULT_ROOT})")
    plan.set_defaults(func=command_plan)

    report = subparsers.add_parser("report", help="print the CIL and DIL result tables")
    report.add_argument("--root", default=DEFAULT_ROOT, help=f"run root (default: {DEFAULT_ROOT})")
    report.set_defaults(func=command_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
