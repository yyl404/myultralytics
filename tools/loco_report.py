"""Summarize a VOC LoCo motivation replication and compare it with the recorded protocol.

The LoCo (Low Co-occurrence) benchmark assigns every image to exactly one
incremental stage. This tool reads the artifacts produced by the LoCo pipeline
(``stage_stats``, ``teacher_errors``, ``history_errors``, ``stage_mAP_sequence``)
and writes one Markdown report with:

1. the LoCo class partition and the per-stage image assignment statistics;
2. stage class counts and stage sample overlap (motivation experiment 1);
3. main old/new/all mAP tables (task-unknown protocol);
4. teacher high-confidence errors and their severity on the trained models
   (motivation experiment 2);
5. incremental-model errors on the history stage (motivation experiment 3);
6. a headline comparison against the values recorded in ``experiments.md``,
   whose tables are parsed and reprinted verbatim in the appendix.

Usage:
    $ python tools/loco_report.py \
        --two_root data/VOC_LoCo_10+10 \
        --four_root data/VOC_LoCo_5+5+5+5 \
        --run_two_naive runs/<naive-10+10> --run_two_lwf runs/<lwf-10+10> \
        --run_four_naive runs/<naive-5+5+5+5> --run_four_lwf runs/<lwf-5+5+5+5> \
        --joint_eval_two runs/<joint-10+10-eval> --joint_eval_four runs/<joint-4-stage-eval> \
        --reference experiments.md \
        --output runs/<summary-dir>/summary.md

The report is read-only; every metric comes from an existing CSV or JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

METRICS = ("mAP50", "mAP50-95")
PROTOCOLS = (("10+10", 2), ("5+5+5+5", 4))
TEACHER_CONFS = (0.05, 0.1, 0.25, 0.5, 0.7, 0.9)
REPORT_CONF = 0.25
TEST_SETS = ("stage-1", "stage-2", "cum-2")
METHOD_ORDER = ("base", "naive", "lwf", "joint")
METHOD_LABELS = {
    "base": "基础模型（stage-1）",
    "naive": "naive",
    "lwf": "YOLO-LwF",
    "joint": "Joint",
}
TEST_SET_LABELS = {
    "stage-1": "阶段一测试集",
    "stage-2": "阶段二测试集",
    "cum-2": "累积测试集（阶段 1–2）",
}


@dataclass(frozen=True)
class ReferenceTable:
    """One Markdown table extracted from a reference report."""

    section: str
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def _split_markdown_row(line: str) -> list[str]:
    """Split one Markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_markdown_tables(text: str) -> list[ReferenceTable]:
    """Parse every Markdown table, recording its section and preceding caption."""
    tables = []
    section = ""
    caption = ""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("#"):
            section = line
            caption = line
            index += 1
            continue
        if not line.startswith("|"):
            if line:
                caption = line
            index += 1
            continue
        block = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index].strip())
            index += 1
        if len(block) < 2:
            continue
        headers = tuple(_split_markdown_row(block[0]))
        rows = tuple(tuple(_split_markdown_row(row)) for row in block[2:])
        if not headers:
            continue
        tables.append(ReferenceTable(section=section, caption=caption, headers=headers, rows=rows))
    return tables


def _table_has(table: ReferenceTable, header: str) -> bool:
    return header in table.headers


def _find_tables(
    tables: Sequence[ReferenceTable],
    required: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    header0: str | None = None,
) -> list[ReferenceTable]:
    """Return tables containing every required header and no forbidden one."""
    matches = []
    for table in tables:
        if header0 is not None and (not table.headers or table.headers[0] != header0):
            continue
        if any(header not in table.headers for header in required):
            continue
        if any(header in table.headers for header in forbidden):
            continue
        matches.append(table)
    return matches


def _parse_percent(value: str) -> float | None:
    """Parse '18.4%' or '18.4' into a percentage; '—' becomes None."""
    cleaned = value.strip()
    if cleaned in {"", "—", "-", "N/A"}:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", cleaned)
    if match:
        return float(match.group(1))
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else None


def _parse_count_percent(value: str) -> tuple[int | None, float | None]:
    """Parse a count/percentage cell in either order.

    The reference report uses both ``852 (18.4%)`` (count first) and
    ``14.2% (652)`` (percentage first); plain ``837`` and ``80.4%`` cells carry
    only one of the two values.
    """
    cleaned = value.strip()
    parenthesized = re.search(r"\((.*?)\)", cleaned)
    if parenthesized is not None:
        inner = parenthesized.group(1)
        inner_number = re.search(r"\d+(?:\.\d+)?", inner)
        if inner_number is None:
            return None, _parse_percent(cleaned)
        if "%" in inner:
            # "852 (18.4%)": the leading number is a count.
            leading = re.match(r"(\d+)", cleaned)
            return (
                int(leading.group(1)) if leading else None,
                float(inner_number.group(0)),
            )
        # "14.2% (652)": the parenthesized number is a count.
        leading = re.match(r"(\d+(?:\.\d+)?)\s*%", cleaned)
        return int(inner_number.group(0)), float(leading.group(1)) if leading else None
    if "%" in cleaned:
        return None, _parse_percent(cleaned)
    count_match = re.match(r"(\d+)", cleaned)
    return (int(count_match.group(1)) if count_match else None), None


def _parse_count(value: str) -> int | None:
    """Parse a plain integer cell."""
    match = re.search(r"\d+", value.strip())
    return int(match.group(0)) if match else None


def _format_percent(value: float | None, digits: int = 1) -> str:
    """Format a percentage or an empty cell."""
    return "—" if value is None else f"{value:.{digits}f}%"


def _format_metric(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 metric as a percentage or an empty cell."""
    return "—" if value is None else f"{value * 100.0:.{digits}f}"


def _format_count_percent(count: int | None, rate: float | None) -> str:
    """Format 'count (rate%)' for an error cell, keeping whichever part exists."""
    if count is None and rate is None:
        return "—"
    if count is None:
        return f"{rate:.1f}%"
    if rate is None:
        return str(count)
    return f"{count} ({rate:.1f}%)"


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table with numeric alignment on every non-first column."""
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join("—" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


class StageSequence:
    """Per-class and per-stage mAP rows of one run's evaluation matrix."""

    def __init__(self, csv_path: Path) -> None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Stage mAP sequence not found: {csv_path}")
        self.rows: dict[tuple[int, str, str], dict[str, float]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                model_task = int(row["Model_Task"].removeprefix("Model_"))
                key = (model_task, row["Eval"], row["Task"])
                self.rows[key] = {metric: float(row[metric]) for metric in METRICS if metric in row}

    def metric(self, model_task: int, eval_name: str, task: str, metric: str) -> float | None:
        row = self.rows.get((model_task, eval_name, task))
        return None if row is None else row.get(metric)

    def final_summary(self, model_task: int, stages: int, metric: str) -> tuple[float | None, float | None, float | None]:
        """Return (old, new, all) for the final cumulative model."""
        eval_name = f"cumulative_{stages}"
        stage_values = [
            self.metric(model_task, eval_name, f"Task_{stage}", metric) for stage in range(1, stages + 1)
        ]
        old = _mean(stage_values[:-1])
        new = stage_values[-1]
        return old, new, self.metric(model_task, eval_name, "All", metric)


def read_error_rows(csv_path: Path) -> dict[float, dict[str, float]]:
    """Read an error_rates.csv as {confidence_threshold: row}."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Error-rate CSV not found: {csv_path}")
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            threshold = round(float(row["confidence_threshold"]), 6)
            rows[threshold] = {key: float(value) for key, value in row.items() if value not in {"", None}}
    return rows


def row_at(rows: Mapping[float, Mapping[str, float]], threshold: float) -> Mapping[str, float]:
    """Return the row at a confidence threshold with a clear error when it is absent."""
    key = round(threshold, 6)
    if key not in rows:
        raise KeyError(f"No error row at confidence {threshold}; available: {sorted(rows)}")
    return rows[key]


def read_stage_stats(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """Read one stats CSV into a header and row list."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Stage statistics CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty stage statistics CSV: {csv_path}")
    return rows[0], rows[1:]


def read_loco_partition(data_root: Path) -> dict:
    """Read loco_partition.json from a created LoCo dataset."""
    path = data_root / "loco_partition.json"
    if not path.is_file():
        raise FileNotFoundError(f"LoCo partition metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_reference_main(
    tables: Sequence[ReferenceTable],
    stages: int,
) -> dict[str, dict[tuple[str, int], list[float | None]]]:
    """Parse the recorded main-result tables into {metric: {(method, stage): values}}.

    Stage 0 denotes the single-stage Joint row; values are
    ``test-1..test-n, seen-n-task-1..seen-n-task-n``. The first matching table
    holds mAP50 and the second holds mAP50-95.
    """
    if stages == 4:
        matches = _find_tables(tables, required=("test-4", "seen-4-task-4"))
    else:
        matches = _find_tables(tables, required=("method&stage", "seen-2-task-2"))
    if len(matches) < 2:
        raise ValueError(f"Expected two main-result tables for {stages} stages, found {len(matches)}")

    parsed: dict[str, dict[tuple[str, int], list[float | None]]] = {metric: {} for metric in METRICS}
    for metric, table in zip(METRICS, matches):
        for row in table.rows:
            if stages == 4:
                method_name, stage_label = row[0], row[1]
                values = [_parse_percent(cell) for cell in row[2 : 2 + 2 * stages]]
                if method_name == "Joint":
                    method, stage = "joint", 0
                elif method_name == "naive":
                    method = "naive"
                    stage = int(stage_label.removeprefix("s"))
                elif method_name == "YOLO-LwF":
                    method = "lwf"
                    stage = int(stage_label.removeprefix("s"))
                else:
                    raise ValueError(f"Unknown reference method row '{method_name}'")
            else:
                label = row[0]
                values = [_parse_percent(cell) for cell in row[1 : 1 + 2 * stages]]
                if label.startswith("Joint"):
                    method, stage = "joint", 0
                else:
                    match = re.fullmatch(r"(naive|YOLO-LwF)-stage-(\d)", label)
                    if match is None:
                        raise ValueError(f"Unknown reference method row '{label}'")
                    method = {"naive": "naive", "YOLO-LwF": "lwf"}[match.group(1)]
                    stage = int(match.group(2))
            if len(values) != 2 * stages:
                raise ValueError(f"Reference row '{row[0]}' has {len(values)} metric cells")
            parsed[metric][(method, stage)] = values
    return parsed


def _parse_reference_teacher(
    tables: Sequence[ReferenceTable],
    stages: int,
) -> dict[float, dict[str, float | None]]:
    """Parse the recorded teacher error table keyed by confidence threshold."""
    required = ("Conf", "上报框", "总误报率", "第一类误报率", "第二类误报率", "历史类漏检率")
    forbidden = () if stages == 4 else ("第三类误报率",)
    if stages == 4:
        required = (*required, "第三类误报率")
    matches = _find_tables(tables, required=required, forbidden=forbidden, header0="Conf")
    if len(matches) != 1:
        raise ValueError(f"Expected one teacher table for {stages} stages, found {len(matches)}")
    table = matches[0]
    index = {header: position for position, header in enumerate(table.headers)}
    parsed = {}
    for row in table.rows:
        threshold = float(row[index["Conf"]])
        parsed[round(threshold, 6)] = {
            "pred": _parse_count(row[index["上报框"]]),
            "total": _parse_percent(row[index["总误报率"]]),
            "type1": _parse_percent(row[index["第一类误报率"]]),
            "type2": _parse_percent(row[index["第二类误报率"]]),
            "type3": _parse_percent(row[index["第三类误报率"]]) if stages == 4 else None,
            "miss": _parse_percent(row[index["历史类漏检率"]]),
        }
    return parsed


def _severity_test_key(caption: str) -> str:
    """Map a reference severity caption to a test-set key."""
    if "阶段一" in caption:
        return "stage-1"
    if "阶段二" in caption:
        return "stage-2"
    if "累积" in caption:
        return "cum-2"
    raise ValueError(f"Cannot map severity table caption '{caption}' to a test set")


def _reference_method(method_name: str) -> str:
    """Map a reference severity-row label to an internal method key."""
    if method_name.startswith("基础模型"):
        return "base"
    if method_name == "naive":
        return "naive"
    if method_name == "YOLO-LwF":
        return "lwf"
    if method_name == "Joint":
        return "joint"
    raise ValueError(f"Unknown reference severity method '{method_name}'")


def _parse_reference_severity(
    tables: Sequence[ReferenceTable],
    stages: int,
) -> dict[str, dict[str, dict[str, object]]]:
    """Parse the recorded severity tables into {test_set: {method: metrics}}."""
    forbidden = () if stages == 4 else ("未来阶段→历史类",)
    matches = _find_tables(
        tables,
        required=("方法", "上报框", "总误报率", "背景→增量类", "背景→历史类", "历史类→增量类", "增量类→历史类"),
        forbidden=forbidden,
        header0="方法",
    )
    if stages == 4:
        matches = [table for table in matches if _table_has(table, "未来阶段→历史类")]
    else:
        matches = [table for table in matches if not _table_has(table, "未来阶段→历史类")]
    if len(matches) != 3:
        raise ValueError(f"Expected three severity tables for {stages} stages, found {len(matches)}")

    parsed: dict[str, dict[str, dict[str, object]]] = {}
    for table in matches:
        index = {header: position for position, header in enumerate(table.headers)}
        test_key = _severity_test_key(table.caption)
        parsed[test_key] = {}
        for row in table.rows:
            method = _reference_method(row[index["方法"]])
            parsed[test_key][method] = {
                "pred": _parse_count(row[index["上报框"]]),
                "total": _parse_percent(row[index["总误报率"]]),
                "type1": _parse_count_percent(row[index["背景→增量类"]]),
                "type2": _parse_count_percent(row[index["背景→历史类"]]),
                "type3": _parse_count_percent(row[index["历史类→增量类"]]),
                "type4": _parse_count_percent(row[index["增量类→历史类"]]),
                "type5": _parse_count_percent(row[index["未来阶段→历史类"]]) if stages == 4 else (None, None),
                "miss": _parse_percent(row[index["历史类漏检率"]]),
            }
    return parsed


def _parse_reference_incremental(
    tables: Sequence[ReferenceTable],
) -> dict[str, dict[str, tuple[int | None, float | None]]]:
    """Parse the recorded two-stage object-level incremental-error table."""
    matches = _find_tables(tables, required=("naive", "YOLO-LwF"), header0="对象级统计（conf 0.25）")
    if len(matches) != 1:
        raise ValueError(f"Expected one incremental object table, found {len(matches)}")
    table = matches[0]
    parsed: dict[str, dict[str, tuple[int | None, float | None]]] = {"naive": {}, "lwf": {}}
    for row in table.rows:
        label = row[0]
        if "正确历史类" in label:
            key = "correct_history"
        elif "增量类（第三类误报）" in label:
            key = "hist_to_inc"
        elif "增量类目标漏检" in label:
            key = "miss_incremental"
        elif "历史类（第四类误报）" in label:
            key = "inc_to_hist"
        elif "正确增量类" in label:
            key = "correct_incremental"
        elif "旧类目标漏检" in label:
            key = "miss_history"
        elif "背景 → 增量类" in label:
            key = "bg_to_inc"
        elif "背景 → 历史类" in label:
            key = "bg_to_hist"
        else:
            continue
        parsed["naive"][key] = _parse_count_percent(row[1])
        parsed["lwf"][key] = _parse_count_percent(row[2])
    return parsed


def _seq_metric(
    sequence: StageSequence,
    model_task: int,
    eval_name: str,
    task: int,
    metric: str,
) -> float | None:
    return sequence.metric(model_task, eval_name, f"Task_{task}", metric)


def _main_table(
    sequences: Mapping[str, StageSequence],
    stages: int,
    metric: str,
) -> str:
    """Render the LoCo main-result table in the layout of the reference report."""
    headers = [
        *(("方法", "阶段") if stages == 4 else ("method&stage",)),
        *(f"test-{stage}" for stage in range(1, stages + 1)),
        *(f"seen-{stages}-task-{stage}" for stage in range(1, stages + 1)),
    ]
    rows: list[list[str]] = []

    joint = sequences["joint"]
    joint_cells = [
        _format_metric(_seq_metric(joint, 1, f"task_{stage}", stage, metric)) for stage in range(1, stages + 1)
    ]
    joint_cells += [
        _format_metric(_seq_metric(joint, 1, f"cumulative_{stages}", stage, metric))
        for stage in range(1, stages + 1)
    ]
    if stages == 4:
        rows.append(["Joint", "单阶段", *joint_cells])
    else:
        rows.append(["Joint（单阶段）", *joint_cells])

    for method, label in (("naive", "naive"), ("lwf", "YOLO-LwF")):
        sequence = sequences[method]
        for model_task in range(1, stages + 1):
            cells = []
            for test_stage in range(1, stages + 1):
                value = (
                    _seq_metric(sequence, model_task, f"task_{test_stage}", test_stage, metric)
                    if test_stage <= model_task
                    else None
                )
                cells.append(_format_metric(value))
            for seen_stage in range(1, stages + 1):
                value = (
                    _seq_metric(sequence, model_task, f"cumulative_{stages}", seen_stage, metric)
                    if seen_stage <= model_task
                    else None
                )
                cells.append(_format_metric(value))
            if stages == 4:
                rows.append([label, f"s{model_task}", *cells])
            else:
                rows.append([f"{label}-stage-{model_task}", *cells])
    return _markdown_table(headers, rows)


def _reference_old_new_all(
    reference: Mapping[tuple[str, int], list[float | None]],
    method: str,
    stages: int,
) -> dict[str, float | None]:
    """Return reference old/new/all percentages from the parsed main tables."""
    stage_key = 0 if method == "joint" else stages
    if (method, stage_key) not in reference:
        raise KeyError(f"Reference main table has no row for {(method, stage_key)}")
    values = reference[(method, stage_key)]
    seen = values[stages : 2 * stages]
    if len(seen) != stages:
        raise ValueError(f"Reference row for {(method, stage_key)} has too few seen-N columns")
    old = _mean(seen[:-1]) if stages > 1 else None
    new = seen[-1]
    all_value = _mean(seen)
    return {"old": old, "new": new, "all": all_value}


def _severity_loco_row(
    rows: Mapping[float, Mapping[str, float]],
    threshold: float,
    stages: int,
) -> dict[str, object]:
    """Build one LoCo severity row at a confidence threshold."""
    row = row_at(rows, threshold)
    pred = row.get("n_pred", 0.0)
    matched = row.get("n_matched", 0.0)
    return {
        "pred": int(pred),
        "total": (pred - matched) / pred * 100.0 if pred else None,
        "type1": (int(row.get("n_type1_background_to_incremental", 0.0)), row.get("type1_rate", 0.0) * 100.0),
        "type2": (int(row.get("n_type2_background_to_history", 0.0)), row.get("type2_rate", 0.0) * 100.0),
        "type3": (int(row.get("n_type3_history_to_incremental", 0.0)), row.get("type3_rate", 0.0) * 100.0),
        "type4": (int(row.get("n_type4_incremental_to_history", 0.0)), row.get("type4_rate", 0.0) * 100.0),
        "type5": (
            int(row.get("n_type5_future_to_history", 0.0)),
            row.get("type5_rate", 0.0) * 100.0,
        )
        if stages == 4
        else (None, None),
        "miss": row.get("miss_history_rate", 0.0) * 100.0,
    }


def _incremental_loco_metrics(
    rows: Mapping[float, Mapping[str, float]],
    threshold: float,
) -> dict[str, tuple[int | None, float | None]]:
    """Build the object-level incremental-error metrics of one model.

    Every GT object is classified by its highest-IoU reported box, so the
    per-group outcomes sum to that group's GT count. Background false positives
    stay box-level because they have no GT object.
    """
    row = row_at(rows, threshold)
    gt_history = row.get("n_gt_history", 0.0)
    gt_incremental = row.get("n_gt_incremental", 0.0)
    gt_future = row.get("n_gt_future", 0.0)

    def rate(numerator: float, denominator: float) -> float | None:
        return numerator / denominator * 100.0 if denominator else None

    def outcome(key: str, denominator: float) -> tuple[int | None, float | None]:
        return int(row.get(key, 0.0)), rate(row.get(key, 0.0), denominator)

    return {
        "correct_history": outcome("n_obj_correct_history", gt_history),
        "hist_to_inc": outcome("n_obj_history_to_incremental", gt_history),
        "hist_to_future": outcome("n_obj_history_to_future", gt_history),
        "miss_history": outcome("n_obj_miss_history", gt_history),
        "correct_incremental": outcome("n_obj_correct_incremental", gt_incremental),
        "inc_to_hist": outcome("n_obj_incremental_to_history", gt_incremental),
        "inc_to_future": outcome("n_obj_incremental_to_future", gt_incremental),
        "miss_incremental": outcome("n_obj_miss_incremental", gt_incremental),
        "future_to_hist": outcome("n_obj_future_to_history", gt_future),
        "bg_to_inc": (int(row.get("n_type1_background_to_incremental", 0.0)), None),
        "bg_to_hist": (int(row.get("n_type2_background_to_history", 0.0)), None),
    }


def _load_reference_tables(path: Path) -> list[ReferenceTable]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference report not found: {path}")
    return parse_markdown_tables(path.read_text(encoding="utf-8"))


def _protocol_title(stages: int) -> str:
    return "5+5+5+5" if stages == 4 else "10+10"


def _artifact_lines(args: argparse.Namespace) -> list[str]:
    """Collect the artifact paths referenced by the report."""
    return [
        f"- dataset: `{args.two_root}` / `{args.four_root}`",
        f"- naive runs: `{args.run_two_naive}` / `{args.run_four_naive}`",
        f"- YOLO-LwF runs: `{args.run_two_lwf}` / `{args.run_four_lwf}`",
        f"- Joint eval runs: `{args.joint_eval_two}` / `{args.joint_eval_four}`",
        f"- reference: `{args.reference}`",
    ]


def build_report(args: argparse.Namespace) -> str:
    """Build the whole Markdown report."""
    reference_tables = _load_reference_tables(args.reference)
    reference_main = {stages: _parse_reference_main(reference_tables, stages) for _, stages in PROTOCOLS}
    reference_teacher = {
        stages: _parse_reference_teacher(reference_tables, stages) for _, stages in PROTOCOLS
    }
    reference_severity = {
        stages: _parse_reference_severity(reference_tables, stages) for _, stages in PROTOCOLS
    }
    reference_incremental = _parse_reference_incremental(reference_tables)

    sequences = {
        "two": {
            "naive": StageSequence(args.run_two_naive / "evaluation_results" / "stage_mAP_sequence.csv"),
            "lwf": StageSequence(args.run_two_lwf / "evaluation_results" / "stage_mAP_sequence.csv"),
            "joint": StageSequence(args.joint_eval_two / "evaluation_results" / "stage_mAP_sequence.csv"),
        },
        "four": {
            "naive": StageSequence(args.run_four_naive / "evaluation_results" / "stage_mAP_sequence.csv"),
            "lwf": StageSequence(args.run_four_lwf / "evaluation_results" / "stage_mAP_sequence.csv"),
            "joint": StageSequence(args.joint_eval_four / "evaluation_results" / "stage_mAP_sequence.csv"),
        },
    }

    lines: list[str] = []
    lines.append("# VOC LoCo (Low Co-occurrence) motivation report")
    lines.append("")
    lines.append(
        "LoCo clusters frequently co-occurring classes into the same incremental stage and assigns "
        "each image to exactly one stage, so no training image is shared between stages. This report "
        "replicates the motivation experiments on the two LoCo protocols and compares them with the "
        "values recorded in the reference report."
    )
    lines.append("")
    lines.append("- generated by: `python tools/loco_report.py`")
    lines.append(f"- assignment seed: `{args.seed}`")
    lines.extend(_artifact_lines(args))
    lines.append("")

    # ------------------------------------------------------------------ #
    # 1. Dataset partition
    # ------------------------------------------------------------------ #
    lines.append("## 1. LoCo class partition and image assignment")
    lines.append("")
    for stages, data_root in ((2, args.two_root), (4, args.four_root)):
        partition = read_loco_partition(data_root)
        lines.append(f"### {_protocol_title(stages)} (`{data_root}`)")
        lines.append("")
        lines.append(
            f"Cluster split `{partition['cluster_split']}`, seed `{partition['seed']}`, "
            f"stage sizes `{partition['classes_per_task']}`."
        )
        lines.append("")
        rows = []
        for stage in partition["stages"]:
            rows.append([stage["stage"], len(stage["classes"]), ", ".join(stage["classes"])])
        lines.append(_markdown_table(["Stage", "#Classes", "Classes"], rows))
        lines.append("")
        assignment = partition.get("image_assignment", {})
        assignment_rows = []
        for split, counts in assignment.items():
            total = sum(counts.values())
            multi = sum(value for candidate_count, value in counts.items() if int(candidate_count) > 1)
            assignment_rows.append(
                [split, total, multi, f"{multi / total * 100.0:.1f}%" if total else "—"]
            )
        lines.append(
            "Images spanning several candidate stages are assigned to one of them; each image is "
            "written to exactly one stage."
        )
        lines.append("")
        lines.append(
            _markdown_table(
                ["Split", "Images", "Multi-stage candidates", "Assigned from overlap"],
                assignment_rows,
            )
        )
        lines.append("")

    # ------------------------------------------------------------------ #
    # 2. Stage statistics
    # ------------------------------------------------------------------ #
    lines.append("## 2. Motivation experiment 1: stage class distribution and sample overlap")
    lines.append("")
    for stages, run_dir in ((2, args.run_two_naive), (4, args.run_four_naive)):
        stats_dir = run_dir / "stage_stats"
        lines.append(f"### {_protocol_title(stages)}")
        lines.append("")
        headers, rows = read_stage_stats(stats_dir / "stage_class_counts.csv")
        lines.append("**Per-stage class object counts**")
        lines.append("")
        lines.append(_markdown_table(headers, rows))
        lines.append("")
        headers, rows = read_stage_stats(stats_dir / "stage_sample_overlap.csv")
        lines.append("**Per-stage training sample overlap**")
        lines.append("")
        lines.append(_markdown_table(headers, rows))
        lines.append("")

    # ------------------------------------------------------------------ #
    # 3. Main results
    # ------------------------------------------------------------------ #
    lines.append("## 3. Main results (task-unknown mAP)")
    lines.append("")
    for stages, key in ((4, "four"), (2, "two")):
        lines.append(f"### {_protocol_title(stages)}")
        lines.append("")
        for metric, title in (("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(_main_table(sequences[key], stages, metric))
            lines.append("")

    lines.append("### Final-model old / new / all")
    lines.append("")
    lines.append(
        "`old` averages the history stages on the final cumulative test set, `new` is the last "
        "stage, and `all` is the cumulative-class average."
    )
    lines.append("")
    for metric in METRICS:
        lines.append(f"**{metric}**")
        lines.append("")
        comparison_rows = []
        for label, stages in PROTOCOLS:
            key = "four" if stages == 4 else "two"
            for method in ("naive", "lwf", "joint"):
                model_task = 1 if method == "joint" else stages
                loco = sequences[key][method].final_summary(model_task, stages, metric)
                ref = _reference_old_new_all(reference_main[stages][metric], method, stages)
                comparison_rows.append(
                    [
                        label,
                        METHOD_LABELS[method],
                        _format_percent(ref["old"]),
                        _format_percent(loco[0] * 100.0 if loco[0] is not None else None),
                        _format_percent(ref["new"]),
                        _format_percent(loco[1] * 100.0 if loco[1] is not None else None),
                        _format_percent(ref["all"]),
                        _format_percent(loco[2] * 100.0 if loco[2] is not None else None),
                    ]
                )
        lines.append(
            _markdown_table(
                [
                    "Protocol",
                    "Method",
                    "Old ref",
                    "Old LoCo",
                    "New ref",
                    "New LoCo",
                    "All ref",
                    "All LoCo",
                ],
                comparison_rows,
            )
        )
        lines.append("")

    # ------------------------------------------------------------------ #
    # 4. Teacher errors and severity
    # ------------------------------------------------------------------ #
    lines.append("## 4. Motivation experiment 2: teacher high-confidence errors")
    lines.append("")
    for stages, key, run_dir in (
        (2, "two", args.run_two_naive),
        (4, "four", args.run_four_naive),
    ):
        lines.append(f"### {_protocol_title(stages)}")
        lines.append("")
        teacher_rows = read_error_rows(run_dir / "teacher_errors" / "task_2" / "error_rates.csv")
        lines.append("**Teacher on the stage-2 training images**")
        lines.append("")
        table_rows = []
        for threshold in TEACHER_CONFS:
            row = row_at(teacher_rows, threshold)
            total = sum(
                row.get(name, 0.0)
                for name in (
                    "n_type1_background_to_history",
                    "n_type2_target_to_history",
                    "n_type3_future_to_history",
                )
            )
            table_rows.append(
                [
                    f"{threshold:.2f}",
                    int(row.get("n_pred", 0.0)),
                    _format_percent(total / row["n_pred"] * 100.0 if row.get("n_pred") else None),
                    _format_percent(row.get("type1_rate", 0.0) * 100.0),
                    _format_percent(row.get("type2_rate", 0.0) * 100.0),
                    _format_percent(row.get("type3_rate", 0.0) * 100.0) if stages == 4 else "—",
                    _format_percent(row.get("miss_rate", 0.0) * 100.0),
                ]
            )
        headers = ["Conf", "Reported", "Total FP", "Type 1 bg→hist", "Type 2 target→hist"]
        if stages == 4:
            headers.append("Type 3 future→hist")
        headers.append("Miss history")
        lines.append(_markdown_table(headers, table_rows))
        lines.append("")

        lines.append("**Severity: errors of the trained models on the test splits (conf = 0.25)**")
        lines.append("")
        for test_key in TEST_SETS:
            lines.append(f"*{TEST_SET_LABELS[test_key]}*")
            lines.append("")
            severity_rows = []
            for method in METHOD_ORDER:
                if method in {"base", "naive", "lwf"}:
                    rows_path = run_dir / (
                        f"history_errors-base/on-{test_key}/error_rates.csv"
                        if method == "base"
                        else f"history_errors/on-{test_key}/error_rates.csv"
                    )
                else:
                    rows_path = (
                        (args.joint_eval_two if stages == 2 else args.joint_eval_four)
                        / f"history_errors/on-{test_key}/error_rates.csv"
                    )
                severity = _severity_loco_row(read_error_rows(rows_path), REPORT_CONF, stages)
                cells = [
                    METHOD_LABELS[method],
                    severity["pred"],
                    _format_percent(severity["total"]),
                    _format_count_percent(*severity["type1"]),
                    _format_count_percent(*severity["type2"]),
                    _format_count_percent(*severity["type3"]),
                    _format_count_percent(*severity["type4"]),
                ]
                if stages == 4:
                    cells.append(_format_count_percent(*severity["type5"]))
                cells.append(_format_percent(severity["miss"]))
                severity_rows.append(cells)
            headers = [
                "Method",
                "Reported",
                "Total FP",
                "bg→inc",
                "bg→hist",
                "hist→inc",
                "inc→hist",
            ]
            if stages == 4:
                headers.append("future→hist")
            headers.append("Miss hist")
            lines.append(_markdown_table(headers, severity_rows))
            lines.append("")
        lines.append(
            "`Total FP` counts reported boxes that are not matched to a GT object of their own class "
            "group; it also includes future-stage objects reported as incremental classes, which has "
            "no dedicated column."
        )
        lines.append("")

    # ------------------------------------------------------------------ #
    # 5. Incremental errors (experiment 3)
    # ------------------------------------------------------------------ #
    lines.append("## 5. Motivation experiment 3: incremental-model errors on the history stage")
    lines.append("")
    for stages, key, run_dir in (
        (2, "two", args.run_two_naive),
        (4, "four", args.run_four_naive),
    ):
        lines.append(f"### {_protocol_title(stages)} (conf = 0.25, stage-1 test images)")
        lines.append("")
        incremental_rows = []
        for method, method_run in (
            ("naive", args.run_two_naive if stages == 2 else args.run_four_naive),
            ("lwf", args.run_two_lwf if stages == 2 else args.run_four_lwf),
        ):
            metrics = _incremental_loco_metrics(
                read_error_rows(method_run / "history_errors" / "on-stage-1" / "error_rates.csv"),
                REPORT_CONF,
            )
            cells = [
                METHOD_LABELS[method],
                _format_count_percent(*metrics["correct_history"]),
                _format_count_percent(*metrics["hist_to_inc"]),
                _format_count_percent(*metrics["miss_history"]),
                _format_count_percent(*metrics["correct_incremental"]),
                _format_count_percent(*metrics["inc_to_hist"]),
                _format_count_percent(*metrics["miss_incremental"]),
            ]
            if stages == 4:
                cells.append(_format_count_percent(*metrics["future_to_hist"]))
            cells.extend(
                [
                    _format_count_percent(*metrics["bg_to_inc"]),
                    _format_count_percent(*metrics["bg_to_hist"]),
                ]
            )
            incremental_rows.append(cells)
        headers = [
            "Method",
            "旧类→正确历史类",
            "旧类→增量类 (3rd)",
            "旧类漏检",
            "增量类→正确增量类",
            "增量类→历史类 (4th)",
            "增量类漏检",
            "背景→增量类 (boxes)",
            "背景→历史类 (boxes)",
        ]
        if stages == 4:
            headers.insert(7, "未来阶段→历史")
        lines.append(_markdown_table(headers, incremental_rows))
        lines.append("")
        lines.append(
            "Object-level rates use the corresponding GT object count as denominator; background "
            "counts are reported boxes. `(3rd)` and `(4th)` follow the reference report's numbering."
        )
        lines.append("")

    # ------------------------------------------------------------------ #
    # 6. Headline comparison
    # ------------------------------------------------------------------ #
    lines.append("## 6. Headline comparison against the recorded values")
    lines.append("")
    lines.append("**Teacher errors on the stage-2 training images**")
    lines.append("")
    teacher_comparison = []
    for label, stages in PROTOCOLS:
        key = "four" if stages == 4 else "two"
        loco_rows = read_error_rows(
            (args.run_four_naive if stages == 4 else args.run_two_naive)
            / "teacher_errors"
            / "task_2"
            / "error_rates.csv"
        )
        for threshold in (0.25, 0.9):
            ref = reference_teacher[stages][round(threshold, 6)]
            loco = row_at(loco_rows, threshold)
            for metric_key, metric_label in (("type1", "Type 1"), ("type2", "Type 2"), ("type3", "Type 3")):
                if stages == 2 and metric_key == "type3":
                    continue
                ref_value = ref.get(metric_key)
                loco_value = loco.get(
                    {
                        "type1": "type1_rate",
                        "type2": "type2_rate",
                        "type3": "type3_rate",
                    }[metric_key],
                    0.0,
                ) * 100.0
                teacher_comparison.append(
                    [label, f"{threshold:.2f}", metric_label, _format_percent(ref_value), _format_percent(loco_value)]
                )
    lines.append(
        _markdown_table(["Protocol", "Conf", "Metric", "Reference", "LoCo"], teacher_comparison)
    )
    lines.append("")

    lines.append(
        "**Teacher total false-positive rate: stage-1 test (same distribution) vs stage-2 "
        "training images (out of distribution), conf = 0.25**"
    )
    lines.append("")
    shift_rows = []
    for label, stages in PROTOCOLS:
        run_dir = args.run_four_naive if stages == 4 else args.run_two_naive
        teacher_rows = read_error_rows(run_dir / "teacher_errors" / "task_2" / "error_rates.csv")
        teacher = row_at(teacher_rows, REPORT_CONF)
        pred = teacher.get("n_pred", 0.0)
        loco_ood = (
            sum(
                teacher.get(name, 0.0)
                for name in (
                    "n_type1_background_to_history",
                    "n_type2_target_to_history",
                    "n_type3_future_to_history",
                )
            )
            / pred
            * 100.0
            if pred
            else None
        )
        loco_same = _severity_loco_row(
            read_error_rows(run_dir / "history_errors-base" / "on-stage-1" / "error_rates.csv"),
            REPORT_CONF,
            stages,
        )["total"]
        ref_same = reference_severity[stages]["stage-1"]["base"]["total"]
        ref_ood = reference_teacher[stages][round(REPORT_CONF, 6)]["total"]
        shift_rows.append(
            [
                label,
                _format_percent(ref_same),
                _format_percent(loco_same),
                _format_percent(ref_ood),
                _format_percent(loco_ood),
            ]
        )
    lines.append(
        _markdown_table(
            ["Protocol", "Stage-1 test ref", "Stage-1 test LoCo", "Stage-2 train ref", "Stage-2 train LoCo"],
            shift_rows,
        )
    )
    lines.append("")

    lines.append("**Trained-model total false-positive rate at conf = 0.25**")
    lines.append("")
    severity_comparison = []
    for label, stages in PROTOCOLS:
        run_dir = args.run_four_naive if stages == 4 else args.run_two_naive
        for test_key in TEST_SETS:
            for method in METHOD_ORDER:
                if method in {"base", "naive", "lwf"}:
                    rows_path = run_dir / (
                        f"history_errors-base/on-{test_key}/error_rates.csv"
                        if method == "base"
                        else f"history_errors/on-{test_key}/error_rates.csv"
                    )
                else:
                    rows_path = (
                        (args.joint_eval_two if stages == 2 else args.joint_eval_four)
                        / f"history_errors/on-{test_key}/error_rates.csv"
                    )
                loco = _severity_loco_row(read_error_rows(rows_path), REPORT_CONF, stages)
                ref = reference_severity[stages][test_key][method]
                severity_comparison.append(
                    [
                        label,
                        _short_test_label(test_key),
                        METHOD_LABELS[method],
                        _format_percent(ref["total"]),
                        _format_percent(loco["total"]),
                    ]
                )
    lines.append(
        _markdown_table(
            ["Protocol", "Test set", "Method", "Total FP ref", "Total FP LoCo"],
            severity_comparison,
        )
    )
    lines.append("")

    lines.append("**Two-stage incremental object-level errors at conf = 0.25**")
    lines.append("")
    incremental_comparison = []
    for metric_key, metric_label in (
        ("correct_history", "Old → correct history"),
        ("hist_to_inc", "Old → incremental (3rd)"),
        ("miss_history", "Old missed"),
        ("correct_incremental", "Incremental → correct"),
        ("inc_to_hist", "Incremental → history (4th)"),
        ("miss_incremental", "Incremental missed"),
        ("bg_to_inc", "Background → incremental (boxes)"),
        ("bg_to_hist", "Background → history (boxes)"),
    ):
        row = [metric_label]
        for method, run_dir in (("naive", args.run_two_naive), ("lwf", args.run_two_lwf)):
            ref_count, ref_rate = reference_incremental[method][metric_key]
            loco = _incremental_loco_metrics(
                read_error_rows(run_dir / "history_errors" / "on-stage-1" / "error_rates.csv"),
                REPORT_CONF,
            )[metric_key]
            row.append(_format_count_percent(ref_count, ref_rate))
            row.append(_format_count_percent(*loco))
        incremental_comparison.append(row)
    lines.append(
        _markdown_table(
            ["Metric", "naive ref", "naive LoCo", "YOLO-LwF ref", "YOLO-LwF LoCo"],
            incremental_comparison,
        )
    )
    lines.append("")

    # ------------------------------------------------------------------ #
    # 7. Reference appendix
    # ------------------------------------------------------------------ #
    lines.append("## 7. Reference values extracted from `experiments.md`")
    lines.append("")
    lines.append(
        "The tables below are parsed from the reference report and reprinted without modification; "
        "they are the sequential VOC baselines used in the comparisons above."
    )
    lines.append("")
    for table in reference_tables:
        if not table.rows:
            continue
        if not _is_relevant_reference(table):
            continue
        title = table.section.lstrip("#").strip()
        caption = table.caption.lstrip("#").strip()
        if caption and caption != title and len(caption) <= 30:
            title = f"{title} — {caption.strip('*：')}"
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(_markdown_table(table.headers, table.rows))
        lines.append("")

    lines.append("## 8. Artifact index")
    lines.append("")
    lines.extend(_artifact_lines(args))
    lines.append(f"- report: `{args.output}`")
    lines.append("")
    return "\n".join(lines)


def _short_test_label(test_key: str) -> str:
    return {"stage-1": "stage-1", "stage-2": "stage-2", "cum-2": "cum-1..2"}[test_key]


def _is_relevant_reference(table: ReferenceTable) -> bool:
    """Select the reference tables that the LoCo report covers."""
    headers = table.headers
    if "seen-4-task-4" in headers:
        return True
    if "method&stage" in headers:
        return True
    if headers and headers[0] == "Class" and "Stage 1" in headers:
        return True
    if headers and headers[0] == "Stage" and "Total train samples" in headers:
        return True
    if headers and headers[0] == "评测集":
        return True
    if headers and headers[0] == "Conf":
        return True
    if headers and headers[0] == "方法" and "背景→增量类" in headers:
        return True
    if headers and headers[0].startswith("对象级统计"):
        return True
    return headers == ("观察", "数值")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--two_root", type=Path, required=True, help="Created 10+10 LoCo dataset root.")
    parser.add_argument("--four_root", type=Path, required=True, help="Created 5+5+5+5 LoCo dataset root.")
    parser.add_argument("--run_two_naive", type=Path, required=True, help="naive 10+10 run directory.")
    parser.add_argument("--run_two_lwf", type=Path, required=True, help="YOLO-LwF 10+10 run directory.")
    parser.add_argument("--run_four_naive", type=Path, required=True, help="naive 5+5+5+5 run directory.")
    parser.add_argument("--run_four_lwf", type=Path, required=True, help="YOLO-LwF 5+5+5+5 run directory.")
    parser.add_argument("--joint_eval_two", type=Path, required=True, help="Joint eval run for 10+10.")
    parser.add_argument("--joint_eval_four", type=Path, required=True, help="Joint eval run for 5+5+5+5.")
    parser.add_argument("--reference", type=Path, default=Path("experiments.md"), help="Reference report.")
    parser.add_argument("--output", type=Path, required=True, help="Output Markdown report.")
    parser.add_argument("--seed", type=int, default=0, help="LoCo assignment seed (report metadata).")
    args = parser.parse_args()
    for path in (args.two_root, args.four_root, args.reference):
        if not path.exists():
            parser.error(f"Path does not exist: {path}")
    return args


def main() -> None:
    """Build and write the LoCo report."""
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
