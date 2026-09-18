"""Summarize prune-vs-random comparison rows and decide whether the coreset passed.

A comparison CSV has one row per (ratio, method) with a mAP50 column. The test
passes when there is a contiguous ratio interval of at least ``--min_interval``
ratios where the coreset mAP50 is strictly higher than random at every point.

Usage:
    $ python tools/summarize_prune_compare.py --csv runs/prune_compare/voc/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_map_by_ratio(csv_path: Path) -> dict[float, dict[str, float]]:
    """Load mAP50 keyed by ratio then method ('coreset' / 'random')."""
    table: dict[float, dict[str, float]] = defaultdict(dict)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ratio" not in reader.fieldnames or "method" not in reader.fieldnames:
            raise KeyError(f"CSV '{csv_path}' must have 'ratio' and 'method' columns")
        metric = "mAP50" if "mAP50" in reader.fieldnames else None
        if metric is None:
            raise KeyError(f"CSV '{csv_path}' must have a mAP50 column")
        for row in reader:
            ratio = float(row["ratio"])
            method = row["method"].strip()
            table[ratio][method] = float(row[metric])
    return dict(table)


def longest_win_interval(table: dict[float, dict[str, float]]) -> list[float]:
    """Return the longest contiguous ratio run where coreset mAP50 > random."""
    ratios = sorted(table)
    best: list[float] = []
    current: list[float] = []
    for ratio in ratios:
        methods = table[ratio]
        if "coreset" not in methods or "random" not in methods:
            raise KeyError(f"Ratio {ratio} is missing coreset or random mAP50")
        if methods["coreset"] > methods["random"]:
            current.append(ratio)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Comparison summary CSV")
    parser.add_argument(
        "--min_interval",
        type=int,
        default=2,
        help="Minimum number of consecutive ratios where coreset must beat random",
    )
    return parser.parse_args()


def main() -> None:
    """Print the comparison table and exit 0 iff the coreset interval test passes."""
    args = parse_args()
    table = load_map_by_ratio(args.csv)
    print("ratio\trandom\tcoreset\tdelta")
    for ratio in sorted(table):
        random_map = table[ratio]["random"]
        coreset_map = table[ratio]["coreset"]
        print(f"{ratio:.4f}\t{random_map:.5f}\t{coreset_map:.5f}\t{coreset_map - random_map:+.5f}")
    interval = longest_win_interval(table)
    if len(interval) >= args.min_interval:
        print(f"PASS: coreset > random on ratios {interval}")
        return
    print(
        f"FAIL: no contiguous interval of {args.min_interval} ratios with coreset > random "
        f"(best run: {interval or 'none'})"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
