"""Estimate the lowest i.i.d. keep ratio from the train-set feature distribution.

The accuracy-vs-size curve of an i.i.d. training set follows a saturating power
law (Hestness et al., 2017; Figueroa et al., 2012): past a knee, extra images
fall inside feature neighborhoods the pretrained backbone already represents,
so mAP50 gains collapse (Sorscher et al., 2022). This tool builds that curve
without training. It embeds the full train split, takes the median distance to
the 5th nearest neighbor as the local scale, and measures the fraction of
images an i.i.d. prefix covers at that scale. The knee of the coverage curve
(Satopaa et al., 2011) is the estimated minimum keep ratio.

Inputs are the dataset yaml and the backbone weights. No pruned-subset training
is run.

Usage:
    $ python tools/estimate_keep_ratio.py \\
        --src_data data/VOC-YOLO/VOC.yaml \\
        --backbone yoloe-26m-seg.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ultralytics.utils import LOGGER, YAML

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prune_dataset import (  # noqa: E402
    _build_visual_matrix,
    _collect_records,
    _extract_visual_features,
    _primary_strata,
    _resolve_split_dirs,
    _squared_distances,
)

EPS = 1e-12
DEFAULT_RATIOS = (
    0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0,
)
CENTER_BLOCK = 256
NEIGHBOR_RANK = 5


def class_name_table(names) -> dict[int, str]:
    """Map class id to name from a YOLO ``names`` list or id-keyed dict."""
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise TypeError(f"Dataset names must be a list or mapping, got {type(names)}")


def knee_ratio(ratios: np.ndarray, coverage: np.ndarray) -> float:
    """Kneedle knee of a concave increasing coverage curve.

    Both axes are mapped to the unit square. The knee is the sample whose
    coverage sits farthest above the chord from the first point to the last
    (Satopaa et al., 2011). That is the keep ratio where further i.i.d. images
    add the least new coverage per image.

    Args:
        ratios: (K,) increasing keep ratios in (0, 1].
        coverage: (K,) coverage in [0, 1], same length, nondecreasing in expectation.

    Returns:
        Knee keep ratio (one of ``ratios``).
    """
    if ratios.shape != coverage.shape or ratios.ndim != 1:
        raise ValueError(f"ratios and coverage must be 1-D and aligned, got {ratios.shape} {coverage.shape}")
    if ratios.size < 2:
        raise ValueError(f"Need at least 2 curve points, got {ratios.size}")
    span_x = float(ratios[-1] - ratios[0])
    span_y = float(coverage[-1] - coverage[0])
    if span_x <= EPS or span_y <= EPS:
        # Flat coverage: already saturated at the first prefix, or never covered.
        return float(ratios[0] if float(coverage[0]) >= 0.5 else ratios[-1])
    unit_x = (ratios - ratios[0]) / span_x
    unit_y = (coverage - coverage[0]) / span_y
    deviation = unit_y - unit_x
    return float(ratios[int(np.argmax(deviation))])


def _ratio_grid(n_samples: int, ratios: tuple[float, ...]) -> np.ndarray:
    """Keep ratios that correspond to at least one image and at most ``n_samples``."""
    if n_samples < 2:
        raise ValueError(f"Need at least 2 images to estimate a knee, got {n_samples}")
    counts = {1, n_samples}
    counts.update(min(n_samples, max(1, int(round(n_samples * ratio)))) for ratio in ratios)
    counts = sorted(counts)
    grid = np.asarray(counts, dtype=np.float64) / float(n_samples)
    if grid.size < 2:
        raise ValueError(f"Ratio grid collapsed for n_samples={n_samples}")
    return grid


def local_scale_sq(features: np.ndarray, rank: int) -> float:
    """Median squared distance to the ``rank``-th nearest other point. ``features`` is (N, D)."""
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 features to estimate a local scale, got {n_samples}")
    rank = min(max(rank, 1), n_samples - 1)
    nearest_sq = np.full((n_samples, rank), np.inf, dtype=np.float64)
    for start in range(0, n_samples, CENTER_BLOCK):
        end = min(n_samples, start + CENTER_BLOCK)
        dist_sq = _squared_distances(features, features[start:end])
        for column, index in enumerate(range(start, end)):
            dist_sq[index, column] = np.inf
        merged = np.concatenate([nearest_sq, dist_sq], axis=1)
        nearest_sq = np.partition(merged, rank - 1, axis=1)[:, :rank]
    return float(np.median(nearest_sq[:, -1]))


def coverage_curve(
    features: np.ndarray,
    ratios: np.ndarray,
    rng: np.random.Generator,
    query_size: int,
    tau_sq: float,
) -> np.ndarray:
    """Fraction of images within the local scale of an i.i.d. prefix.

    A query is covered when its nearest selected image is at most ``tau_sq``
    away (squared Euclidean distance on L2-normalized features). ``features``
    is (N, D). Returns coverage (K,) aligned with ``ratios``.
    """
    n_samples = features.shape[0]
    if ratios.ndim != 1 or ratios.size < 2:
        raise ValueError(f"ratios must be (K,) with K>=2, got {ratios.shape}")
    if query_size < 1:
        raise ValueError(f"query_size must be >= 1, got {query_size}")
    if tau_sq < 0:
        raise ValueError(f"tau_sq must be >= 0, got {tau_sq}")
    query = np.arange(n_samples)
    if n_samples > query_size:
        query = rng.choice(n_samples, size=query_size, replace=False)
    queries = features[query]
    order = rng.permutation(n_samples)
    min_sq = np.full(query.shape[0], np.inf, dtype=np.float64)
    cursor = 0
    coverage = np.empty(ratios.shape[0], dtype=np.float64)
    for point_id, ratio in enumerate(ratios):
        target = min(n_samples, max(1, int(round(float(ratio) * n_samples))))
        while cursor < target:
            step = min(CENTER_BLOCK, target - cursor)
            block = features[order[cursor : cursor + step]]
            dist_sq = _squared_distances(queries, block)
            min_sq = np.minimum(min_sq, dist_sq.min(axis=1))
            cursor += step
        coverage[point_id] = float(np.mean(min_sq <= tau_sq + EPS))
    return coverage


def average_coverage(
    features: np.ndarray,
    ratios: np.ndarray,
    n_repeats: int,
    query_size: int,
    seed: int,
    tau_sq: float,
) -> np.ndarray:
    """Average ``coverage_curve`` over ``n_repeats`` i.i.d. orders. Returns (K,)."""
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    curves = [
        coverage_curve(features, ratios, np.random.default_rng(seed + repeat), query_size, tau_sq)
        for repeat in range(n_repeats)
    ]
    return np.mean(np.stack(curves, axis=0), axis=0)


def estimate_split_knee(
    features: np.ndarray,
    primary: np.ndarray,
    names: dict[int, str],
    ratios: tuple[float, ...],
    n_repeats: int,
    query_size: int,
    min_stratum: int,
    seed: int,
) -> dict:
    """Knee keep ratio of one split.

    Coverage is the fraction of images that fall within the dataset's local
    feature scale of a random prefix. The reported ratio is the Kneedle knee of
    that i.i.d. curve on the full split: past it, extra random images mostly
    land in neighborhoods that are already represented. Per-class knees are
    recorded for inspection.

    Args:
        features: (N, D) L2-normalized features aligned with ``primary``.
        primary: (N,) rarest-class stratum id, ``-1`` for images with no boxes.
        names: Class id to name.
        ratios: Candidate keep ratios.
        n_repeats: Random orders averaged into each coverage curve.
        query_size: Query subsample used to estimate mean radius.
        min_stratum: Minimum images for a stratum knee to be eligible.
        seed: Base seed.

    Returns:
        Result dict with ``keep_ratio``, ``keep_count``, per-stratum rows, and curves.
    """
    n_samples = features.shape[0]
    if primary.shape != (n_samples,):
        raise ValueError(f"primary must be ({n_samples},), got {primary.shape}")
    global_ratios = _ratio_grid(n_samples, ratios)
    global_tau_sq = local_scale_sq(features, NEIGHBOR_RANK)
    global_coverage = average_coverage(
        features, global_ratios, n_repeats, query_size, seed, global_tau_sq
    )
    global_knee = knee_ratio(global_ratios, global_coverage)

    strata = []
    eligible: list[float] = []
    for stratum in sorted(int(value) for value in np.unique(primary)):
        member = np.where(primary == stratum)[0]
        n_member = int(member.size)
        label = "background" if stratum < 0 else names.get(stratum, str(stratum))
        row = {
            "stratum": stratum,
            "name": label,
            "n_images": n_member,
            "knee_ratio": None,
            "knee_count": None,
            "eligible": False,
        }
        if n_member >= max(min_stratum, 2):
            stratum_ratios = _ratio_grid(n_member, ratios)
            coverage = average_coverage(
                features[member],
                stratum_ratios,
                n_repeats,
                query_size,
                seed + 10007 * (stratum + 2),
                local_scale_sq(features[member], NEIGHBOR_RANK),
            )
            ratio_hat = knee_ratio(stratum_ratios, coverage)
            row["knee_ratio"] = ratio_hat
            row["knee_count"] = int(round(ratio_hat * n_member))
            row["eligible"] = True
            row["curve"] = [
                {"ratio": float(ratio), "coverage": float(value)}
                for ratio, value in zip(stratum_ratios, coverage)
            ]
            eligible.append(ratio_hat)
        strata.append(row)

    keep_ratio = global_knee
    binding = None
    if eligible:
        binding = max(
            (row for row in strata if row["eligible"]),
            key=lambda row: row["knee_ratio"],
        )
    keep_count = int(max(1, min(n_samples, round(keep_ratio * n_samples))))
    return {
        "n_images": n_samples,
        "keep_ratio": keep_ratio,
        "keep_count": keep_count,
        "global_knee_ratio": global_knee,
        "class_knee_max": None if not eligible else float(max(eligible)),
        "binding_stratum": None if binding is None else binding["name"],
        "strata": [
            {key: value for key, value in row.items() if key != "curve"}
            for row in strata
        ],
        "global_curve": [
            {"ratio": float(ratio), "coverage": float(value)}
            for ratio, value in zip(global_ratios, global_coverage)
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_data", type=Path, required=True, help="Saturated source dataset YAML")
    parser.add_argument("--split", type=str, default="train", help="Split whose size knee is estimated")
    parser.add_argument(
        "--backbone",
        type=str,
        default="yoloe-26m-seg.pt",
        help="Pretrained backbone used as the feature map (same weights as fine-tuning)",
    )
    parser.add_argument("--device", type=str, default="0", help="Feature-extraction device")
    parser.add_argument("--batch_size", type=int, default=32, help="Feature-extraction batch size")
    parser.add_argument("--imgsz", type=int, default=0, help="Feature image size; 0 = extractor default")
    parser.add_argument("--feature_cache_dir", type=Path, default=None, help="Directory of visual-feature caches")
    parser.add_argument("--repeats", type=int, default=4, help="i.i.d. orders averaged into each coverage curve")
    parser.add_argument("--query_size", type=int, default=2048, help="Images used to estimate mean covering radius")
    parser.add_argument(
        "--min_stratum",
        type=int,
        default=12,
        help="Minimum images for a class knee to constrain the dataset ratio",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out", type=Path, default=None, help="Optional JSON result path")
    parser.add_argument("--curve_csv", type=Path, default=None, help="Optional global coverage-curve CSV")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.query_size < 1:
        parser.error("--query_size must be >= 1")
    if args.min_stratum < 2:
        parser.error("--min_stratum must be >= 2")
    return args


def _write_curve_csv(path: Path, curve: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ratio,coverage"]
    lines.extend(f"{row['ratio']:.6f},{row['coverage']:.6f}" for row in curve)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Embed ``--split`` and print the estimated i.i.d. keep ratio."""
    args = parse_args()
    source_cfg = args.src_data.resolve()
    if not source_cfg.is_file():
        raise FileNotFoundError(f"Source dataset yaml does not exist: {source_cfg}")
    if args.backbone.lower() == "none":
        raise ValueError("A pretrained backbone is required to estimate the keep ratio")
    dataset_config = YAML().load(source_cfg)
    if "names" not in dataset_config:
        raise KeyError(f"Source dataset config has no 'names': {source_cfg}")
    names = class_name_table(dataset_config["names"])
    resolved = _resolve_split_dirs(source_cfg, dataset_config, args.split)
    if resolved is None:
        raise KeyError(f"Source dataset config '{source_cfg}' has no '{args.split}' split")
    image_root, label_root = resolved
    records = _collect_records(image_root, label_root, len(names))
    cache_path = None
    if args.feature_cache_dir is not None:
        stem = args.backbone.replace(":", "_").replace("/", "_")
        cache_path = args.feature_cache_dir / f"{args.split}_{stem}.npz"
    visual = _extract_visual_features(
        records,
        backbone=args.backbone,
        device=args.device,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        cache_path=cache_path,
    )
    features = _build_visual_matrix(visual)
    class_counts = np.stack([record.class_counts for record in records], axis=0)
    primary = _primary_strata(class_counts)
    result = estimate_split_knee(
        features,
        primary,
        names,
        ratios=DEFAULT_RATIOS,
        n_repeats=args.repeats,
        query_size=args.query_size,
        min_stratum=args.min_stratum,
        seed=args.seed,
    )
    result["src_data"] = source_cfg.as_posix()
    result["split"] = args.split
    result["backbone"] = args.backbone
    payload = json.dumps(result, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        LOGGER.info(f"Wrote {args.json_out}")
    if args.curve_csv is not None:
        _write_curve_csv(args.curve_csv, result["global_curve"])
        LOGGER.info(f"Wrote {args.curve_csv}")
    print(
        f"keep_ratio={result['keep_ratio']:.4f}  keep_count={result['keep_count']}/{result['n_images']}  "
        f"class_knee_max={result['class_knee_max']}  latest_class={result['binding_stratum']}"
    )
    for row in result["strata"]:
        if row["knee_ratio"] is None:
            print(f"  {row['name']}: n={row['n_images']}  (below min_stratum)")
        else:
            mark = " *" if row["eligible"] and row["name"] == result["binding_stratum"] else ""
            print(
                f"  {row['name']}: n={row['n_images']}  knee_ratio={row['knee_ratio']:.4f}  "
                f"knee_count={row['knee_count']}{mark}"
            )


if __name__ == "__main__":
    main()
