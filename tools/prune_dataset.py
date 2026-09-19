"""Prune a YOLO detection dataset by selecting a subset of existing samples.

Only original images and labels are kept (symlinked). No new samples are synthesized.

Selection is independent per requested split. Splits listed in ``random_splits``
are drawn uniformly without replacement so val/test stay on the source
distribution. Remaining splits use a rarest-class stratified coreset: keep-slots
are allocated per class stratum, then filled by matching the stratum's deep
feature distribution (iCaRL-style herding by default; k-means prototypes or
k-center coverage as alternatives). Without a visual backbone the stratum is
filled by systematic sampling on box count.

Usage:
    $ python tools/prune_dataset.py \\
        --src_data data/VOC-YOLO/VOC.yaml \\
        --out_dir data/VOC-pruned \\
        --ratio 0.1 \\
        --splits train val test \\
        --random_splits val test \\
        --backbone yoloe-26m-seg.pt
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ultralytics.utils import LOGGER, YAML

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from utils import parse_label_line

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp")
SIZE_BIN_EDGES = (0.0, 0.02, 0.05, 0.15, 1.01)
VISUAL_PCA_DIM = 64
EPS = 1e-12
KMEANS_ITERS = 15
SELECTORS = ("herding", "prototype", "kcenter", "pc1", "bucket")
STRATUM_SIZE_EXPONENT = 0.85


@dataclass(frozen=True)
class SampleRecord:
    """One source sample and the label statistics used for coreset scoring."""

    image_path: Path
    label_path: Path | None
    class_counts: np.ndarray  # (num_classes,)
    size_hist: np.ndarray  # (num_size_bins,)
    n_boxes: int


def _resolve_dataset_root(source_cfg: Path, dataset_config: dict) -> Path:
    """Resolve the dataset root used to interpret split paths."""
    configured_root = dataset_config.get("path")
    dataset_root = source_cfg.parent if configured_root is None else Path(configured_root)
    if not dataset_root.is_absolute():
        dataset_root = source_cfg.parent / dataset_root
    return dataset_root.resolve()


def _resolve_split_dirs(source_cfg: Path, dataset_config: dict, split: str) -> tuple[Path, Path] | None:
    """Resolve image and label directories of one split; None when the split key is absent."""
    split_value = dataset_config.get(split)
    if split_value is None:
        return None
    if not isinstance(split_value, str):
        raise TypeError(f"Split '{split}' in '{source_cfg}' must be a path, got {split_value!r}")
    dataset_root = _resolve_dataset_root(source_cfg, dataset_config)
    image_root = Path(split_value)
    if not image_root.is_absolute():
        image_root = dataset_root / image_root
    image_root = image_root.resolve()
    label_root = Path(str(image_root).replace("images", "labels"))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory for split '{split}' does not exist: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory for split '{split}' does not exist: {label_root}")
    return image_root, label_root


def _size_bin(area: float) -> int:
    """Map a normalized box area to a size-histogram bin."""
    for bin_id in range(len(SIZE_BIN_EDGES) - 1):
        if SIZE_BIN_EDGES[bin_id] <= area < SIZE_BIN_EDGES[bin_id + 1]:
            return bin_id
    return len(SIZE_BIN_EDGES) - 2


def _parse_label_stats(label_path: Path, num_classes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse a YOLO label file into class counts, box-size histogram, and box count."""
    class_counts = np.zeros(num_classes, dtype=np.float64)
    size_hist = np.zeros(len(SIZE_BIN_EDGES) - 1, dtype=np.float64)
    n_boxes = 0
    if not label_path.is_file():
        return class_counts, size_hist, n_boxes
    for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            LOGGER.warning(f"Skipping invalid YOLO label in '{label_path}' at line {line_index}")
            continue
        class_id, _, _, width, height = parsed
        if class_id < 0 or class_id >= num_classes:
            LOGGER.warning(
                f"Skipping out-of-range class id {class_id} in '{label_path}' at line {line_index} "
                f"(expected 0..{num_classes - 1})"
            )
            continue
        class_counts[class_id] += 1
        size_hist[_size_bin(max(0.0, width * height))] += 1
        n_boxes += 1
    return class_counts, size_hist, n_boxes


def _collect_records(image_root: Path, label_root: Path, num_classes: int) -> list[SampleRecord]:
    """Pair every image with its label statistics, sorted for deterministic selection."""
    records: list[SampleRecord] = []
    for image_path in sorted(image_root.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        label_path = label_root / f"{image_path.stem}.txt"
        class_counts, size_hist, n_boxes = _parse_label_stats(label_path, num_classes)
        records.append(
            SampleRecord(
                image_path=image_path,
                label_path=label_path if label_path.is_file() else None,
                class_counts=class_counts,
                size_hist=size_hist,
                n_boxes=n_boxes,
            )
        )
    if not records:
        raise RuntimeError(f"No images found under {image_root}")
    return records


def resolve_keep_count(n_samples: int, ratio: float | None, num_out: int | None) -> int:
    """Map a ratio or an absolute count onto a keep size for one split."""
    if n_samples < 1:
        raise ValueError(f"Cannot prune an empty split (n_samples={n_samples})")
    if num_out is not None:
        if num_out < 1:
            raise ValueError(f"num_out must be >= 1, got {num_out}")
        return min(num_out, n_samples)
    if ratio is None:
        raise ValueError("Either ratio or num_out must be specified")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    keep = int(round(n_samples * ratio))
    return max(1, min(keep, n_samples))


def _row_l2(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; all-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    out = matrix / norms
    out[np.linalg.norm(matrix, axis=1) < EPS] = 0.0
    return out


def _pca_reduce(features: np.ndarray, dim: int) -> np.ndarray:
    """Project features onto the leading ``dim`` principal axes.

    Uses the covariance eigenproblem on the smaller side of (N, D) so a split
    with thousands of 2048-d vectors does not run a full rectangular SVD.
    """
    n_samples, n_dims = features.shape
    if n_samples < 2 or n_dims < 2:
        return features.astype(np.float64, copy=True)
    centered = features.astype(np.float64) - features.mean(axis=0)
    keep_dim = min(dim, n_samples - 1, n_dims)
    if keep_dim < 1:
        return centered
    if n_samples >= n_dims:
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, ::-1][:, :keep_dim]
        return centered @ basis
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:keep_dim].T


def _extract_visual_features(
    records: list[SampleRecord],
    backbone: str,
    device: str,
    batch_size: int,
    imgsz: int,
    cache_path: Path | None,
) -> np.ndarray:
    """Extract one visual feature row per record, aligned with ``records`` order.

    Args:
        records: Samples whose ``image_path`` is read.
        backbone: Backbone spec accepted by ``BackboneFeature``.
        device: Torch device string.
        batch_size: Forward batch size.
        imgsz: Input size; 0 lets the extractor use its default.

    Returns:
        features: (N, D) float32 matrix.
    """
    import torch
    from feature_extractors import BackboneFeature, read_image_bgr

    if device.isdigit():
        torch_device = torch.device(f"cuda:{device}")
    else:
        torch_device = torch.device(device)

    keys = [record.image_path.resolve().as_posix() for record in records]
    if cache_path is not None and cache_path.is_file():
        payload = np.load(cache_path, allow_pickle=True)
        cached_keys = [str(key) for key in payload["keys"].tolist()]
        if cached_keys == keys:
            LOGGER.info(f"Loaded cached visual features from {cache_path}")
            return payload["features"].astype(np.float32)
        LOGGER.warning(f"Feature cache '{cache_path}' does not match this split; recomputing")

    extractor = BackboneFeature(backbone, torch_device, batch_size=batch_size, imgsz=imgsz)
    feats, batch_arrays = [], []
    try:
        for record in records:
            batch_arrays.append(extractor._preprocess(read_image_bgr(str(record.image_path))))
            if len(batch_arrays) == batch_size:
                batch = torch.from_numpy(np.stack(batch_arrays)).to(torch_device)
                feats.append(extractor._forward(batch).cpu().numpy())
                batch_arrays = []
        if batch_arrays:
            batch = torch.from_numpy(np.stack(batch_arrays)).to(torch_device)
            feats.append(extractor._forward(batch).cpu().numpy())
    finally:
        extractor.close()
    if not feats:
        raise RuntimeError(f"No visual features extracted from {len(records)} images")
    features = np.concatenate(feats).astype(np.float32)
    if features.shape[0] != len(records):
        raise RuntimeError(
            f"Visual feature rows ({features.shape[0]}) do not match sample count ({len(records)})"
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(keys), features=features)
        LOGGER.info(f"Wrote visual feature cache to {cache_path}")
    return features


def _squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances.

    Args:
        points: (N, D)
        centers: (K, D)

    Returns:
        dist_sq: (N, K)
    """
    if centers.ndim != 2 or points.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got points {points.shape} centers {centers.shape}")
    if points.shape[1] != centers.shape[1]:
        raise ValueError(f"Feature dim mismatch: points {points.shape} vs centers {centers.shape}")
    point_norm = np.einsum("nd,nd->n", points, points)[:, None]
    center_norm = np.einsum("kd,kd->k", centers, centers)[None, :]
    return np.maximum(point_norm + center_norm - 2.0 * (points @ centers.T), 0.0)


def _select_herding(features: np.ndarray, quota: int) -> np.ndarray:
    """Greedy moment matching of the empirical feature mean (iCaRL / Welling herding).

    Args:
        features: (N, D) L2-normalized (or otherwise comparable) vectors.
        quota: Number of local indices to return.

    Returns:
        local_indices: (quota,) int64
    """
    n_samples = features.shape[0]
    if quota < 1 or quota > n_samples:
        raise ValueError(f"herding quota must be in [1, {n_samples}], got {quota}")
    mean = features.mean(axis=0)
    running = np.zeros_like(mean)
    available = np.ones(n_samples, dtype=bool)
    chosen = np.empty(quota, dtype=np.int64)
    for step in range(quota):
        residual = (step + 1) * mean - running
        scores = features @ residual
        scores[~available] = -np.inf
        pick = int(np.argmax(scores))
        if not np.isfinite(scores[pick]):
            raise RuntimeError("Herding exhausted the candidate pool before filling the quota")
        chosen[step] = pick
        available[pick] = False
        running += features[pick]
    return chosen


def _select_kcenter(features: np.ndarray, quota: int) -> np.ndarray:
    """Farthest-first traversal covering the feature support (Sener & Savarese k-center).

    The first point is the sample nearest the stratum mean (a moderate prototype),
    then each new point maximizes its minimum distance to the set already chosen.

    Args:
        features: (N, D)
        quota: Number of local indices to return.

    Returns:
        local_indices: (quota,) int64
    """
    n_samples = features.shape[0]
    if quota < 1 or quota > n_samples:
        raise ValueError(f"k-center quota must be in [1, {n_samples}], got {quota}")
    mean = features.mean(axis=0, keepdims=True)
    start = int(np.argmin(_squared_distances(features, mean)[:, 0]))
    chosen = np.empty(quota, dtype=np.int64)
    chosen[0] = start
    min_dist = _squared_distances(features, features[start : start + 1])[:, 0]
    min_dist[start] = -np.inf
    for step in range(1, quota):
        pick = int(np.argmax(min_dist))
        chosen[step] = pick
        d_new = _squared_distances(features, features[pick : pick + 1])[:, 0]
        min_dist = np.minimum(min_dist, d_new)
        min_dist[pick] = -np.inf
    return chosen


def _kmeans_centroids(features: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    """k-means++ initialization followed by Lloyd iteration.

    Args:
        features: (N, D)
        k: Number of centroids, 1 <= k <= N.
        rng: Used only for k-means++ seeding.

    Returns:
        centroids: (k, D)
    """
    n_samples, n_dims = features.shape
    if k < 1 or k > n_samples:
        raise ValueError(f"k-means k must be in [1, {n_samples}], got {k}")
    centroids = np.empty((k, n_dims), dtype=np.float64)
    centroids[0] = features[int(rng.randrange(n_samples))]
    closest_sq = np.full(n_samples, np.inf, dtype=np.float64)
    for center_id in range(1, k):
        d_last = _squared_distances(features, centroids[center_id - 1 : center_id])[:, 0]
        closest_sq = np.minimum(closest_sq, d_last)
        total = float(closest_sq.sum())
        if total <= EPS:
            centroids[center_id] = features[int(rng.randrange(n_samples))]
            continue
        threshold = rng.random() * total
        pick = int(np.searchsorted(np.cumsum(closest_sq), threshold, side="right"))
        pick = min(max(pick, 0), n_samples - 1)
        centroids[center_id] = features[pick]
    for _ in range(KMEANS_ITERS):
        assign = _squared_distances(features, centroids).argmin(axis=1)
        updated = np.empty_like(centroids)
        moved = False
        for center_id in range(k):
            members = features[assign == center_id]
            if members.shape[0] == 0:
                updated[center_id] = centroids[center_id]
                continue
            updated[center_id] = members.mean(axis=0)
            moved = moved or float(np.max(np.abs(updated[center_id] - centroids[center_id]))) > EPS
        centroids = updated
        if not moved:
            break
    return centroids


def _select_prototype(features: np.ndarray, quota: int, rng: random.Random) -> np.ndarray:
    """One original sample nearest each k-means centroid (feature-mode representatives).

    Args:
        features: (N, D)
        quota: Number of centroids / samples.
        rng: k-means++ RNG.

    Returns:
        local_indices: (quota,) int64
    """
    n_samples = features.shape[0]
    if quota < 1 or quota > n_samples:
        raise ValueError(f"prototype quota must be in [1, {n_samples}], got {quota}")
    centroids = _kmeans_centroids(features, quota, rng)
    dist_sq = _squared_distances(features, centroids)
    used = np.zeros(n_samples, dtype=bool)
    chosen = np.empty(quota, dtype=np.int64)
    center_order = np.argsort(dist_sq.min(axis=0))
    for step, center_id in enumerate(center_order):
        ranking = np.argsort(dist_sq[:, center_id])
        pick = next((int(local) for local in ranking if not used[local]), None)
        if pick is None:
            raise RuntimeError("Prototype selection ran out of unused samples")
        used[pick] = True
        chosen[step] = pick
    return chosen


def _select_systematic(order: np.ndarray, quota: int) -> np.ndarray:
    """Evenly spaced indices along a 1-D order (fallback when no visual features)."""
    if quota < 1 or quota > order.size:
        raise ValueError(f"systematic quota must be in [1, {order.size}], got {quota}")
    return order[(np.arange(quota) * order.size) // quota]


def _primary_strata(class_counts: np.ndarray) -> np.ndarray:
    """Assign each image to its rarest present class; unlabeled images get -1.

    Args:
        class_counts: (N, C) per-image instance counts.

    Returns:
        primary: (N,) int64 stratum ids.
    """
    n_samples = class_counts.shape[0]
    presence = class_counts > 0
    freq = presence.sum(axis=0).astype(np.float64)
    primary = np.full(n_samples, -1, dtype=np.int64)
    for index in range(n_samples):
        classes = np.where(presence[index])[0]
        if classes.size:
            primary[index] = int(classes[np.argmin(freq[classes])])
    return primary


def _build_visual_matrix(visual: np.ndarray) -> np.ndarray:
    """PCA-reduce and L2-normalize visual rows for distribution matching. (N, D')."""
    if visual.ndim != 2:
        raise ValueError(f"visual features must be 2D (N, D), got {visual.shape}")
    reduced = _pca_reduce(visual.astype(np.float64, copy=False), VISUAL_PCA_DIM)
    return _row_l2(reduced)


def _largest_remainder_quotas(sizes: dict[int, int], keep: int) -> dict[int, int]:
    """Allocate ``keep`` integer slots across strata in proportion to stratum size."""
    total = sum(sizes.values())
    if total <= 0 or keep <= 0:
        return {key: 0 for key in sizes}
    raw = {key: keep * size / total for key, size in sizes.items()}
    quotas = {key: int(np.floor(value)) for key, value in raw.items()}
    leftover = keep - sum(quotas.values())
    remainders = sorted(sizes, key=lambda key: (raw[key] - quotas[key], sizes[key]), reverse=True)
    for key in remainders:
        if leftover <= 0:
            break
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            leftover -= 1
    if leftover > 0:
        for key in sorted(sizes, key=lambda item: sizes[item], reverse=True):
            take = min(leftover, sizes[key] - quotas[key])
            quotas[key] += take
            leftover -= take
            if leftover <= 0:
                break
    return quotas


def _select_pc1(features: np.ndarray, secondary: np.ndarray, quota: int) -> np.ndarray:
    """Systematic sample along the leading principal axis of the stratum features.

    Args:
        features: (N, D) stratum vectors.
        secondary: (N,) tie-breaker (typically box counts).
        quota: Number of local indices to return.

    Returns:
        local_indices: (quota,) int64 into the stratum.
    """
    n_samples, n_dims = features.shape
    if quota < 1 or quota > n_samples:
        raise ValueError(f"pc1 quota must be in [1, {n_samples}], got {quota}")
    if secondary.shape != (n_samples,):
        raise ValueError(f"secondary must be (N,), got {secondary.shape} vs N={n_samples}")
    centered = features - features.mean(axis=0, keepdims=True)
    if n_samples >= 2 and n_dims >= 1:
        if n_dims <= n_samples:
            covariance = centered.T @ centered
            _, eigenvectors = np.linalg.eigh(covariance)
            scores = centered @ eigenvectors[:, -1]
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            scores = centered @ vt[0]
    else:
        scores = np.zeros(n_samples, dtype=np.float64)
    z_feat = (scores - scores.mean()) / (float(scores.std()) + EPS)
    log_boxes = np.log1p(np.clip(secondary.astype(np.float64), 0.0, None))
    z_boxes = (log_boxes - log_boxes.mean()) / (float(log_boxes.std()) + EPS)
    axis = z_feat + z_boxes
    order = np.lexsort((np.arange(n_samples), axis))
    return _select_systematic(order, quota)


def _select_bucket(features: np.ndarray, secondary: np.ndarray, quota: int) -> np.ndarray:
    """Pick the feature-mean prototype in each box-count systematic bucket.

    Images are ordered by ``secondary`` (box count) and split into ``quota``
    equal-size slices; each slice contributes the sample nearest that slice's
    empirical feature mean. This covers the source density distribution while
    matching the local appearance mean.

    Args:
        features: (N, D) stratum vectors.
        secondary: (N,) box counts.
        quota: Number of local indices to return.

    Returns:
        local_indices: (quota,) int64
    """
    n_samples = features.shape[0]
    if quota < 1 or quota > n_samples:
        raise ValueError(f"bucket quota must be in [1, {n_samples}], got {quota}")
    if secondary.shape != (n_samples,):
        raise ValueError(f"secondary must be (N,), got {secondary.shape} vs N={n_samples}")
    order = np.lexsort((np.arange(n_samples), secondary))
    used = np.zeros(n_samples, dtype=bool)
    chosen = np.empty(quota, dtype=np.int64)
    for step in range(quota):
        start = (step * n_samples) // quota
        end = max(((step + 1) * n_samples) // quota, start + 1)
        bucket = [int(index) for index in order[start:end] if not used[index]]
        if not bucket:
            bucket = [int(index) for index in order if not used[index]]
        if not bucket:
            raise RuntimeError("Bucket selection exhausted the candidate pool before filling the quota")
        feat_bucket = features[bucket]
        center = feat_bucket.mean(axis=0, keepdims=True)
        pick = bucket[int(np.argmin(_squared_distances(feat_bucket, center)[:, 0]))]
        used[pick] = True
        chosen[step] = pick
    return chosen


def select_stratified_coreset(
    records: list[SampleRecord],
    keep: int,
    visual: np.ndarray | None,
    rng: random.Random,
    selector: str = "pc1",
) -> list[int]:
    """Select ``keep`` images by rarest-class strata plus feature-distribution matching.

    Each image is assigned to the rarest class it contains (background images
    form their own stratum). Strata receive keep-slots in proportion to
    ``n_s ** STRATUM_SIZE_EXPONENT`` (largest-remainder rounding, capped at
    stratum size). Inside a stratum:

    - with visual features: match that stratum's empirical feature distribution
      via ``selector`` (``pc1`` default, or ``bucket`` / ``kcenter`` / ``herding`` /
      ``prototype``);
    - without visual features: systematic sample along box-count order.

    Args:
        records: Split samples in deterministic file-name order.
        keep: Target subset size.
        visual: (N, D) backbone features aligned with ``records``, or None.
        rng: RNG for prototype k-means++; unused by pc1 / herding / k-center / bucket.
        selector: Within-stratum feature rule.

    Returns:
        selected: Length-``keep`` indices into ``records``.
    """
    if selector not in SELECTORS:
        raise ValueError(f"Unknown selector {selector!r}; expected one of {SELECTORS}")
    n_samples = len(records)
    if visual is not None and visual.shape[0] != n_samples:
        raise ValueError(
            f"visual rows ({visual.shape[0]}) do not match sample count ({n_samples})"
        )
    if keep >= n_samples:
        return list(range(n_samples))
    class_counts = np.stack([record.class_counts for record in records], axis=0)
    n_boxes = class_counts.sum(axis=1)
    primary = _primary_strata(class_counts)
    sizes = {int(stratum): int(np.sum(primary == stratum)) for stratum in np.unique(primary)}
    boosted = {stratum: max(1.0, size ** STRATUM_SIZE_EXPONENT) for stratum, size in sizes.items()}
    boosted_int = {stratum: max(1, int(round(weight))) for stratum, weight in boosted.items()}
    quotas = _largest_remainder_quotas(boosted_int, keep)
    quotas = {stratum: min(quotas[stratum], sizes[stratum]) for stratum in quotas}

    features = None if visual is None else _build_visual_matrix(visual)
    selected: list[int] = []
    taken = np.zeros(n_samples, dtype=bool)
    for stratum, quota in quotas.items():
        if quota <= 0:
            continue
        members = np.where(primary == stratum)[0]
        if quota >= members.size:
            chosen_global = members
        elif features is None:
            order = members[np.lexsort((members, n_boxes[members]))]
            chosen_global = _select_systematic(order, quota)
        else:
            stratum_feat = features[members]
            if selector == "herding":
                local = _select_herding(stratum_feat, quota)
            elif selector == "kcenter":
                local = _select_kcenter(stratum_feat, quota)
            elif selector == "pc1":
                local = _select_pc1(stratum_feat, n_boxes[members], quota)
            elif selector == "bucket":
                local = _select_bucket(stratum_feat, n_boxes[members], quota)
            else:
                local = _select_prototype(stratum_feat, quota, rng)
            chosen_global = members[local]
        for index in chosen_global:
            index = int(index)
            if taken[index]:
                continue
            selected.append(index)
            taken[index] = True
            if len(selected) >= keep:
                return _repair_class_coverage(records, selected)
    if len(selected) < keep:
        remaining = np.where(~taken)[0]
        rest = remaining[np.lexsort((remaining, n_boxes[remaining]))]
        for index in rest:
            selected.append(int(index))
            if len(selected) >= keep:
                break
    return _repair_class_coverage(records, selected[:keep])

def _repair_class_coverage(records: list[SampleRecord], selected: list[int]) -> list[int]:
    """Swap in unused images so every source class that can fit is represented."""
    class_counts = np.stack([record.class_counts for record in records], axis=0)
    source_support = class_counts.sum(axis=0) > 0
    selected_set = set(selected)
    subset_support = class_counts[selected].sum(axis=0) > 0 if selected else np.zeros_like(source_support)
    missing = np.where(source_support & ~subset_support)[0]
    if missing.size == 0:
        return selected
    selected = list(selected)
    for class_id in missing:
        unused = [index for index in range(len(records)) if index not in selected_set and class_counts[index, class_id] > 0]
        if not unused:
            continue
        replacement = unused[0]
        drop_at = _index_of_redundant_sample(class_counts, selected, source_support)
        if drop_at is None:
            break
        dropped = selected[drop_at]
        selected[drop_at] = replacement
        selected_set.remove(dropped)
        selected_set.add(replacement)
    return selected


def _index_of_redundant_sample(
    class_counts: np.ndarray,
    selected: list[int],
    source_support: np.ndarray,
) -> int | None:
    """Return the selected index whose removal drops no uniquely covered class."""
    selected_counts = class_counts[selected]
    coverage = selected_counts.sum(axis=0)
    for position, sample_index in enumerate(selected):
        remaining = coverage - class_counts[sample_index]
        still_covers = (remaining > 0) | ~source_support
        if np.all(still_covers):
            return position
    return None


def select_random(records: list[SampleRecord], keep: int, rng: random.Random) -> list[int]:
    """Uniform sample of ``keep`` indices without replacement."""
    n_samples = len(records)
    if keep >= n_samples:
        return list(range(n_samples))
    return rng.sample(range(n_samples), keep)


def _materialize_split(records: list[SampleRecord], indices: list[int], output_dir: Path, split: str) -> int:
    """Symlink selected images and labels into the YOLO output layout."""
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in indices:
        record = records[index]
        destination_image = image_dir / record.image_path.name
        if destination_image.exists():
            continue
        destination_image.symlink_to(record.image_path.resolve())
        if record.label_path is not None:
            destination_label = label_dir / record.label_path.name
            destination_label.symlink_to(record.label_path.resolve())
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_data", type=Path, required=True, help="Source dataset YAML")
    parser.add_argument("--out_dir", type=Path, required=True, help="Pruned dataset output directory")
    parser.add_argument("--ratio", type=float, default=None, help="Keep ratio per compressed split, in (0, 1]")
    parser.add_argument(
        "--num_out",
        type=int,
        default=None,
        help="Absolute keep count per compressed split; when set, --ratio is ignored",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to write into the output dataset",
    )
    parser.add_argument(
        "--random_splits",
        nargs="*",
        default=None,
        help="Among --splits, those sampled uniformly instead of the coreset (default: val test)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random splits and any ties")
    parser.add_argument(
        "--backbone",
        type=str,
        default="yoloe-26m-seg.pt",
        help="Visual backbone for coreset splits: local .pt, torchvision:<name>, timm:<name>, or 'none'",
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="pc1",
        choices=list(SELECTORS),
        help="Within-stratum feature rule: pc1 (default), bucket, kcenter, herding, or prototype",
    )
    parser.add_argument("--device", type=str, default="0", help="Device for visual feature extraction")
    parser.add_argument("--batch_size", type=int, default=32, help="Feature-extraction batch size")
    parser.add_argument("--imgsz", type=int, default=0, help="Feature-extraction image size; 0 = extractor default")
    parser.add_argument(
        "--feature_cache_dir",
        type=Path,
        default=None,
        help="Directory for per-split visual feature caches (skipped when --backbone none)",
    )
    args = parser.parse_args()
    if args.num_out is None and args.ratio is None:
        parser.error("Specify --ratio or --num_out")
    if args.num_out is None and not 0.0 < args.ratio <= 1.0:
        parser.error("--ratio must be in (0, 1]")
    if args.num_out is not None and args.num_out < 1:
        parser.error("--num_out must be >= 1")
    if args.random_splits is None:
        args.random_splits = ["val", "test"]
    unknown_random = [split for split in args.random_splits if split not in args.splits]
    if unknown_random:
        LOGGER.warning(f"--random_splits not in --splits will be ignored: {unknown_random}")
    return args


def main() -> None:
    """Prune each requested split and write a YOLO dataset yaml with relative paths."""
    args = parse_args()
    source_cfg = args.src_data.resolve()
    if not source_cfg.is_file():
        raise FileNotFoundError(f"Source dataset yaml does not exist: {source_cfg}")
    dataset_config = YAML().load(source_cfg)
    if "names" not in dataset_config:
        raise KeyError(f"Source dataset config has no 'names': {source_cfg}")
    names = dataset_config["names"]
    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, list):
        num_classes = len(names)
    else:
        raise TypeError(f"Dataset names must be a list or mapping, got {type(names)}")

    output_dir = args.out_dir
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink() or output_dir.is_file():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    random_splits = set(args.random_splits) & set(args.splits)
    output_config = {"names": names}
    for split in args.splits:
        resolved = _resolve_split_dirs(source_cfg, dataset_config, split)
        if resolved is None:
            LOGGER.warning(f"Source dataset config '{source_cfg}' has no '{split}' split; skipping")
            continue
        image_root, label_root = resolved
        records = _collect_records(image_root, label_root, num_classes)
        keep = resolve_keep_count(len(records), args.ratio, args.num_out)
        if split in random_splits:
            indices = select_random(records, keep, rng)
            method = "random"
        else:
            visual = None
            if args.backbone.lower() != "none":
                visual = _extract_visual_features(
                    records,
                    backbone=args.backbone,
                    device=args.device,
                    batch_size=args.batch_size,
                    imgsz=args.imgsz,
                    cache_path=(
                        None
                        if args.feature_cache_dir is None
                        else args.feature_cache_dir / f"{split}_{args.backbone.replace(':', '_').replace('/', '_')}.npz"
                    ),
                )
            indices = select_stratified_coreset(
                records,
                keep,
                visual,
                rng,
                selector=args.selector,
            )
            method = f"stratified-{args.selector}"
        written = _materialize_split(records, indices, output_dir, split)
        output_config[split] = f"images/{split}"
        LOGGER.info(
            f"{split}: kept {written}/{len(records)} images via {method} "
            f"(target {keep}, ratio={args.ratio}, num_out={args.num_out})"
        )

    if len(output_config) == 1:
        raise RuntimeError(f"No splits were written to {output_dir}; check --splits against {source_cfg}")
    YAML().save(output_dir / "dataset.yaml", output_config)
    LOGGER.info(f"Pruned dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
