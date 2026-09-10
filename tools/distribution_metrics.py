"""Distribution-level distance/similarity metrics between two feature sample
matrices.

Uniform interface: every metric is called as ``metric(X, Y, rng)`` with
``X`` of shape (N1, D) and ``Y`` of shape (N2, D), and returns a float.
``rng`` is a ``numpy.random.Generator`` for subsampling / projections.

Each metric declares:

- ``similarity``: True means larger = more similar (mean cosine, histogram
  intersection); False means larger = more different (distances,
  divergences, domain-classifier AUC).
- ``histogram_only``: True for metrics defined on non-negative
  (L1-normalized) histogram features.

Available metrics (``--metrics`` names): ``mean_cosine``, ``mmd_rbf_linear``
(linear-time MMD, default for large sets), ``mmd_rbf`` (quadratic MMD on a
subsample), ``frechet``, ``sliced_wasserstein``, ``energy_distance``,
``js_divergence``, ``hist_intersection``, ``chi2_distance``, ``domain_auc``.
"""

import math
from typing import Callable, Dict, List

import numpy as np

# Lower bound for denominators / histogram entries.
EPS = 1e-12

# Sample caps for quadratic-complexity metrics (per side).
MMD_QUADRATIC_CAP = 1024
ENERGY_CAP = 2000
SW_CAP = 5000
AUC_CAP = 4000

# Fréchet distance is computed on D x D covariances; refuse pathological
# dimensions (PCA-reduce first via --pca-dim).
FRECHET_MAX_DIM = 4096


def _check_pair(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate the common metric input contract: two (N, D) float matrices
    with matching D and at least 2 rows each."""
    for name, arr in (('X', X), ('Y', Y)):
        if arr.ndim != 2:
            raise ValueError(f'{name} must be a 2D (N, D) matrix, got shape '
                             f'{arr.shape}.')
        if arr.shape[0] < 2:
            raise ValueError(f'{name} needs at least 2 samples, got '
                             f'{arr.shape[0]}.')
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f'Feature dim mismatch: X has {X.shape[1]}, Y has '
                         f'{Y.shape[1]}.')


def _subsample(X: np.ndarray, cap: int, rng: np.random.Generator
               ) -> np.ndarray:
    """Return ``X`` unchanged or a random row subset of size ``cap``."""
    if len(X) <= cap:
        return X
    return X[rng.choice(len(X), cap, replace=False)]


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared euclidean distances between the rows of A (N, D) and B (M, D),
    shape (N, M)."""
    aa = (A * A).sum(axis=1, keepdims=True)  # (N, 1)
    bb = (B * B).sum(axis=1, keepdims=True)  # (M, 1)
    d2 = aa + bb.T - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


def _median_sigma(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                  cap: int = 1000) -> float:
    """RBF kernel bandwidth by the median heuristic over a combined
    subsample."""
    Z = np.concatenate([_subsample(X, cap, rng), _subsample(Y, cap, rng)])
    d2 = _sq_dists(Z.astype(np.float64), Z.astype(np.float64))
    positive = d2[d2 > 0]
    if positive.size == 0:  # identical points everywhere
        return 1.0
    return float(math.sqrt(np.median(positive)))


def _rbf(d2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mean_cosine(X: np.ndarray, Y: np.ndarray,
                rng: np.random.Generator) -> float:
    """Cosine similarity between the per-dataset mean features."""
    _check_pair(X, Y)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    denom = max(np.linalg.norm(mx) * np.linalg.norm(my), EPS)
    return float(mx @ my / denom)


def mmd_rbf_linear(X: np.ndarray, Y: np.ndarray,
                   rng: np.random.Generator) -> float:
    """Linear-time MMD with an RBF kernel (median heuristic bandwidth).

    Pairs consecutive samples of the shuffled sets: O(N) kernel evaluations,
    suited for large image sets. Returns sqrt(max(MMD^2, 0)).
    """
    _check_pair(X, Y)
    sigma = _median_sigma(X, Y, rng)
    Xs = X[rng.permutation(len(X))].astype(np.float64)
    Ys = Y[rng.permutation(len(Y))].astype(np.float64)
    m = min(len(Xs), len(Ys))
    m -= m % 2
    xa, xb = Xs[0:m:2], Xs[1:m:2]
    ya, yb = Ys[0:m:2], Ys[1:m:2]

    def k(A, B):  # aligned-pair RBF kernel, (m/2,)
        return np.exp(-((A - B) ** 2).sum(axis=1) / (2.0 * sigma * sigma))

    h = k(xa, xb) + k(ya, yb) - k(xa, yb) - k(xb, ya)
    return float(math.sqrt(max(h.mean(), 0.0)))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> float:
    """Quadratic (biased, diagonals included) MMD with an RBF kernel on a
    subsample of at most ``MMD_QUADRATIC_CAP`` rows per side. Returns
    sqrt(max(MMD^2, 0))."""
    _check_pair(X, Y)
    Xs = _subsample(X, MMD_QUADRATIC_CAP, rng).astype(np.float64)
    Ys = _subsample(Y, MMD_QUADRATIC_CAP, rng).astype(np.float64)
    sigma = _median_sigma(X, Y, rng)
    kxx = _rbf(_sq_dists(Xs, Xs), sigma).mean()
    kyy = _rbf(_sq_dists(Ys, Ys), sigma).mean()
    kxy = _rbf(_sq_dists(Xs, Ys), sigma).mean()
    return float(math.sqrt(max(kxx + kyy - 2.0 * kxy, 0.0)))


def frechet(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> float:
    """Fréchet distance between Gaussian approximations of the two feature
    distributions (compares means and covariances; FID-style)."""
    from scipy.linalg import sqrtm
    _check_pair(X, Y)
    if X.shape[1] > FRECHET_MAX_DIM:
        raise RuntimeError(
            f'frechet needs DxD covariances; D={X.shape[1]} exceeds '
            f'{FRECHET_MAX_DIM}. Reduce features first (--pca-dim).')
    X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
    mu1, mu2 = X64.mean(axis=0), Y64.mean(axis=0)
    cov1 = np.cov(X64, rowvar=False)
    cov2 = np.cov(Y64, rowvar=False)
    reg = 1e-6 * np.eye(X.shape[1])
    covmean = sqrtm((cov1 + reg) @ (cov2 + reg))
    if np.iscomplexobj(covmean):  # numerical artifact: tiny imaginary part
        if abs(covmean.imag).max() > 1e-3:
            raise RuntimeError('frechet: sqrtm returned a substantially '
                               'complex matrix; feature covariances are '
                               'ill-conditioned.')
        covmean = covmean.real
    diff = mu1 - mu2
    value = diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean)
    return float(max(value, 0.0))


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                       rng: np.random.Generator, n_proj: int = 64) -> float:
    """Sliced Wasserstein distance: mean 1D Wasserstein-1 distance over
    random unit projections."""
    from scipy.stats import wasserstein_distance
    _check_pair(X, Y)
    Xs = _subsample(X, SW_CAP, rng).astype(np.float64)
    Ys = _subsample(Y, SW_CAP, rng).astype(np.float64)
    proj = rng.standard_normal((X.shape[1], n_proj))
    proj /= np.clip(np.linalg.norm(proj, axis=0, keepdims=True), EPS, None)
    px, py = Xs @ proj, Ys @ proj  # (N, n_proj)
    total = 0.0
    for i in range(n_proj):
        total += wasserstein_distance(px[:, i], py[:, i])
    return float(total / n_proj)


def energy_distance(X: np.ndarray, Y: np.ndarray,
                    rng: np.random.Generator) -> float:
    """Energy distance: 2*E||x-y|| - E||x-x'|| - E||y-y'|| on subsamples of
    at most ``ENERGY_CAP`` rows per side (within-set terms exclude the
    diagonal)."""
    _check_pair(X, Y)
    Xs = _subsample(X, ENERGY_CAP, rng).astype(np.float64)
    Ys = _subsample(Y, ENERGY_CAP, rng).astype(np.float64)
    dxy = np.sqrt(_sq_dists(Xs, Ys)).mean()
    dxx = np.sqrt(_sq_dists(Xs, Xs)).sum() / (len(Xs) * (len(Xs) - 1))
    dyy = np.sqrt(_sq_dists(Ys, Ys)).sum() / (len(Ys) * (len(Ys) - 1))
    return float(max(2.0 * dxy - dxx - dyy, 0.0))


def _mean_hist(X: np.ndarray) -> np.ndarray:
    """L1-normalized mean histogram of the rows of X (N, D) -> (D,)."""
    hist = X.astype(np.float64).mean(axis=0)
    total = hist.sum()
    if total <= 0:
        raise RuntimeError('Zero-sum histogram feature; histogram metrics '
                           'are undefined.')
    return hist / total


def js_divergence(X: np.ndarray, Y: np.ndarray,
                  rng: np.random.Generator) -> float:
    """Jensen-Shannon divergence (natural log) between the two mean
    histograms. 0 = identical."""
    _check_pair(X, Y)
    p, q = _mean_hist(X), _mean_hist(Y)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def hist_intersection(X: np.ndarray, Y: np.ndarray,
                      rng: np.random.Generator) -> float:
    """Histogram intersection similarity between the two mean histograms.
    1 = identical."""
    _check_pair(X, Y)
    p, q = _mean_hist(X), _mean_hist(Y)
    return float(np.minimum(p, q).sum())


def chi2_distance(X: np.ndarray, Y: np.ndarray,
                  rng: np.random.Generator) -> float:
    """Chi-square distance sum((p-q)^2/(p+q)) between the two mean
    histograms. 0 = identical."""
    _check_pair(X, Y)
    p, q = _mean_hist(X), _mean_hist(Y)
    return float(((p - q) ** 2 / (p + q + EPS)).sum())


def domain_auc(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
               folds: int = 5) -> float:
    """Domain-classifier AUC: cross-validated logistic regression separates
    the two sample sets; the mean ROC AUC measures distributional
    separability. 0.5 = indistinguishable, 1.0 = perfectly separable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    _check_pair(X, Y)
    Xs = _subsample(X, AUC_CAP, rng).astype(np.float64)
    Ys = _subsample(Y, AUC_CAP, rng).astype(np.float64)
    folds = min(folds, len(Xs), len(Ys))
    if folds < 2:
        raise RuntimeError(f'domain_auc needs at least 2 samples per side, '
                           f'got {len(Xs)} and {len(Ys)}.')
    Z = np.concatenate([Xs, Ys])
    labels = np.array([0] * len(Xs) + [1] * len(Ys))
    Z = StandardScaler().fit_transform(Z)
    seed = int(rng.integers(2 ** 31))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(Z, labels):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z[train_idx], labels[train_idx])
        scores = clf.decision_function(Z[test_idx])
        aucs.append(roc_auc_score(labels[test_idx], scores))
    return float(np.mean(aucs))


class Metric:
    """One distribution metric under the uniform ``(X, Y, rng) -> float``
    interface."""

    def __init__(self, name: str, fn: Callable, similarity: bool,
                 histogram_only: bool, description: str):
        self.name = name
        self.fn = fn
        self.similarity = similarity
        self.histogram_only = histogram_only
        self.description = description

    def __call__(self, X: np.ndarray, Y: np.ndarray,
                 rng: np.random.Generator) -> float:
        return self.fn(X, Y, rng)


METRIC_REGISTRY: Dict[str, Metric] = {m.name: m for m in [
    Metric('mean_cosine', mean_cosine, similarity=True, histogram_only=False,
           description='cosine similarity of per-dataset mean features'),
    Metric('mmd_rbf_linear', mmd_rbf_linear, similarity=False,
           histogram_only=False,
           description='linear-time MMD, RBF kernel, median heuristic'),
    Metric('mmd_rbf', mmd_rbf, similarity=False, histogram_only=False,
           description=f'quadratic MMD, RBF kernel, <='
                       f'{MMD_QUADRATIC_CAP} samples per side'),
    Metric('frechet', frechet, similarity=False, histogram_only=False,
           description='Fréchet distance between Gaussian approximations '
                       '(mean + covariance)'),
    Metric('sliced_wasserstein', sliced_wasserstein, similarity=False,
           histogram_only=False,
           description='sliced Wasserstein-1 distance over random '
                       'projections'),
    Metric('energy_distance', energy_distance, similarity=False,
           histogram_only=False,
           description='energy distance (E-statistic)'),
    Metric('js_divergence', js_divergence, similarity=False,
           histogram_only=True,
           description='JS divergence between mean histograms'),
    Metric('hist_intersection', hist_intersection, similarity=True,
           histogram_only=True,
           description='histogram intersection of mean histograms'),
    Metric('chi2_distance', chi2_distance, similarity=False,
           histogram_only=True,
           description='chi-square distance between mean histograms'),
    Metric('domain_auc', domain_auc, similarity=False, histogram_only=False,
           description='domain-classifier AUC (0.5 = indistinguishable)'),
]}


def build_metrics(spec: str, feature_kind: str) -> List[Metric]:
    """Resolve a '+'- or ','-joined metric spec ('all' = every metric
    compatible with the feature kind) against the registry."""
    if spec == 'all':
        return [m for m in METRIC_REGISTRY.values()
                if not m.histogram_only or feature_kind == 'histogram']
    names = [name for part in spec.split('+')
             for name in part.split(',') if name.strip()]
    names = [name.strip() for name in names]
    unknown = [name for name in names if name not in METRIC_REGISTRY]
    if unknown:
        raise ValueError(f'Unknown metric(s): {unknown}. Available: '
                         f'{sorted(METRIC_REGISTRY)} (or "all").')
    metrics = [METRIC_REGISTRY[name] for name in names]
    if feature_kind != 'histogram':
        bad = [m.name for m in metrics if m.histogram_only]
        if bad:
            raise ValueError(f'Metric(s) {bad} require histogram features, '
                             f'but the current feature combination is dense. '
                             f'Use a single histogram block (color_hist, lbp, '
                             f'sift_bovw, dense_sift_bovw) without PCA.')
    return metrics
