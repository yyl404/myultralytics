"""Pluggable per-image feature extractors for dataset image-domain comparison.

Every extractor implements the :class:`FeatureExtractor` interface:

- ``fit(image_lists)``: optional preparation over the combined image sets of
  all compared datasets (fitting a PCA basis or a BoVW codebook). No-op by
  default.
- ``extract(image_paths)``: per-image feature rows, float32 matrix (N, D).

Each extractor declares a feature ``kind``:

- ``'dense'``: unconstrained real vectors; scaled per dimension by the
  combined-dataset std (no centering) before the metrics.
- ``'histogram'``: non-negative vectors; rows L1-normalized. Only histogram
  features admit the histogram-specific metrics (JS divergence, histogram
  intersection, chi-square distance).

Blocks are composable: :class:`CompositeExtractor` concatenates several
extractors, with an optional per-block PCA reduction (fitted on a sample of
the combined datasets) before concatenation.

Available blocks (``--features`` names): ``backbone``, ``pixel_pca``,
``color_hist``, ``color_moments``, ``lbp``, ``hog``, ``gabor``, ``glcm``,
``sift_bovw``, ``dense_sift_bovw``.
"""

import math
import os.path as osp
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from feature_drift import build_model, extract_backbone_last_feat, get_backbone_last_hook
from torch import nn
from tqdm import tqdm

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER

# ImageNet statistics for generic (timm / torchvision) backbone inputs.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Lower bound for row sums / std denominators.
EPS = 1e-12


def read_image_bgr(path: str) -> np.ndarray:
    """Read one image as (H, W, 3) BGR uint8; raise on unreadable files."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale ``img`` so its longer side is at most ``max_side`` pixels."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def compute_per_image(image_paths: List[str], fn, desc: str) -> np.ndarray:
    """Apply ``fn`` to every image, returning one feature row per image.

    Per-item failures (unreadable / corrupt files) are logged and skipped;
    if no image yields a feature, raise.
    """
    feats, skipped = [], 0
    for path in tqdm(image_paths, desc=desc):
        try:
            feats.append(fn(read_image_bgr(path)))
        except Exception as e:
            skipped += 1
            LOGGER.warning(f'Skipping image {path}: {type(e).__name__}: {e}')
    if not feats:
        raise RuntimeError(f'No usable image among {len(image_paths)} paths '
                           f'({skipped} failed).')
    if skipped:
        LOGGER.warning(f'{skipped}/{len(image_paths)} images failed and were '
                       f'skipped.')
    return np.stack(feats).astype(np.float32)


def sample_fit_paths(image_lists: List[List[str]], max_total: int,
                     seed: int = 0) -> List[str]:
    """Deterministic round-robin sample of up to ``max_total`` paths across
    the per-dataset image lists."""
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(len(paths)).tolist() for paths in image_lists]
    cursors = [0] * len(image_lists)
    sample, added = [], True
    while added and len(sample) < max_total:
        added = False
        for li, paths in enumerate(image_lists):
            if cursors[li] < len(paths) and len(sample) < max_total:
                sample.append(paths[shuffled[li][cursors[li]]])
                cursors[li] += 1
                added = True
    return sample


class FeatureExtractor:
    """Base interface of one pluggable per-image feature block."""

    kind = 'dense'  # 'dense' or 'histogram'
    # True when extract() applies a centering reduction (PCA), so the raw
    # (uncentered) features differ and mean-feature cosine must use them.
    raw_is_different = False

    def fit(self, image_lists: List[List[str]]) -> None:
        """Optional preparation over all compared per-dataset image lists."""

    def extract(self, image_paths: List[str]) -> np.ndarray:
        """Return per-image feature rows, float32 matrix (N, D)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release held resources (models, hooks). No-op by default."""


class _TorchvisionTrunk(nn.Module):
    """A torchvision model minus its last module (classifier), pooled to a
    single (B, C) vector per image."""

    def __init__(self, model: nn.Module):
        super().__init__()
        children = list(model.children())
        if len(children) < 2:
            raise RuntimeError('Torchvision model has fewer than 2 top-level '
                               'modules; cannot strip the classifier.')
        self.trunk = nn.Sequential(*children[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        if feat.ndim == 4:  # (B, C, H, W) -> (B, C)
            feat = F.adaptive_avg_pool2d(feat, 1)
        return feat.flatten(1)


def _load_torchvision_backbone(name: str) -> nn.Module:
    """Build a torchvision model with default pretrained weights, minus its
    classifier module."""
    from torchvision.models import get_model
    try:
        model = get_model(name, weights='DEFAULT')
    except Exception as e:
        raise RuntimeError(
            f'Failed to load torchvision model {name!r} with pretrained '
            f'weights (unknown name or weights download failed): {e}') from e
    return _TorchvisionTrunk(model)


def _load_timm_backbone(name: str) -> nn.Module:
    """Build a timm model with pretrained weights and pooled feature output
    (B, C)."""
    import timm
    try:
        return timm.create_model(name, pretrained=True, num_classes=0,
                                 global_pool='avg')
    except Exception as e:
        raise RuntimeError(
            f'Failed to load timm model {name!r} with pretrained weights '
            f'(unknown name or weights download failed): {e}') from e


class BackboneFeature(FeatureExtractor):
    """Deep features from a pretrained backbone, global-average-pooled to one
    (C,) vector per image.

    Backbone spec forms:

    - an existing local file (e.g. ``yoloe-26m-seg.pt``): a YOLO checkpoint;
      the stride-32 SPPF backbone output is pooled (no neck/head features);
    - ``torchvision:<name>`` (e.g. ``torchvision:resnet50``);
    - ``timm:<name>`` or a bare model name (e.g. ``timm:resnet50`` or
      ``resnet50``).
    """

    kind = 'dense'

    def __init__(self, spec: str, device: torch.device, batch_size: int = 16,
                 imgsz: int = 0):
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self._hook_handle = None
        self._cache = None
        if osp.isfile(spec):
            self.source = 'yolo'
            self.imgsz = imgsz if imgsz > 0 else 640
            self.model = build_model(spec, device)
            self._hook_handle, self._cache = get_backbone_last_hook(self.model)
            self.letterbox = LetterBox(new_shape=(self.imgsz, self.imgsz),
                                       stride=32)
            self.mean = self.std = None
        elif spec.endswith('.pt') or osp.sep in spec:
            raise FileNotFoundError(
                f'Backbone weights file not found: {spec}. A local .pt path '
                f'must exist; for hub models use torchvision:<name> or '
                f'timm:<name>.')
        else:
            if spec.startswith('torchvision:'):
                self.source, name = 'torchvision', spec[len('torchvision:'):]
                self.model = _load_torchvision_backbone(name)
            elif spec.startswith('timm:'):
                self.source, name = 'timm', spec[len('timm:'):]
                self.model = _load_timm_backbone(name)
            else:
                self.source, name = 'timm', spec
                self.model = _load_timm_backbone(name)
            self.imgsz = imgsz if imgsz > 0 else 224
            self.model.to(device).eval().requires_grad_(False)
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD
        LOGGER.info(f'Backbone feature: source={self.source} spec={spec} '
                    f'imgsz={self.imgsz}')

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """One BGR uint8 image -> (3, H, W) float32 model input."""
        if self.source == 'yolo':
            img = self.letterbox(image=img)
            return img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz),
                         interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    @torch.no_grad()
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) batch -> (B, C) pooled features."""
        if self.source == 'yolo':
            feat = extract_backbone_last_feat(self.model, self._cache, batch)
            return feat.float().mean(dim=(2, 3))  # (B, C, H, W) -> (B, C)
        return self.model(batch).float()

    @torch.no_grad()
    def extract(self, image_paths: List[str]) -> np.ndarray:
        feats, batch_arrays, skipped = [], [], 0
        for path in tqdm(image_paths, desc=f'backbone ({self.spec})'):
            try:
                batch_arrays.append(self._preprocess(read_image_bgr(path)))
            except Exception as e:
                skipped += 1
                LOGGER.warning(
                    f'Skipping image {path}: {type(e).__name__}: {e}')
            if len(batch_arrays) == self.batch_size:
                batch = torch.from_numpy(np.stack(batch_arrays)).to(self.device)
                feats.append(self._forward(batch).cpu().numpy())
                batch_arrays = []
        if batch_arrays:
            batch = torch.from_numpy(np.stack(batch_arrays)).to(self.device)
            feats.append(self._forward(batch).cpu().numpy())
        if not feats:
            raise RuntimeError(f'No usable image among {len(image_paths)} '
                               f'paths ({skipped} failed).')
        if skipped:
            LOGGER.warning(f'{skipped}/{len(image_paths)} images failed and '
                           f'were skipped.')
        return np.concatenate(feats).astype(np.float32)

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PixelPCAFeature(FeatureExtractor):
    """Raw pixel values (resized RGB, scaled to [0, 1]) projected onto a PCA
    basis fitted on a sample of the combined datasets."""

    kind = 'dense'
    raw_is_different = True

    def __init__(self, size: int = 32, n_components: int = 64,
                 fit_samples: int = 2048, seed: int = 0):
        self.size = size
        self.n_components = n_components
        self.fit_samples = fit_samples
        self.seed = seed
        self.pca = None

    def _raw_pixels(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size),
                         interpolation=cv2.INTER_AREA)
        return (img.astype(np.float32) / 255.0).ravel()  # (3*size*size,)

    def fit(self, image_lists: List[List[str]]) -> None:
        from sklearn.decomposition import PCA
        paths = sample_fit_paths(image_lists, self.fit_samples, self.seed)
        raw = compute_per_image(paths, self._raw_pixels, 'pixel_pca fit')
        k = min(self.n_components, raw.shape[0], raw.shape[1])
        if k < self.n_components:
            LOGGER.warning(f'pixel_pca: n_components clamped '
                           f'{self.n_components} -> {k} (samples/dims bound).')
        self.pca = PCA(n_components=k, random_state=self.seed).fit(raw)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError('PixelPCAFeature.extract before fit.')
        raw = compute_per_image(image_paths, self._raw_pixels, 'pixel_pca')
        return self.pca.transform(raw).astype(np.float32)

    def extract_raw(self, image_paths: List[str]) -> np.ndarray:
        """Per-image features before PCA projection, float32 (N, D_raw)."""
        return compute_per_image(image_paths, self._raw_pixels, 'pixel_pca')


class ColorHistogramFeature(FeatureExtractor):
    """Per-channel color histogram (HSV or RGB), L1-normalized."""

    kind = 'histogram'

    def __init__(self, bins: int = 8, space: str = 'hsv'):
        if space not in ('hsv', 'rgb'):
            raise ValueError(f'Unknown color space {space!r} (hsv|rgb).')
        self.bins = bins
        self.space = space

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        if self.space == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ranges = [(0, 180), (0, 256), (0, 256)]
        else:
            img = img[:, :, ::-1]  # BGR -> RGB
            ranges = [(0, 256)] * 3
        hists = [np.histogram(img[:, :, c], bins=self.bins,
                              range=ranges[c])[0] for c in range(3)]
        feat = np.concatenate(hists).astype(np.float32)  # (3*bins,)
        return feat / max(feat.sum(), EPS)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'color_hist')


class ColorMomentsFeature(FeatureExtractor):
    """Per-channel (HSV) mean, std and skewness: 9 values per image."""

    kind = 'dense'

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        feats = []
        for c in range(3):
            v = img[:, :, c].ravel()
            mu, sigma = v.mean(), v.std()
            skew = (((v - mu) ** 3).mean() /
                    max(sigma ** 3, EPS)) if sigma > EPS else 0.0
            feats += [mu, sigma, skew]
        return np.array(feats, dtype=np.float32)  # (9,)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'color_moments')


def _uniform_lbp_lut(n_points: int) -> np.ndarray:
    """256-entry LUT mapping an LBP code to its uniform-LBP bin.

    Uniform codes (at most 2 circular bit transitions) map to their popcount
    (0..P); all other codes share bin P+1. Total bins: P+2.
    """
    lut = np.zeros(256, dtype=np.int64)
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(n_points)]
        transitions = sum(bits[k] != bits[(k + 1) % n_points]
                          for k in range(n_points))
        lut[code] = sum(bits) if transitions <= 2 else n_points + 1
    return lut


class LBPFeature(FeatureExtractor):
    """Uniform Local Binary Pattern histogram of the grayscale image."""

    kind = 'histogram'

    def __init__(self, radius: float = 1.0, n_points: int = 8,
                 max_side: int = 256):
        self.radius = radius
        self.n_points = n_points
        self.max_side = max_side
        self.lut = _uniform_lbp_lut(n_points)

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        gray = resize_max_side(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               self.max_side).astype(np.float32)
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        codes = np.zeros((h, w), dtype=np.uint8)
        for k in range(self.n_points):
            angle = 2.0 * math.pi * k / self.n_points
            dx = self.radius * math.cos(angle)
            dy = -self.radius * math.sin(angle)
            neighbor = cv2.remap(gray, xx + dx, yy + dy, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            codes += (neighbor >= gray).astype(np.uint8) << k
        hist = np.bincount(self.lut[codes.ravel()],
                           minlength=self.n_points + 2).astype(np.float32)
        return hist / max(hist.sum(), EPS)  # (P+2,)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'lbp')


class HOGFeature(FeatureExtractor):
    """Histogram of Oriented Gradients: orientation-bin cell histograms with
    L2-Hys block normalization, flattened to one dense vector."""

    kind = 'dense'

    def __init__(self, size: int = 128, orientations: int = 9,
                 cell_size: int = 8, block_size: int = 2):
        if size % cell_size != 0:
            raise ValueError(f'HOG size ({size}) must be a multiple of '
                             f'cell_size ({cell_size}).')
        self.size = size
        self.orientations = orientations
        self.cell_size = cell_size
        self.block_size = block_size

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.size, self.size),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)  # (H, W)
        ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # unsigned
        n_cells = self.size // self.cell_size
        bin_pos = ang / (180.0 / self.orientations)
        bin0 = np.floor(bin_pos).astype(np.int32) % self.orientations
        frac = (bin_pos - np.floor(bin_pos)).astype(np.float32)
        # Cell histograms with linear interpolation between adjacent bins.
        cell_hist = np.zeros((n_cells, n_cells, self.orientations),
                             dtype=np.float32)
        for cy in range(n_cells):
            for cx in range(n_cells):
                cell = (slice(cy * self.cell_size, (cy + 1) * self.cell_size),
                        slice(cx * self.cell_size, (cx + 1) * self.cell_size))
                m = mag[cell].ravel()
                b = bin0[cell].ravel()
                f = frac[cell].ravel()
                cell_hist[cy, cx] = (
                    np.bincount(b, weights=m * (1.0 - f),
                                minlength=self.orientations) +
                    np.bincount((b + 1) % self.orientations, weights=m * f,
                                minlength=self.orientations))
        # Overlapping block normalization (L2-Hys).
        b = self.block_size
        blocks = []
        for by in range(n_cells - b + 1):
            for bx in range(n_cells - b + 1):
                v = cell_hist[by:by + b, bx:bx + b].ravel()
                v = v / np.sqrt((v * v).sum() + EPS)
                v = np.minimum(v, 0.2)
                v = v / np.sqrt((v * v).sum() + EPS)
                blocks.append(v)
        return np.concatenate(blocks)  # ((n_cells-b+1)^2 * b^2 * O,)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'hog')


class GaborFeature(FeatureExtractor):
    """Mean and std of the Gabor filter-bank responses (a bank over
    frequencies x orientations) of the grayscale image."""

    kind = 'dense'

    def __init__(self, ksize: int = 31,
                 frequencies: tuple = (0.1, 0.25, 0.4),
                 orientations: tuple = (0, 45, 90, 135),
                 max_side: int = 256):
        self.max_side = max_side
        self.kernels = []
        for freq in frequencies:
            lambd = 1.0 / freq  # wavelength in pixels
            sigma = 0.56 * lambd  # roughly one-octave bandwidth
            for deg in orientations:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma,
                                            np.radians(deg), lambd, 0.5, 0,
                                            ktype=cv2.CV_32F)
                self.kernels.append(kernel - kernel.mean())

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        gray = resize_max_side(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               self.max_side).astype(np.float32) / 255.0
        feats = []
        for kernel in self.kernels:
            resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
            feats += [resp.mean(), resp.std()]
        return np.array(feats, dtype=np.float32)  # (2*n_kernels,)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'gabor')


def _glcm_props(glcm: np.ndarray, props: tuple) -> List[float]:
    """Texture statistics of one normalized symmetric GLCM (L, L)."""
    levels = glcm.shape[0]
    idx = np.arange(levels, dtype=np.float64)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    values = []
    for prop in props:
        if prop == 'contrast':
            values.append(float(((ii - jj) ** 2 * glcm).sum()))
        elif prop == 'dissimilarity':
            values.append(float((np.abs(ii - jj) * glcm).sum()))
        elif prop == 'homogeneity':
            values.append(float((glcm / (1.0 + (ii - jj) ** 2)).sum()))
        elif prop == 'energy':
            values.append(float(np.sqrt((glcm ** 2).sum())))
        elif prop == 'correlation':
            p_i, p_j = glcm.sum(axis=1), glcm.sum(axis=0)
            mu_i, mu_j = (idx * p_i).sum(), (idx * p_j).sum()
            sd_i = np.sqrt((((idx - mu_i) ** 2) * p_i).sum())
            sd_j = np.sqrt((((idx - mu_j) ** 2) * p_j).sum())
            values.append(float((((ii - mu_i) * (jj - mu_j) * glcm).sum()) /
                                (sd_i * sd_j + EPS)))
        else:
            raise ValueError(f'Unknown GLCM property {prop!r}.')
    return values


class GLCMFeature(FeatureExtractor):
    """Gray-Level Co-occurrence Matrix texture statistics, concatenated over
    distances x angles."""

    kind = 'dense'

    PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'energy',
             'correlation')

    def __init__(self, levels: int = 16, distances: tuple = (1, 2),
                 angles: tuple = (0, 45, 90, 135), max_side: int = 256):
        self.levels = levels
        self.distances = distances
        self.angles = angles
        self.max_side = max_side

    def _coocurrence(self, q: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Symmetric normalized GLCM (levels, levels) for offset (dy, dx)."""
        h, w = q.shape
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        src = q[y0:y1, x0:x1]
        tgt = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        glcm = np.bincount((src * self.levels + tgt).ravel(),
                           minlength=self.levels * self.levels)
        glcm = glcm.reshape(self.levels, self.levels).astype(np.float64)
        glcm = glcm + glcm.T
        return glcm / max(glcm.sum(), EPS)

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        gray = resize_max_side(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               self.max_side)
        q = (gray.astype(np.int64) * self.levels // 256)  # quantized levels
        feats = []
        for d in self.distances:
            for deg in self.angles:
                dy = round(d * math.sin(math.radians(deg)))
                dx = round(d * math.cos(math.radians(deg)))
                glcm = self._coocurrence(q, dy, dx)
                feats += _glcm_props(glcm, self.PROPS)
        return np.array(feats, dtype=np.float32)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        return compute_per_image(image_paths, self._featurize, 'glcm')


class SiftBovwFeature(FeatureExtractor):
    """SIFT (or dense-grid SIFT) descriptors aggregated into a Bag of Visual
    Words histogram. The codebook is k-means fitted on descriptors sampled
    from the combined datasets."""

    kind = 'histogram'

    def __init__(self, n_words: int = 64, dense_step: int = 0,
                 fit_images: int = 512, max_descriptors: int = 100000,
                 seed: int = 0):
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError('cv2.SIFT_create is unavailable in this '
                               'OpenCV build.')
        self.n_words = n_words
        self.dense_step = dense_step
        self.fit_images = fit_images
        self.max_descriptors = max_descriptors
        self.seed = seed
        self.sift = cv2.SIFT_create()
        self.kmeans = None

    def _descriptors(self, img: np.ndarray) -> np.ndarray:
        """SIFT descriptors of one image, float32 (M, 128); possibly (0, 128)
        when no keypoint is found."""
        gray = resize_max_side(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               max_side=512)
        if self.dense_step > 0:
            step = self.dense_step
            h, w = gray.shape
            keypoints = [cv2.KeyPoint(float(x), float(y), float(step))
                         for y in range(step, h - step, step)
                         for x in range(step, w - step, step)]
            _, desc = self.sift.compute(gray, keypoints)
        else:
            _, desc = self.sift.detectAndCompute(gray, None)
        if desc is None:
            return np.empty((0, 128), dtype=np.float32)
        return desc.astype(np.float32)

    def fit(self, image_lists: List[List[str]]) -> None:
        from sklearn.cluster import MiniBatchKMeans
        rng = np.random.default_rng(self.seed)
        paths = sample_fit_paths(image_lists, self.fit_images, self.seed)
        descs, skipped = [], 0
        for path in tqdm(paths, desc='sift_bovw fit'):
            try:
                desc = self._descriptors(read_image_bgr(path))
            except Exception as e:
                skipped += 1
                LOGGER.warning(
                    f'Skipping image {path}: {type(e).__name__}: {e}')
                continue
            if len(desc) == 0:
                continue
            if len(desc) > 2000:  # cap per image so no image dominates
                desc = desc[rng.choice(len(desc), 2000, replace=False)]
            descs.append(desc)
        if skipped:
            LOGGER.warning(f'{skipped}/{len(paths)} images failed and were '
                           f'skipped.')
        if not descs:
            raise RuntimeError('BoVW codebook fit found no SIFT descriptors.')
        all_desc = np.concatenate(descs)
        if len(all_desc) < self.n_words:
            raise RuntimeError(f'Only {len(all_desc)} SIFT descriptors for a '
                               f'{self.n_words}-word codebook; lower '
                               f'--bovw-words.')
        if len(all_desc) > self.max_descriptors:
            all_desc = all_desc[rng.choice(len(all_desc), self.max_descriptors,
                                           replace=False)]
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_words,
                                      batch_size=8192, n_init=3,
                                      random_state=self.seed).fit(all_desc)

    def _featurize(self, img: np.ndarray) -> np.ndarray:
        desc = self._descriptors(img)
        if len(desc) == 0:  # no keypoints: fall back to a uniform histogram
            return np.full(self.n_words, 1.0 / self.n_words,
                           dtype=np.float32)
        words = self.kmeans.predict(desc)
        hist = np.bincount(words, minlength=self.n_words).astype(np.float32)
        return hist / max(hist.sum(), EPS)  # (n_words,)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError('SiftBovwFeature.extract before fit.')
        return compute_per_image(image_paths, self._featurize, 'sift_bovw')


class CompositeExtractor(FeatureExtractor):
    """Concatenation of several feature blocks, with an optional per-block
    PCA reduction (fitted on a sample of the combined datasets) applied
    before concatenation.

    The composite is a histogram feature only when it wraps exactly one
    histogram block without PCA; otherwise it is dense.
    """

    def __init__(self, extractors: List[FeatureExtractor], pca_dim: int = 0,
                 pca_fit_samples: int = 2048, seed: int = 0):
        if not extractors:
            raise ValueError('CompositeExtractor needs at least one block.')
        self.extractors = extractors
        self.pca_dim = pca_dim
        self.pca_fit_samples = pca_fit_samples
        self.seed = seed
        self.pcas: Optional[list] = None
        self.kind = ('histogram'
                     if len(extractors) == 1 and pca_dim == 0 and
                     extractors[0].kind == 'histogram'
                     else 'dense')
        self.raw_is_different = (pca_dim > 0 or any(
            e.raw_is_different for e in extractors))

    def fit(self, image_lists: List[List[str]]) -> None:
        for extractor in self.extractors:
            extractor.fit(image_lists)
        if self.pca_dim <= 0:
            self.pcas = None
            return
        from sklearn.decomposition import PCA
        fit_paths = sample_fit_paths(image_lists, self.pca_fit_samples,
                                     self.seed)
        self.pcas = []
        for extractor in self.extractors:
            block = extractor.extract(fit_paths)
            k = min(self.pca_dim, block.shape[0], block.shape[1])
            if k < self.pca_dim:
                LOGGER.warning(f'PCA dim clamped {self.pca_dim} -> {k} '
                               f'(samples/dims bound).')
            self.pcas.append(PCA(n_components=k,
                                 random_state=self.seed).fit(block))

    def extract(self, image_paths: List[str]) -> np.ndarray:
        blocks = []
        for i, extractor in enumerate(self.extractors):
            block = extractor.extract(image_paths)  # (N, D_block)
            if self.pcas is not None:
                block = self.pcas[i].transform(block)
            blocks.append(block.astype(np.float32))
        return np.concatenate(blocks, axis=1)  # (N, sum(D_block))

    def extract_raw(self, image_paths: List[str]) -> np.ndarray:
        """Per-image features before the per-block PCA, float32
        (N, sum(D_raw))."""
        blocks = []
        for extractor in self.extractors:
            extract_raw = getattr(extractor, 'extract_raw',
                                  extractor.extract)
            blocks.append(extract_raw(image_paths).astype(np.float32))
        return np.concatenate(blocks, axis=1)

    def close(self) -> None:
        for extractor in self.extractors:
            extractor.close()


def build_extractors(spec: str, args, device: torch.device
                     ) -> CompositeExtractor:
    """Build the composite extractor from a '+'- or ','-joined name spec."""
    builders: Dict[str, object] = {
        'backbone': lambda: BackboneFeature(args.backbone, device,
                                            args.batch, args.imgsz),
        'pixel_pca': lambda: PixelPCAFeature(n_components=args.pixel_pca_dim,
                                             seed=args.seed),
        'color_hist': lambda: ColorHistogramFeature(bins=args.hist_bins),
        'color_moments': lambda: ColorMomentsFeature(),
        'lbp': lambda: LBPFeature(),
        'hog': lambda: HOGFeature(),
        'gabor': lambda: GaborFeature(),
        'glcm': lambda: GLCMFeature(),
        'sift_bovw': lambda: SiftBovwFeature(n_words=args.bovw_words,
                                             dense_step=0, seed=args.seed),
        'dense_sift_bovw': lambda: SiftBovwFeature(n_words=args.bovw_words,
                                                   dense_step=10,
                                                   seed=args.seed),
    }
    names = [name for part in spec.split('+')
             for name in part.split(',') if name.strip()]
    names = [name.strip() for name in names]
    unknown = [name for name in names if name not in builders]
    if unknown:
        raise ValueError(f'Unknown feature block(s): {unknown}. Available: '
                         f'{sorted(builders)}')
    return CompositeExtractor([builders[name]() for name in names],
                              pca_dim=args.pca_dim, seed=args.seed)
