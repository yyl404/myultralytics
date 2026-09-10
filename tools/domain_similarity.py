"""Image-domain feature-distribution similarity between datasets.

Datasets are given either as image directories (--images) or as a dataset
yaml sequence (--data; each yaml contributes one split of images).

Two decoupled stages, both pluggable (see tools/feature_extractors.py and
tools/distribution_metrics.py):

1. Feature extraction: a '+'-joined combination of feature blocks
   (pretrained backbone / pixel PCA / color histogram / color moments / LBP /
   HOG / Gabor / GLCM / SIFT-BoVW), optionally PCA-reduced per block before
   concatenation. Dense features are scaled per dimension by the
   combined-dataset std (no centering, so mean-feature cosine keeps its
   direction semantics); histogram features stay L1-normalized.
2. Distribution metrics: every metric maps two feature sample matrices
   (N1, D) x (N2, D) to one distance/similarity value.

Output: one N x N matrix per metric (diagonal = split-half self-comparison),
printed as a readable report and saved to a JSON file plus a long-form
`<stem>_pairs.csv` (columns: metric, dataset_a, dataset_b, value).

Usage (run from the repository root):

    $ python tools/domain_similarity.py \
        --images data/A/images/train data/B/images/train \
        --features backbone+color_hist --backbone torchvision:resnet50 \
        --metrics all --save_path runs/domain_similarity/A_vs_B.json

    $ python tools/domain_similarity.py \
        --data t1.yaml t2.yaml [t3.yaml ...] \
        --features sift_bovw --metrics all --save_path out.json
"""

import argparse
import json
import os
import os.path as osp
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from distribution_metrics import Metric, build_metrics
from feature_drift import get_split_image_dir, list_images
from feature_extractors import build_extractors

from ultralytics.utils import LOGGER

# Lower bound for normalization denominators.
NORM_EPS = 1e-12

# Minimum images per dataset: the diagonal is a split-half self-comparison
# and every metric needs at least 2 samples per side.
MIN_IMAGES = 4


def resolve_device(device: str) -> torch.device:
    """Turn a CLI device string ('0', 'cuda:0', 'cpu') into a torch.device."""
    if device.isdigit():
        device = f'cuda:{device}'
    return torch.device(device)


def resolve_split(data_yaml: str, split: str) -> str:
    """Pick the split to sample images from ('auto': test, else val)."""
    if split != 'auto':
        return split
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    return 'test' if 'test' in data else 'val'


def dataset_label(data_yaml: str) -> str:
    """Label a dataset by its yaml parent directory name (stem as fallback)."""
    parent = osp.basename(osp.dirname(osp.normpath(data_yaml)))
    return parent if parent else osp.splitext(osp.basename(data_yaml))[0]


def print_matrix(title: str, labels: List[str], matrix: np.ndarray) -> None:
    """Print a labeled square matrix to stdout with a title line."""
    width = max(7, max(len(label) for label in labels))
    print(title)
    print(' ' * (width + 1) + ' '.join(f'{label:>{width}}' for label in labels))
    for label, row in zip(labels, matrix):
        print(f'{label:>{width}} ' + ' '.join(f'{value:>{width}.4f}'
                                              for value in row))


def make_labels(dirs: List[str]) -> List[str]:
    """Label each image directory by its basename; prepend the parent
    directory name when basenames collide."""
    labels = [osp.basename(osp.normpath(d)) for d in dirs]
    if len(set(labels)) == len(labels):
        return labels
    labels = [osp.join(osp.basename(osp.dirname(osp.normpath(d))),
                       osp.basename(osp.normpath(d))) for d in dirs]
    if len(set(labels)) != len(labels):
        raise ValueError(f'Cannot disambiguate image directory labels: '
                         f'{dirs}')
    return labels


def resolve_inputs(args) -> Tuple[List[str], List[str]]:
    """Resolve the input mode into (dirs, labels).

    --data: each dataset yaml contributes the image directory of one split
    ('auto': test, else val), labeled by the yaml's parent directory.
    --images: image directories, labeled by basename (disambiguated with the
    parent directory on collision).
    """
    if (args.images is None) == (args.data is None):
        raise ValueError('Exactly one of --images / --data is required.')
    if args.data is not None:
        dirs, labels = [], []
        for data_yaml in args.data:
            split = resolve_split(data_yaml, args.split)
            dirs.append(get_split_image_dir(data_yaml, split))
            labels.append(dataset_label(data_yaml))
            LOGGER.info(f'{data_yaml}: split={split}')
        return dirs, labels
    return args.images, make_labels(args.images)


def load_image_lists(dirs: List[str], max_images: int, seed: int
                     ) -> List[List[str]]:
    """List the images of each directory (optionally capped by a seeded
    random subset)."""
    rng = np.random.default_rng(seed)
    image_lists = []
    for img_dir in dirs:
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f'Image dir not found: {img_dir}')
        paths = list_images(img_dir)
        if 0 < max_images < len(paths):
            paths = sorted(rng.choice(paths, max_images,
                                      replace=False).tolist())
        if len(paths) < MIN_IMAGES:
            raise ValueError(f'{img_dir}: need at least {MIN_IMAGES} images, '
                             f'got {len(paths)}.')
        image_lists.append(paths)
    return image_lists


def normalize_features(feat_list: List[np.ndarray], kind: str
                       ) -> Tuple[List[np.ndarray], str]:
    """Normalize per-dataset feature matrices (N, D) for cross-metric
    comparability.

    Dense features: scale per dimension by the combined-dataset std, without
    centering (all distribution metrics here are translation-invariant, and
    centering would force the per-dataset means to be exactly anti-parallel,
    which breaks mean-feature cosine). Histogram features: L1-normalize rows.
    """
    if kind == 'histogram':
        return [X / np.clip(X.sum(axis=1, keepdims=True), NORM_EPS, None)
                for X in feat_list], 'l1_rows'
    combined = np.concatenate(feat_list)
    std = np.clip(combined.std(axis=0), NORM_EPS, None)
    return [X / std for X in feat_list], 'std_scale'


def compute_matrix(metric: Metric, feats: List[np.ndarray],
                   rng: np.random.Generator,
                   raw_feats: Optional[List[np.ndarray]] = None
                   ) -> np.ndarray:
    """N x N matrix of one metric over the per-dataset feature matrices.

    Off-diagonal cells compare two datasets; diagonal cells compare two
    disjoint halves of the same dataset (split-half self-comparison).
    ``raw_feats`` (pre-PCA, pre-normalization) is used instead of ``feats``
    for mean_cosine, whose mean-direction semantics require the uncentered
    feature space.
    """
    use = (raw_feats
           if metric.name == 'mean_cosine' and raw_feats is not None
           else feats)
    n = len(use)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                perm = rng.permutation(len(use[i]))
                half = len(use[i]) // 2
                value = metric(use[i][perm[:half]], use[i][perm[half:]],
                               rng)
            else:
                value = metric(use[i], use[j], rng)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def write_results(save_path: str, labels: List[str], dirs: List[str],
                  image_lists: List[List[str]], feature_info: dict,
                  metric_results: dict) -> str:
    """Write the JSON report and the long-form pairs CSV next to it.

    Returns the CSV path.
    """
    dirname = osp.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    report = {
        'datasets': [{'label': label, 'dir': img_dir,
                      'num_images': len(paths)}
                     for label, img_dir, paths in
                     zip(labels, dirs, image_lists)],
        'feature': feature_info,
        'metrics': {name: {'similarity': res['similarity'],
                           'description': res['description'],
                           'matrix': [[round(float(v), 6) for v in row]
                                      for row in res['matrix']]}
                    for name, res in metric_results.items()},
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_path = osp.splitext(save_path)[0] + '_pairs.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('metric,dataset_a,dataset_b,value\n')
        for name, res in metric_results.items():
            matrix = res['matrix']
            for i in range(len(labels)):
                f.writelines(f'{name},{labels[i]},{labels[j]},'
                            f'{matrix[i][j]:.6f}\n' for j in range(i, len(labels)))
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image-domain feature-distribution similarity between '
        'image directories (pluggable feature extractors x distribution '
        'metrics).')
    parser.add_argument('--images', nargs='+', default=None,
                        help='Image directories to compare (order = matrix '
                        'order; N=1 yields the split-half self-baseline). '
                        'Mutually exclusive with --data.')
    parser.add_argument('--data', nargs='+', default=None,
                        help='Dataset yaml sequence (each yaml contributes '
                        'the image directory of one split). Mutually '
                        'exclusive with --images.')
    parser.add_argument('--split', default='auto',
                        help="Split to sample with --data: 'auto' (test, "
                        "else val) or an explicit split name.")
    parser.add_argument('--features', default='backbone',
                        help="'+'-joined feature blocks: backbone, pixel_pca, "
                        "color_hist, color_moments, lbp, hog, gabor, glcm, "
                        "sift_bovw, dense_sift_bovw.")
    parser.add_argument('--backbone', default='torchvision:resnet50',
                        help="Backbone spec for the 'backbone' block: local "
                        "YOLO .pt path | torchvision:<name> | timm:<name> | "
                        "bare timm model name.")
    parser.add_argument('--metrics', default='all',
                        help="'+'-joined metrics or 'all' (every metric "
                        "compatible with the feature kind): mean_cosine, "
                        "mmd_rbf_linear, mmd_rbf, frechet, "
                        "sliced_wasserstein, energy_distance, js_divergence, "
                        "hist_intersection, chi2_distance, domain_auc.")
    parser.add_argument('--pca-dim', '--pca_dim', type=int, default=0,
                        help='Per-block PCA dim before concatenation '
                        '(0 = off).')
    parser.add_argument('--pixel-pca-dim', '--pixel_pca_dim', type=int,
                        default=64, help='Output dim of the pixel_pca block.')
    parser.add_argument('--hist-bins', '--hist_bins', type=int, default=8,
                        help='Bins per channel of the color_hist block.')
    parser.add_argument('--bovw-words', '--bovw_words', type=int, default=64,
                        help='Codebook size of the sift_bovw blocks.')
    parser.add_argument('--max-images', '--max_images', type=int, default=0,
                        help='Cap images per directory (0 = all).')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for backbone inference.')
    parser.add_argument('--imgsz', type=int, default=0,
                        help='Input size for the backbone block '
                        '(0 = auto: 640 for YOLO checkpoints, 224 for '
                        'timm/torchvision).')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for backbone inference.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (subsampling, projections, folds).')
    parser.add_argument('--save-path', '--save_path', required=True,
                        help='JSON file the report is written to; a '
                        '<stem>_pairs.csv is written alongside.')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    dirs, labels = resolve_inputs(args)
    image_lists = load_image_lists(dirs, args.max_images, args.seed)
    for label, paths in zip(labels, image_lists):
        LOGGER.info(f'{label}: {len(paths)} images')

    composite = build_extractors(args.features, args, device)
    composite.fit(image_lists)
    feats = []
    for label, paths in zip(labels, image_lists):
        feat = composite.extract(paths)  # (N, D)
        feats.append(feat)
        LOGGER.info(f'{label}: features {feat.shape}')
    metrics = build_metrics(args.metrics, composite.kind)
    # mean_cosine keeps the legacy mean-feature semantics: it is computed on
    # the raw concatenated features (pre-PCA, pre-normalization).
    raw_feats = None
    if (composite.raw_is_different
            and any(m.name == 'mean_cosine' for m in metrics)):
        raw_feats = [composite.extract_raw(paths) for paths in image_lists]
    composite.close()
    raw_input = raw_feats if raw_feats is not None else feats
    feats, normalization = normalize_features(feats, composite.kind)
    feature_info = {'spec': args.features, 'kind': composite.kind,
                    'dim': int(feats[0].shape[1]),
                    'pca_dim': args.pca_dim, 'normalization': normalization,
                    'mean_cosine_space': 'raw concatenated features '
                    '(pre-PCA, pre-normalization)'}
    LOGGER.info(f'Feature: {feature_info}')

    metric_results = {}
    for metric in metrics:
        matrix = compute_matrix(metric, feats, rng, raw_feats=raw_input)
        metric_results[metric.name] = {
            'matrix': matrix,
            'similarity': metric.similarity,
            'description': metric.description,
        }
        direction = ('higher = more similar' if metric.similarity
                     else 'higher = more different')
        print_matrix(f'{metric.name} ({direction}):', labels, matrix)

    csv_path = write_results(args.save_path, labels, dirs,
                             image_lists, feature_info, metric_results)
    LOGGER.info(f'Report saved to {args.save_path} and {csv_path}')


if __name__ == '__main__':
    main()
