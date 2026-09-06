"""Compute the N x N image-domain deep-feature similarity matrix of a dataset
sequence with the backbone of a pretrained YOLO model.

Each dataset contributes one split of images. Every image is letterboxed and
forwarded once; the last backbone scale (SPPF output, stride 32, the feature
fed to the neck; no neck/head features are used) is global-average-pooled
into a single (C,) deep feature vector. A dataset's image-domain feature is
the mean of its per-image vectors, and the similarity between two datasets is
the cosine similarity of their mean features.

Usage (run from the repository root):

    $ python tools/dataset_similarity.py \
        --data a.yaml b.yaml [c.yaml ...] \
        --weights yoloe-26m-seg.pt \
        --save_path similarity_matrix.csv
"""

import argparse
import csv
import os
import os.path as osp
from typing import List

import numpy as np
import torch
import yaml

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER

from feature_drift import (
    build_model,
    extract_backbone_last_feat,
    get_backbone_last_hook,
    get_split_image_dir,
    iter_batches,
    list_images,
    preprocess_batch,
)

# Lower bound for feature norms in the cosine similarity.
NORM_EPS = 1e-12


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


def resolve_device(device: str) -> torch.device:
    """Turn a CLI device string ('0', 'cuda:0', 'cpu') into a torch.device."""
    if device.isdigit():
        device = f'cuda:{device}'
    return torch.device(device)


@torch.no_grad()
def extract_dataset_feature(model: torch.nn.Module, cache: dict,
                            img_paths: List[str], letterbox: LetterBox,
                            batch_size: int,
                            device: torch.device) -> np.ndarray:
    """Return the dataset image-domain feature of shape (C,).

    Per image, the stride-32 backbone map (C, H/32, W/32) is global-average-
    pooled over space into one (C,) vector; the dataset feature is the mean
    of the per-image vectors.
    """
    feat_sum = None  # (C,) float64 running sum
    count = 0
    for batch_paths in iter_batches(img_paths, batch_size):
        inputs = preprocess_batch(batch_paths, letterbox, device)
        feat = extract_backbone_last_feat(model, cache, inputs)  # (B, C, H, W)
        pooled = feat.float().mean(dim=(2, 3)).to('cpu', torch.float64)  # (B, C)
        batch_sum = pooled.sum(dim=0)
        feat_sum = batch_sum if feat_sum is None else feat_sum + batch_sum
        count += pooled.shape[0]
    if count == 0:
        raise RuntimeError('Empty image set: no image was processed.')
    return (feat_sum / count).numpy()


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """N x N cosine similarity between the rows of ``features`` (N, C)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)  # (N, 1)
    normed = features / np.clip(norms, NORM_EPS, None)
    return normed @ normed.T


def print_matrix(labels: List[str], matrix: np.ndarray) -> None:
    """Print the labeled similarity matrix to stdout."""
    width = max(7, max(len(label) for label in labels))
    print('Image-domain deep-feature cosine similarity matrix:')
    print(' ' * (width + 1) + ' '.join(f'{label:>{width}}' for label in labels))
    for label, row in zip(labels, matrix):
        print(f'{label:>{width}} ' + ' '.join(f'{value:>{width}.4f}'
                                              for value in row))


def write_matrix_csv(save_path: str, labels: List[str],
                     matrix: np.ndarray) -> None:
    """Write the labeled similarity matrix to CSV."""
    dirname = osp.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f'{value:.6f}' for value in row])


def parse_args():
    parser = argparse.ArgumentParser(
        description='N x N image-domain deep-feature similarity matrix of a '
        'dataset sequence with a pretrained YOLO backbone.')
    parser.add_argument('--data', nargs='+', required=True,
                        help='Dataset yaml sequence (N yamls, order = matrix order).')
    parser.add_argument('--weights', default='yoloe-26m-seg.pt',
                        help='Pretrained weights whose backbone is used.')
    parser.add_argument('--split', default='auto',
                        help="Split to sample: 'auto' (test, else val) or an "
                        "explicit split name.")
    parser.add_argument('--batch', type=int, default=16, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Letterbox size, should match model training.')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference.')
    parser.add_argument('--save_path', required=True,
                        help='CSV file the similarity matrix is written to.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.weights, device)
    handle, cache = get_backbone_last_hook(model)
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), stride=32)

    labels, features = [], []
    for data_yaml in args.data:
        split = resolve_split(data_yaml, args.split)
        img_paths = list_images(get_split_image_dir(data_yaml, split))
        LOGGER.info(f'{data_yaml}: {len(img_paths)} images (split={split})')
        features.append(extract_dataset_feature(
            model, cache, img_paths, letterbox, args.batch, device))
        labels.append(dataset_label(data_yaml))
    handle.remove()

    matrix = cosine_similarity_matrix(np.stack(features))  # (N, N)
    print_matrix(labels, matrix)
    write_matrix_csv(args.save_path, labels, matrix)
    LOGGER.info(f'Similarity matrix saved to {args.save_path}')


if __name__ == '__main__':
    main()
