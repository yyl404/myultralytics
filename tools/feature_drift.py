"""Evaluate backbone feature drift between two YOLO model checkpoints on the
task-1 sample set of an incremental dataset.

Both models receive the *same* letterboxed batches and the feature map is
taken from the last backbone scale, i.e. the SPPF output (stride 32, the
backbone output fed to the neck; no neck/head features are used).

Per spatial position with feature vectors ``a`` (model-1) and ``b``
(model-2), the following scale-free quantities are computed
(``rho = ||b|| / ||a||``):

- ``abs_residual_l2``:  ||a - b||                    (raw, scale-dependent)
- ``rel_total_drift``:  ||a - b|| / ||a||           (total drift, unit-free;
    rel^2 = (1-rho)^2 + 2*rho*(1-cos), i.e. magnitude^2 + direction)
- ``direction_cos``:    1 - cos(a, b)               (direction component)
- ``magnitude_rel``:    | ||b|| - ||a|| | / ||a||   (magnitude component)
- ``magnitude_rel_signed``: (||b|| - ||a||) / ||a|| (signed: grow/shrink)
- ``model1_feat_l2`` / ``model2_feat_l2``: ||a|| and ||b|| (context)
- ``zscore_residual_l2``: residual after channel-wise z-score with
    model-1 statistics (unit: model-1 std, magnitude-preserving)

Positions with ||a|| <= 1e-6 are excluded from the ratio-based metrics
(their ratios are undefined); the masked fraction is reported.

All per-position quantities are averaged over space per image, then
aggregated over the image set as mean and population std.

The script runs two passes over the sample set: pass 1 collects model-1
channel statistics for the z-score metric, pass 2 computes all metrics.

Usage (run from the repository root):

    $ python tools/feature_drift.py \
        --data data/VOC_15+5/task_1_cls_15/dataset.yaml \
        --model1 <task1_ckpt.pt> --model2 <task2_ckpt.pt> \
        --save_path <out.json>
"""

import argparse
import glob
import json
import os.path as osp
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules import SPPF
from ultralytics.utils import LOGGER

# Minimum ||f1|| for a spatial position to enter ratio-based metrics
# (rel_total_drift, direction_cos, magnitude_rel[*]).
RATIO_EPS = 1e-6

# Metric keys accumulated over per-image spatial means.
METRIC_KEYS = ('d_abs', 'rel', 'dir', 'mag', 'smag', 'm1', 'm2', 'z')

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def get_split_image_dir(data_yaml: str, split: str) -> str:
    """Resolve the image directory of ``split`` from a dataset yaml file.

    Relative entries in the yaml are resolved against the yaml's directory.

    Raises:
        KeyError: If the yaml has no entry for ``split``.
        FileNotFoundError: If the resolved directory does not exist.
    """
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    if split not in data:
        raise KeyError(f'{data_yaml} has no {split!r} split entry')
    split_path = data[split]
    if not osp.isabs(split_path):
        split_path = osp.join(osp.dirname(data_yaml), split_path)
    if not osp.isdir(split_path):
        raise FileNotFoundError(f'Split image dir not found: {split_path}')
    return split_path


def list_images(img_dir: str) -> List[str]:
    """List image files of a directory, sorted for determinism."""
    images = []
    for ext in IMG_EXTENSIONS:
        images += glob.glob(osp.join(img_dir, f'*{ext}'))
    if not images:
        raise FileNotFoundError(f'No images found in {img_dir}')
    return sorted(images)


def build_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Load a YOLO checkpoint and return the frozen detection model."""
    model = YOLO(checkpoint).model
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def get_backbone_last_hook(model: torch.nn.Module):
    """Register a forward hook on the model's SPPF module.

    The SPPF output is the last backbone scale (stride 32), i.e. the
    backbone feature fed to the neck.

    Returns:
        Tuple of (hook handle, cache dict). After each forward, the cached
        feature map is available as ``cache['feat']`` of shape (B, C, H, W).

    Raises:
        RuntimeError: If the model does not contain exactly one SPPF module.
    """
    sppf_modules = [m for m in model.modules() if isinstance(m, SPPF)]
    if len(sppf_modules) != 1:
        raise RuntimeError(
            f'Expected exactly 1 SPPF module (last backbone scale), found '
            f'{len(sppf_modules)}; unsupported model architecture.')
    cache = {}

    def hook(module, args, output):
        cache['feat'] = output

    handle = sppf_modules[0].register_forward_hook(hook)
    return handle, cache


@torch.no_grad()
def extract_backbone_last_feat(model: torch.nn.Module, cache: Dict,
                               inputs: torch.Tensor) -> torch.Tensor:
    """Run a full forward and return the cached SPPF feature map.

    Args:
        inputs: (B, 3, H, W) batch, already letterboxed and scaled to [0, 1].

    Returns:
        Tensor of shape (B, C, H/32, W/32): the stride-32 backbone output.
    """
    cache.pop('feat', None)
    model(inputs)
    if 'feat' not in cache:
        raise RuntimeError('SPPF hook captured no feature map; the forward '
                           'pass did not reach the backbone output.')
    return cache['feat']


def _new_channel_stats() -> Dict:
    """Streaming per-channel sums of model-1 features (for z-score)."""
    return dict(sum=None, sumsq=None, count=0)  # sum/sumsq: (C,) float64


@torch.no_grad()
def update_channel_stats(chan_stats: Dict, feat: torch.Tensor) -> None:
    """Accumulate per-channel sums of one batch of feature maps.

    Args:
        chan_stats: Streaming channel sums, see :func:`_new_channel_stats`.
        feat: (B, C, H, W) feature map.
    """
    f = feat.float()
    chan_sum = f.sum(dim=(0, 2, 3)).to('cpu', torch.float64)  # (C,)
    chan_sumsq = f.pow(2).sum(dim=(0, 2, 3)).to('cpu', torch.float64)
    if chan_stats['sum'] is None:
        chan_stats['sum'] = chan_sum
        chan_stats['sumsq'] = chan_sumsq
    else:
        chan_stats['sum'] += chan_sum
        chan_stats['sumsq'] += chan_sumsq
    chan_stats['count'] += f.shape[0] * f.shape[2] * f.shape[3]


def finalize_channel_stats(chan_stats: Dict,
                           device: torch.device) -> Tuple[torch.Tensor,
                                                          torch.Tensor]:
    """Turn streaming channel sums into (mean, std) tensors of shape (C,).

    Std is clamped below by ``RATIO_EPS`` so standardized features stay
    finite on constant channels.
    """
    count = chan_stats['count']
    if count == 0:
        raise RuntimeError('Empty task-1 sample set: no image was evaluated.')
    mean = chan_stats['sum'] / count
    var = (chan_stats['sumsq'] / count - mean.pow(2)).clamp_min(0.0)
    std = var.sqrt().clamp_min(RATIO_EPS)
    return (mean.to(device, torch.float32), std.to(device, torch.float32))


def _new_stats() -> Dict[str, float]:
    """Streaming sums over per-image spatial-mean scalars."""
    stats = dict(n=0, ratio_valid=0.0, ratio_total=0.0)
    for key in METRIC_KEYS:
        stats[f'sum_{key}'] = 0.0
        stats[f'sumsq_{key}'] = 0.0
    return stats


@torch.no_grad()
def update_stats(stats: Dict[str, float], feat1: torch.Tensor,
                 feat2: torch.Tensor, chan_mean: torch.Tensor,
                 chan_std: torch.Tensor) -> None:
    """Accumulate per-image drift statistics for one batch.

    Args:
        stats: Streaming sums, see :func:`_new_stats`.
        feat1: (B, C, H, W) model-1 backbone feature map.
        feat2: (B, C, H, W) model-2 backbone feature map, same batch.
        chan_mean: (C,) model-1 channel means on the sample set.
        chan_std: (C,) model-1 channel stds on the sample set.
    """
    if feat1.shape != feat2.shape:
        raise RuntimeError(
            f'Feature shape mismatch: model-1 {tuple(feat1.shape)} vs '
            f'model-2 {tuple(feat2.shape)}')
    f1 = feat1.float()
    f2 = feat2.float()
    norm1 = f1.norm(p=2, dim=1)  # (B, H, W)
    norm2 = f2.norm(p=2, dim=1)
    res = (f1 - f2).norm(p=2, dim=1)  # (B, H, W)

    # Ratio-based metrics are only defined where ||f1|| > RATIO_EPS.
    valid = (norm1 > RATIO_EPS).float()  # (B, H, W)
    valid_count = valid.flatten(1).sum(dim=1).clamp_min(1.0)  # (B,)
    norm1_safe = norm1.clamp_min(RATIO_EPS)
    rel = res / norm1_safe
    mag = (norm2 - norm1).abs() / norm1_safe
    smag = (norm2 - norm1) / norm1_safe
    cos = F.cosine_similarity(f1, f2, dim=1, eps=RATIO_EPS)  # (B, H, W)
    direction = (1.0 - cos).clamp_min(0.0)

    # z-score residual with model-1 channel statistics.
    mu = chan_mean.view(1, -1, 1, 1)
    sd = chan_std.view(1, -1, 1, 1)
    z_res = ((f1 - mu) / sd - (f2 - mu) / sd).norm(p=2, dim=1)  # (B, H, W)

    def img_mean(x: torch.Tensor) -> torch.Tensor:
        # (B, H, W) -> (B,) spatial mean over all positions
        return x.flatten(1).mean(dim=1)

    def img_mean_masked(x: torch.Tensor) -> torch.Tensor:
        # (B, H, W) -> (B,) spatial mean over valid positions only
        return (x * valid).flatten(1).sum(dim=1) / valid_count

    per_image = dict(
        d_abs=img_mean(res),
        rel=img_mean_masked(rel),
        dir=img_mean_masked(direction),
        mag=img_mean_masked(mag),
        smag=img_mean_masked(smag),
        m1=img_mean(norm1),
        m2=img_mean(norm2),
        z=img_mean(z_res))

    stats['n'] += f1.shape[0]
    stats['ratio_valid'] += valid.sum().item()
    stats['ratio_total'] += valid.numel()
    for key, values in per_image.items():
        values = values.to('cpu', torch.float64)  # (B,)
        stats[f'sum_{key}'] += values.sum().item()
        stats[f'sumsq_{key}'] += values.pow(2).sum().item()


def finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """Turn streaming sums into mean / population-std metrics."""
    n = stats['n']
    if n == 0:
        raise RuntimeError('Empty task-1 sample set: no image was evaluated.')

    def mean_std(key: str) -> Dict[str, float]:
        mean = stats[f'sum_{key}'] / n
        var = max(stats[f'sumsq_{key}'] / n - mean**2, 0.0)
        return dict(mean=mean, std=var**0.5)

    out = dict(
        num_images=n,
        abs_residual_l2=mean_std('d_abs'),
        rel_total_drift=mean_std('rel'),
        direction_cos=mean_std('dir'),
        magnitude_rel=mean_std('mag'),
        magnitude_rel_signed=mean_std('smag'),
        model1_feat_l2=mean_std('m1'),
        model2_feat_l2=mean_std('m2'),
        zscore_residual_l2=mean_std('z'),
        ratio_masked_fraction=1.0 -
        stats['ratio_valid'] / max(stats['ratio_total'], 1.0))
    out['global_rel_drift'] = (out['abs_residual_l2']['mean'] /
                               out['model1_feat_l2']['mean'])
    return out


def preprocess_batch(img_paths: List[str], letterbox: LetterBox,
                     device: torch.device) -> torch.Tensor:
    """Letterbox a batch of images and pad them to a common size.

    Returns:
        Tensor of shape (B, 3, H, W) on ``device``, scaled to [0, 1].
    """
    images = []
    for path in img_paths:
        img = cv2.imread(path)  # (h, w, 3) BGR uint8
        if img is None:
            raise RuntimeError(f'Failed to read image: {path}')
        img = letterbox(image=img)  # letterboxed, BGR
        images.append(img)
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    batch = np.zeros((len(images), 3, max_h, max_w), dtype=np.float32)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        # HWC BGR -> CHW RGB, scaled to [0, 1], bottom-right padding.
        batch[i, :, :h, :w] = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return torch.from_numpy(batch).to(device)


def iter_batches(items: List[str], batch_size: int):
    """Yield consecutive chunks of ``items`` of size ``batch_size``."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def evaluate_pair(data_yaml: str, model1_ckpt: str, model2_ckpt: str,
                  split: str, batch_size: int, imgsz: int,
                  device: torch.device) -> Dict:
    """Evaluate feature drift between two checkpoints on the task-1 set."""
    img_dir = get_split_image_dir(data_yaml, split)
    img_paths = list_images(img_dir)
    LOGGER.info(f'{len(img_paths)} images from {img_dir}')

    model1 = build_model(model1_ckpt, device)
    model2 = build_model(model2_ckpt, device)
    handle1, cache1 = get_backbone_last_hook(model1)
    handle2, cache2 = get_backbone_last_hook(model2)
    letterbox = LetterBox(new_shape=(imgsz, imgsz), stride=32)

    # Pass 1: model-1 channel statistics for the z-score metric.
    chan_stats = _new_channel_stats()
    for batch_paths in iter_batches(img_paths, batch_size):
        inputs = preprocess_batch(batch_paths, letterbox, device)
        feat1 = extract_backbone_last_feat(model1, cache1, inputs)
        update_channel_stats(chan_stats, feat1)
    chan_mean, chan_std = finalize_channel_stats(chan_stats, device)

    # Pass 2: all drift metrics.
    stats = _new_stats()
    for batch_paths in iter_batches(img_paths, batch_size):
        inputs = preprocess_batch(batch_paths, letterbox, device)
        feat1 = extract_backbone_last_feat(model1, cache1, inputs)
        feat2 = extract_backbone_last_feat(model2, cache2, inputs)
        update_stats(stats, feat1, feat2, chan_mean, chan_std)
    handle1.remove()
    handle2.remove()

    metrics = finalize_stats(stats)
    metrics.update(
        data=data_yaml,
        split=split,
        model1_checkpoint=model1_ckpt,
        model2_checkpoint=model2_ckpt,
        imgsz=imgsz,
        feature_level='SPPF output (stride 32, last backbone scale, no neck)')
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate backbone feature drift between two YOLO '
        'checkpoints on the task-1 sample set.')
    parser.add_argument('--data', required=True,
                        help='Task-1 dataset yaml file.')
    parser.add_argument('--model1', required=True,
                        help='Task-1 (reference) model checkpoint.')
    parser.add_argument('--model2', required=True,
                        help='Task-2 (drifted) model checkpoint.')
    parser.add_argument('--split', default='test',
                        help='Dataset split used as the sample set.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Letterbox size, must match model training.')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference.')
    parser.add_argument('--save_path', required=True,
                        help='JSON file the metrics are written to.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    metrics = evaluate_pair(args.data, args.model1, args.model2, args.split,
                            args.batch, args.imgsz, device)
    print(json.dumps(metrics, indent=2))
    with open(args.save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f'Result saved to {args.save_path}')


if __name__ == '__main__':
    main()
