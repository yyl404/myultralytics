"""Estimate per-conv input statistics (PCA / second moment) for ESPReg and NSGP.

Usage:
    $ python tools/pca.py --model <path/to/model.pt> --dataset <dataset.yaml> \
        --save_path <path/to/pca_cache.pkl> [--split val] [--imgsz 640] [--seed 42] \
        [--pca_mode {incremental,gram,exact}] [--use_bbox] [--exclude_head] \
        [--load_hist <previous.pkl>]

Arguments:
    --model: Path to the model checkpoint (.pt file)
    --dataset: Dataset yaml; samples are drawn from its validation dataloader, so the
        images are letterboxed and normalized exactly like training/validation.
    --save_path: Path to save the PCA cache file
    --split: Dataset split (default: val; falls back to val/test/train)
    --imgsz: Inference size of the validation dataloader (default: 640)
    --seed: Random seed for image/position sampling (default: 42)
    --sample_num: Number of images to use, drawn as a seeded random subset (0 = all, default: 500)
    --use_bbox: Sample only feature positions inside ground-truth boxes (ESPReg's
        foreground-local statistics) instead of the whole feature map.
    --layers: (optional) Layers to calculate PCA for, space-separated
    --modules: (optional) Specific module names to calculate PCA for
    --mode: (optional) unfold (default) or fold
    --pca_mode: incremental (upstream IncrementalPCA, centered; default), gram
        (streaming D x D uncentered Gram), or exact (collect vectors, chunked exact
        uncentered second moment).
    --load_hist: Previous cache to accumulate into
    --check: (optional) Verify the convolution kernel unfolding
    --device: Device to use (default: "cuda")

Examples:
    $ python tools/pca.py --model runs/task-1/best.pt \
        --dataset data/VOC_10+10/task_1_cls_10/dataset.yaml \
        --save_path runs/task-1/pca_cache.pkl --pca_mode incremental --sample_num 0 --exclude_head
"""

import argparse
import os.path as OSP
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pca_on_gpu import ExactMomentPCAonGPU as ExactMomentPCA
from pca_on_gpu import IncrementalPCAonGPU as IncrementalPCA
from pca_on_gpu import UncenteredPCAonGPU as UncenteredPCA
from tqdm import tqdm
from utils import RealTimeMemoryMonitor

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy


def module_meta(module):
    if isinstance(module, nn.Conv2d):
        return {
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "groups": module.groups,
            "dilation": module.dilation,
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
        }
    if isinstance(module, nn.Linear):
        return {
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "groups": 1,
            "dilation": (1, 1),
            "in_channels": module.in_features,
            "out_channels": module.out_features,
        }
    if isinstance(module, nn.MultiheadAttention):
        weight = module.in_proj_weight if module.in_proj_weight is not None else module.q_proj_weight
        return {
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "groups": 1,
            "dilation": (1, 1),
            "in_channels": module.embed_dim,
            "out_channels": weight.shape[0],
        }
    # CosineConv2d is a 1x1 dense classifier (no groups/dilation/padding by design).
    return {
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "groups": 1,
        "dilation": (1, 1),
        "in_channels": module.weight.shape[1],
        "out_channels": module.weight.shape[0],
    }


def module_param(module):
    """Return the governed parameter name of a supported module."""
    if isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            return "in_proj_weight"
        if module.q_proj_weight is not None:
            return "q_proj_weight"
        raise ValueError("MultiheadAttention has no in_proj_weight/q_proj_weight to constrain")
    return "weight"


def is_supported_module(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return True
    return isinstance(module, nn.MultiheadAttention) and (
        module.in_proj_weight is not None or module.q_proj_weight is not None
    )


class PCAHooker:
    def __init__(
        self, model, layers=None, modules=None, device="cuda", check=False, unfold=True, pca_mode="incremental"
    ):
        self.model = model
        self.modules = {}
        self.pca_operators = {}
        if not torch.cuda.is_available():
            device = "cpu"
            LOGGER.warning("CUDA is not available, using CPU")
        self.device = device
        self.check = check
        self.unfold = unfold
        self.pca_mode = pca_mode
        self.use_bbox = False

        def _build_operator(module, input_dim):
            if self.pca_mode == "gram":
                return UncenteredPCA(n_components=input_dim, device=self.device)
            if self.pca_mode == "exact":
                return ExactMomentPCA(n_components=input_dim, device=self.device)
            return IncrementalPCA(n_components=input_dim, device=self.device)
        
        def _match(n, m, lid):
            "dfl layer is always frozen, so we don't need to calculate PCA for it"
            return f"model.{lid}." in n and is_supported_module(m) and "dfl" not in n

        self.feature_caches, self._handles = {}, []

        if modules is not None:
            # If modules are provided, only calculate PCA for the specified modules
            for n, m in model.named_modules():
                if n in modules:
                    meta = module_meta(m)
                    k, c_in, g = meta["kernel_size"], meta["in_channels"], meta["groups"]
                    self.modules[n] = m
                    self.pca_operators[n] = []
                    for i in range(g):
                        if self.unfold:
                            n_components = c_in//g*k[0]*k[1]
                        else:
                            n_components = c_in//g
                        self.pca_operators[n].append(_build_operator(m, n_components))
                    self.feature_caches[n] = []
        elif layers is not None:
            for lid in layers:
                # If layers are provided, calculate PCA for all conv modules within layers
                for n, m in model.named_modules():
                    if _match(n, m, lid):
                        meta = module_meta(m)
                        k, c_in, g = meta["kernel_size"], meta["in_channels"], meta["groups"]
                        self.modules[n] = m
                        self.pca_operators[n] = []
                        for i in range(g):
                            if self.unfold:
                                n_components = c_in//g*k[0]*k[1]
                            else:
                                n_components = c_in//g
                            self.pca_operators[n].append(_build_operator(m, n_components))
                        self.feature_caches[n] = []
        else:
            raise ValueError("Either modules or layers must be provided")

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        return torch.arange(bs * h_out * w_out, device=self.device)

    @property
    def names(self):
        return list(self.modules.keys())
        
    def register_hook(self):
        self.remove_handle_()
        for n, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._hook(n, self.unfold)))

    def _consume(self, module_name, feat_sampled, groups):
        """Feed one sampled patch matrix to the configured PCA operator(s)."""
        pca_operators = self.pca_operators[module_name]
        if self.pca_mode in {"gram", "exact"}:
            for group_idx in range(groups):
                pca_operators[group_idx].partial_fit(feat_sampled[group_idx].T)
            torch.cuda.empty_cache()
            return

        # Bound each incremental update while keeping at least four updates worth of
        # data in the accumulation, so small datasets still reach n_components rows.
        cap = max(100, pca_operators[0].n_components // 4)
        if feat_sampled.shape[2] > cap:
            indices = torch.randperm(feat_sampled.shape[2], device=feat_sampled.device)[:cap]
            feat_sampled = feat_sampled[:, :, indices]
        feature_cache = self.feature_caches[module_name]
        feature_cache.append(feat_sampled)
        # Incremental PCA needs its first batch to contain at least n_components rows.
        if sum(x.shape[2] for x in feature_cache) >= pca_operators[0].n_components:
            merged = torch.cat(feature_cache, dim=2)
            feature_cache.clear()
            for group_idx in range(groups):
                try:
                    pca_operators[group_idx].partial_fit(merged[group_idx].T)
                except Exception as error:
                    raise RuntimeError(
                        f"PCA fit failed for module '{module_name}' group {group_idx} "
                        f"with {merged[group_idx].shape[1]} rows and {merged[group_idx].shape[0]} features"
                    ) from error

    def _hook(self, module_name, unfold=True):
        """
        If unfold is True, the input feature will be unfolded into a sequence of vectors with shape [c_in*k[0]*k[1], bs*h_out*w_out],
        otherwise, the input feature will be reshaped into a matrix with shape [c_in, bs*h_out*w_out].
        """
        def fn(module, feat_in, feat_out):
            if is_supported_module(module):
                meta = module_meta(module)
                k, s, p, g, d, c_in, c_out = (
                    meta["kernel_size"],
                    meta["stride"],
                    meta["padding"],
                    meta["groups"],
                    meta["dilation"],
                    meta["in_channels"],
                    meta["out_channels"],
                )

                raw_in = feat_in
                feat_in = feat_in[0]  # Module may accept multiple input features, and we only extract the first
                if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                    if isinstance(module, nn.MultiheadAttention):
                        # in_proj_weight is shared by query/key/value; use all three inputs.
                        parts = [part for part in raw_in[:3] if isinstance(part, torch.Tensor)]
                        tokens = torch.cat([part.reshape(-1, part.shape[-1]) for part in parts], dim=0)
                    else:
                        tokens = feat_in.reshape(-1, feat_in.shape[-1])
                    feat_reshaped = tokens.T.unsqueeze(0)  # (1, in_features, num_tokens)
                    sample_feature_indices = PCAHooker._get_sample_feature_indices(
                        self, 1, 1, tokens.shape[0]
                    )
                    if sample_feature_indices.shape[0] == 0:
                        return
                    sampled = feat_reshaped[:, :, sample_feature_indices]
                    self._consume(module_name, sampled, 1)
                    return
                if self.pca_mode == "gram" and not self.use_bbox:
                    feat_in = feat_in.mean(dim=0, keepdim=True)
                bs, _, h_in_raw, w_in_raw = feat_in.shape
                if p[0] > 0 or p[1] > 0:
                    feat_in_padded = torch.nn.functional.pad(feat_in, (p[1], p[1], p[0], p[0]), mode='constant', value=0)
                else:
                    feat_in_padded = feat_in
                _, _, h_in, w_in = feat_in_padded.shape
                h_out, w_out = feat_out.shape[2], feat_out.shape[3]

                # Group the input features
                c_in_grouped = c_in//g

                if unfold:
                    feat_in_padded_grouped = feat_in_padded.reshape(bs, g, c_in_grouped, h_in, w_in)

                    # Use the sliding window with the same settings as convolution kernels
                    # to unfold input features into a sequence of vectors, considering dilation
                    # For dilated convolution, we need to unfold with larger window size
                    # Effective kernel size becomes: k[0] + (k[0]-1)*(d[0]-1), k[1] + (k[1]-1)*(d[1]-1)
                    if d[0] > 1 or d[1] > 1:
                        # For dilation > 1, we need to unfold with larger window size
                        # Effective kernel size becomes: k[0] + (k[0]-1)*(d[0]-1), k[1] + (k[1]-1)*(d[1]-1)
                        effective_k_h = k[0] + (k[0] - 1) * (d[0] - 1)
                        effective_k_w = k[1] + (k[1] - 1) * (d[1] - 1)

                        # [bs, g, c_in//g, h, w] --> [bs, g, c_in//g, h_out, w, effective_k_h]
                        feat_unfold_h = feat_in_padded_grouped.unfold(3, effective_k_h, s[0])
                        # [bs, g, c_in//g, h_out, w, effective_k_h] --> [bs, g, c_in//g, h_out, w_out, effective_k_h, effective_k_w]
                        feat_unfold = feat_unfold_h.unfold(4, effective_k_w, s[1])

                        # Now subsample to get the actual dilated kernel positions
                        # feat_unfold shape: [bs, g, c_in//g, h_out, w_out, effective_k_h, effective_k_w]
                        # We keep only every d[0]-th and d[1]-th element in kernel dimensions
                        feat_unfold = feat_unfold[:, :, :, :, :, ::d[0], ::d[1]]
                    else:
                        # Standard convolution (dilation = 1)
                        # [bs, g, c_in//g, h, w] --> [bs, g, c_in//g, h_out, w, k[0]]
                        feat_unfold_h = feat_in_padded_grouped.unfold(3, k[0], s[0])
                        # [bs, g, c_in//g, h_out, w, k[0]] --> [bs, g, c_in//g, h_out, w_out, k[0], k[1]]
                        feat_unfold = feat_unfold_h.unfold(4, k[1], s[1])
                    # Get actual kernel dimensions after dilation processing
                    actual_k_h, actual_k_w = feat_unfold.shape[5], feat_unfold.shape[6]

                    # Permute the dims: [bs, g, c_in//g, h_out, w_out, actual_k_h, actual_k_w] -> [g, c_in//g, actual_k_h, actual_k_w, bs, h_out, w_out]
                    feat_unfold = feat_unfold.permute(1, 2, 5, 6, 0, 3, 4).contiguous()
                    # Squeeze: [g, c_in//g, actual_k_h, actual_k_w, bs, h_out, w_out] --> [g, c_in//g*actual_k_h*actual_k_w, bs*h_out*w_out]
                    feat_reshaped = feat_unfold.view(g, c_in_grouped*actual_k_h*actual_k_w, bs*h_out*w_out)

                    pos_h, pos_w = h_out, w_out

                    # The following code is used to check whether the unfolding representation of convolution operation
                    # is equivalent with the original convolution operation
                    if self.check:
                        weight = module.weight.data.reshape(g, c_out//g, -1)  # [g, c_out//g, c_in//g*actual_k_h*actual_k_w]
                        if module.bias is not None:
                            weight = torch.concat((weight, module.bias.data.reshape(g, c_out//g, 1)), dim=2)
                        feat_out_reshaped = weight @ feat_reshaped  # [g, c_out//g, c_in//g*actual_k_h*actual_k_w] @ [c_in//g*actual_k_h*actual_k_w, bs*h_out*w_out] --> [g, c_out//g, bs*h_out*w_out]
                        feat_out_reshaped = feat_out_reshaped.reshape(c_out, -1)  # [c_out, bs*h_out*w_out]

                        # [c_out, bs*h_out*w_out] --> [c_out, bs, h_out, w_out] --> [bs, c_out, h_out, w_out]
                        feat_out_reshaped_reversed = feat_out_reshaped.view(-1, bs, h_out, w_out).permute(1, 0, 2, 3).contiguous()

                        LOGGER.info(f"Module {module_name}'s unfolding error: {F.mse_loss(feat_out, feat_out_reshaped_reversed)}")
                else:
                    # Fold mode: channel vectors at every input position, [g, c_in//g, bs*h_in*w_in]
                    feat_fold = feat_in.reshape(bs, g, c_in_grouped, h_in_raw * w_in_raw)
                    feat_reshaped = feat_fold.permute(1, 2, 0, 3).reshape(g, c_in_grouped, bs * h_in_raw * w_in_raw)
                    pos_h, pos_w = h_in_raw, w_in_raw

                sample_feature_indices = self._get_sample_feature_indices(bs, pos_h, pos_w)
                if sample_feature_indices.shape[0] == 0:
                    # Some batches may have no bounding boxes, so we need to return here
                    return
                if sample_feature_indices.max() >= feat_reshaped.shape[2]:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.max()} >= {feat_reshaped.shape[2]}")
                if sample_feature_indices.min() < 0:
                    raise RuntimeError(f"Sample feature indices out of range: {sample_feature_indices.min()} < 0")
                feat_sampled = feat_reshaped[:, :, sample_feature_indices]
                # unfold true: [g, c_in//g*k*k, len(sample_feature_indices)] | unfold false: [g, c_in//g, len(sample_feature_indices)]
                
                self._consume(module_name, feat_sampled, g)
            else:
                raise RuntimeError(f"Module type {type(module)} is not supported")

        return fn

    def remove_handle_(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_feature_cache(self):
        for n, cache in self.feature_caches.items():
            if len(cache) > 0:
                merged = torch.cat(cache, dim=2)
                for ig in range(len(self.pca_operators[n])):
                    samples = merged[ig].T
                    operator = self.pca_operators[n][ig]
                    if hasattr(operator, "components_"):
                        operator.partial_fit(samples)
                    elif self.pca_mode == "incremental" and samples.shape[0] < operator.n_components:
                        # Fewer samples than dimensions: exact SVD of what we have.
                        _fit_exact_incremental(operator, samples)
                    else:
                        operator.fit(samples)
            cache.clear()
            for operator in self.pca_operators[n]:
                if hasattr(operator, "finalize"):
                    operator.finalize()
    
    def get_pca_results(self, name, ig=None):
        if ig is not None:
            return (self.pca_operators[name][ig].components_, 
                    self.pca_operators[name][ig].explained_variance_)
        else:
            componets_ = []
            variances_ = []
            for ig in range(len(self.pca_operators[name])):
                componets_.append(self.pca_operators[name][ig].components_)
                variances_.append(self.pca_operators[name][ig].explained_variance_)
            return (torch.stack(componets_), torch.stack(variances_))

    def get_pca_operators(self, name):
        return self.pca_operators[name]

    def save_pca_cache(self, save_path):
        """Save PCA operators as plain attribute dicts (CPU-serialized).

        Each entry is {"class": <operator class name>, "state": <state_dict()>}
        instead of the operator object itself, so loading the artifact never
        requires the operator's defining module to be importable (e.g. in DDP
        worker processes launched outside this directory). Tensors are stored
        on CPU, so the artifact loads on any device.
        """
        pca_cache = {}
        for n in self.names:
            operators = self.get_pca_operators(n)
            for ig, operator in enumerate(operators):
                if not hasattr(operator, "components_"):
                    raise RuntimeError(
                        f"PCA operator for module '{n}' group {ig} was never fitted "
                        f"(missing components_). Check that enough boxed samples" 
                        f"were collected."
                    )
            pca_cache[n] = [
                {
                    "class": type(operator).__name__,
                    "param": module_param(self.modules[n]),
                    "state": operator.state_dict(),
                }
                for operator in operators
            ]

        LOGGER.info(f"Saving PCA cache to {save_path}")
        with open(save_path, "wb") as f:
            joblib.dump(pca_cache, f)
    
    def load_pca_cache(self, load_path):
        """Load PCA cache and use it as initial state.

        Cache entries are restored into the operators already built by this
        hooker (matched by class name), and tensors are moved to this hooker's
        device, so caches serialized on a different device (including
        CPU-serialized caches) can be used seamlessly.

        Args:
            load_path: Path to PCA cache file
        """
        with open(load_path, "rb") as f:
            pca_cache = joblib.load(f)
        
        for n in self.names:
            if n in pca_cache:
                if len(pca_cache[n]) != len(self.pca_operators[n]):
                    LOGGER.warning(
                        f"Module {n}: PCA cache has {len(pca_cache[n])} groups, "
                        f"but expected {len(self.pca_operators[n])} groups. "
                        f"This may indicate errors."
                    )
                    continue
                for ig in range(min(len(self.pca_operators[n]), len(pca_cache[n]))):
                    entry = pca_cache[n][ig]
                    operator = self.pca_operators[n][ig]
                    if not isinstance(entry, dict) or "class" not in entry or "state" not in entry:
                        raise TypeError(
                            f"PCA cache entry for module '{n}' group {ig} has an unexpected format "
                            f"(expected a dict with 'class' and 'state' keys, got {type(entry)}). "
                            f"Regenerate the cache with tools/pca.py."
                        )
                    if entry["class"] != type(operator).__name__:
                        raise TypeError(
                            f"PCA cache entry for module '{n}' group {ig} holds a "
                            f"{entry['class']} state, but this run built a "
                            f"{type(operator).__name__} operator."
                        )
                    operator.load_state_dict(entry["state"])
        
        LOGGER.info(f"Loaded PCA cache from {load_path}")


class PCAHookerWithBboxes(PCAHooker):
    def __init__(
        self,
        model,
        layers,
        modules=None,
        bboxes=None,
        device="cuda",
        check=False,
        unfold=True,
        pca_mode="incremental",
    ):
        super().__init__(model, layers, modules, device, check, unfold, pca_mode)
        self.bboxes = bboxes
        self.use_bbox = True
        
    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    def _get_sample_feature_indices(self, bs, h_out, w_out):
        """Use tensor operation to accelerate the extraction of feature indices within bounding boxes
        """
        # Collect all bounding box information
        all_bboxes = []
        batch_ids = []
        for _batch_id, _bboxes in enumerate(self.bboxes):
            for _bbox in _bboxes:
                all_bboxes.append(_bbox)
                batch_ids.append(_batch_id)
        
        # If boxes in this particular batch is empty, return an empty tensor
        if len(all_bboxes) == 0:
            return torch.tensor([], device=self.device)

        # Convert to tensor for vectorized calculation
        bbox_tensor = torch.stack(all_bboxes).to(self.device)  # [N, 4]
        
        # Scale bbox coordinates to feature map size
        feat_coords = bbox_tensor * torch.tensor([w_out, h_out, w_out, h_out], device=self.device)
        feat_x_min = torch.clamp(feat_coords[:, 0].int(), 0, w_out-1)
        feat_y_min = torch.clamp(feat_coords[:, 1].int(), 0, h_out-1)
        feat_x_max = torch.clamp(feat_coords[:, 2].int(), 0, w_out-1)
        feat_y_max = torch.clamp(feat_coords[:, 3].int(), 0, h_out-1)

        # Generate x and y ranges
        sample_feature_indices = []
        for i in range(len(feat_y_min)):
            y_ranges = torch.arange(feat_y_min[i].item(), feat_y_max[i].item()+1, device=self.device)
            x_ranges = torch.arange(feat_x_min[i].item(), feat_x_max[i].item()+1, device=self.device)

            # Generate grid indices
            grid = torch.meshgrid(y_ranges, x_ranges, indexing='ij')
            grid = grid[0].flatten().tolist(), grid[1].flatten().tolist()

            # Calculate feature indices
            for grid_y, grid_x in zip(grid[0], grid[1]):
                _batch_id = batch_ids[i]
                sample_feature_indices.append(_batch_id * h_out * w_out + grid_y * w_out + grid_x)
        sample_feature_indices = torch.tensor(sample_feature_indices, device=self.device)

        return sample_feature_indices


def _fit_exact_incremental(operator, samples):
    """Exact centered PCA over ``samples`` when fewer rows than components exist.

    The upstream IncrementalPCA refuses the first update when ``rows < n_components``.
    Fall back to a direct SVD of the collected rows, keeping ``min(rows, D)``
    components (rank-deficient but exact for the available data) and warn.
    """
    rows, features = samples.shape
    mean = samples.mean(dim=0)
    centered = samples - mean
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    dof = max(1, rows - 1)
    energy = centered.square().sum()
    operator.mean_ = mean
    operator.var_ = centered.square().mean(dim=0)
    operator.components_ = right_vectors
    operator.singular_values_ = singular_values
    operator.explained_variance_ = singular_values.square() / dof
    operator.explained_variance_ratio_ = (
        singular_values.square() / energy if energy > 0 else torch.zeros_like(singular_values)
    )
    operator.noise_variance_ = 0.0
    operator.mean_proj_ = mean @ right_vectors.mH
    operator.n_components_ = right_vectors.shape[0]
    operator.n_features_ = features
    operator.n_samples_seen_ = rows
    operator.weight_sum_ = float(rows)
    operator.weight_square_sum_ = float(rows)
    LOGGER.warning(
        f"Only {rows} samples for {features} features in an incremental PCA; "
        f"falling back to the exact covariance decomposition with {right_vectors.shape[0]} components"
    )


def _seed_everything(seed: int) -> None:
    """Seed every RNG used by artifact sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _SubsetWithCollate(torch.utils.data.Subset):
    """Subset that forwards the parent dataset's collate function to the loader."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.collate_fn = dataset.collate_fn


def _resolve_split(data: dict, split: str) -> str:
    """Return the first requested split that the dataset config provides."""
    for candidate in (split, "val", "test", "train"):
        if candidate and candidate in data:
            if candidate != split:
                LOGGER.warning(f"Split '{split}' not found in the dataset config; using '{candidate}'")
            return candidate
    raise KeyError(f"No usable split in the dataset config; tried '{split}', 'val', 'test' and 'train'")


def _build_val_loader(dataset, imgsz, batch_size, workers, split, device, stride=32, sample_num=500, seed=42):
    """Build the same letterboxed, unaugmented loader that validation uses.

    When ``sample_num`` is positive, the images are a seeded random subset of the
    split (not the first N), matching the historical artifact sampling.
    """
    data = check_det_dataset(dataset)
    split = _resolve_split(data, split)
    cfg = get_cfg(overrides={"imgsz": imgsz, "task": "detect", "rect": False, "cache": False, "workers": workers})
    dataset_obj = build_yolo_dataset(cfg, data[split], batch_size, data, mode="val", stride=stride)
    if sample_num > 0 and len(dataset_obj) > sample_num:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset_obj), generator=generator)[:sample_num].tolist()
        dataset_obj = _SubsetWithCollate(dataset_obj, indices)
    loader = build_dataloader(
        dataset_obj,
        batch_size,
        workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
        device=device,
    )
    LOGGER.info(f"PCA samples: dataset '{dataset}' split '{split}' ({len(dataset_obj)} images, imgsz={imgsz})")
    return loader


def do_pca(model, layers, modules, dataset, device="cuda", check=False, pca_cache_save_path=None,
           sample_num=500, unfold=True, load_hist=None, pca_mode="incremental", batch_size=1,
           split="val", imgsz=640, workers=8, use_bbox=False, seed=42):
    """Accumulate input statistics and write the PCA cache.

    Args:
        model: Model in eval mode.
        layers: Layers whose conv modules are analysed when ``modules`` is None.
        modules: Explicit conv module names to analyse.
        dataset: Dataset yaml; samples come from its validation dataloader.
        device: Device for inference and Gram accumulation.
        check: Verify the unfolding against the real convolution.
        pca_cache_save_path: Output cache path.
        sample_num: Number of images to sample (0 = all); the subset is drawn with ``seed``.
        unfold: Unfold patches with the conv kernel (True) or use channel vectors (False).
        load_hist: Previous cache to accumulate into.
        pca_mode: ``incremental`` (upstream centered IncrementalPCA), ``gram``
            (streaming uncentered Gram), or ``exact`` (collected vectors, exact
            uncentered second moment with chunked Gram accumulation).
        batch_size: Batch size of the sampling pass. With ``use_bbox`` every image
            contributes its own in-box positions (no batch averaging).
        split: Dataset split to sample from.
        imgsz: Inference size of the validation dataloader.
        workers: Dataloader workers.
        use_bbox: Sample only feature positions inside ground-truth boxes.
        seed: Seed for the image subset and position sampling.
    """
    if dataset is None:
        raise ValueError("PCA sampling requires --dataset")
    if batch_size < 1:
        raise ValueError(f"PCA batch_size must be positive, got {batch_size}")
    if use_bbox:
        pca_hooker = PCAHookerWithBboxes(model, layers, modules, None, device, check, unfold, pca_mode)
    else:
        pca_hooker = PCAHooker(model, layers, modules, device, check, unfold, pca_mode)

    if load_hist is not None:
        if OSP.exists(load_hist):
            pca_hooker.load_pca_cache(load_hist)
        else:
            LOGGER.warning(f"Historical PCA cache file not found: {load_hist}. Starting from scratch.")

    memory_monitor = RealTimeMemoryMonitor(update_interval=0.2)
    try:
        stride = int(torch.as_tensor(getattr(model, "stride", 32)).max())
        loader = _build_val_loader(
            dataset, imgsz, batch_size, workers, split, device, stride, sample_num=sample_num, seed=seed
        )
        pbar = tqdm(loader, desc="PCA computing")
        memory_monitor.set_progress_bar(pbar)
        memory_monitor.start_monitoring()
        pca_hooker.register_hook()
        with torch.no_grad():
            for batch in pbar:
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                if use_bbox:
                    # Dataset boxes are (cx, cy, w, h) normalized to the letterboxed
                    # image, concatenated over the batch; split them per image and
                    # convert to xyxy for the feature-grid sampler.
                    counts = torch.bincount(batch["batch_idx"].long(), minlength=images.shape[0])
                    pca_hooker.set_bboxes(list(torch.split(xywh2xyxy(batch["bboxes"]), counts.tolist())))
                _ = model(images)
    finally:
        pca_hooker.remove_handle_()
        memory_monitor.stop_monitoring()
    pca_hooker.clear_feature_cache()

    if pca_cache_save_path:
        pca_hooker.save_pca_cache(pca_cache_save_path)


def main(args):
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    if "cuda" in args.device and not torch.cuda.is_available():
        LOGGER.warning(f"{args.device} is not available, using cpu instead")
        args.device = "cpu"

    _seed_everything(args.seed)

    model = YOLO(args.model).model.to(args.device).eval()

    if args.layers is None and args.modules is None:
        end_layer = len(model.model) - 1 if args.exclude_head else len(model.model)
        layers = list(range(end_layer))
    else:
        layers = args.layers

    if args.mode == "unfold":
        unfold = True
    elif args.mode == "fold":
        unfold = False
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    do_pca(
        model,
        layers,
        args.modules,
        args.dataset,
        device=args.device,
        check=args.check,
        pca_cache_save_path=args.save_path,
        sample_num=args.sample_num,
        unfold=unfold,
        load_hist=args.load_hist,
        pca_mode=args.pca_mode,
        batch_size=args.batch_size,
        split=args.split,
        imgsz=args.imgsz,
        workers=args.workers,
        use_bbox=args.use_bbox,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset yaml; samples come from its validation dataloader (letterboxed, imgsz, no augmentation).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val",
        help="Dataset split (falls back to val/test/train, in that order).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size of the validation dataloader.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for position sampling.")
    parser.add_argument("--sample_num", type=int, default=500,
        help="Number of images to sample, drawn as a seeded random subset (0 = all images).")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size; uncentered statistics average the batch before unfolding, so this is a statistical knob.")
    parser.add_argument("--use_bbox", action="store_true",
        help="Sample only feature positions inside ground-truth boxes (ESPReg's foreground-local statistics).")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
        help="Layers to calculate PCA for, use comma to separate, conv modules within layers are analyzed.")
    parser.add_argument("--modules", nargs="+", type=str, default=None,
        help="Modules to calculate PCA for, use comma to separate, providing more detailed control over the "
             "modules to calculate.")
    parser.add_argument("--exclude_head", action="store_true",
        help="Exclude the final detection head; NSGP governs the feature extractor only.")
    parser.add_argument("--pca_mode", type=str, choices=["incremental", "gram", "exact"], default="incremental",
        help="incremental = upstream centered IncrementalPCA (default); gram = streaming uncentered Gram; "
             "exact = collected vectors with a chunked exact uncentered second moment.")
    parser.add_argument("--mode", type=str, choices=["unfold", "fold"], default="unfold",
        help="Mode to calculate PCA, choices: unfold (default), fold.")
    parser.add_argument("--load_hist", type=str, default=None,
        help="Optional path to historical PCA cache file to load as initial state. "
             "Only modules that exist in both current and historical cache will be loaded.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    main(args)
