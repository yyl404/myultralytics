#!/usr/bin/env python3
"""Build the detection-head input patch PCA and multi-centre class prototypes.

The artifact is shared by the two prototype-based losses:

- `prad` (prototype-reliability-aware YOLO LwF) matches teacher foreground responses to
  the old-class prototypes and needs the fixed PCA codec plus the prototype bank;
- `prd` (patch replay distillation) samples codes from the local distributions and pushes
  the reconstructed patches through the teacher and student heads.

Base stage (no `--load_hist`): one shared PCA per detection level is fitted on every
class of the task, then each class is clustered into local components. Later stages reuse
that fixed PCA and only append the new classes, so the code coordinates of the earlier
prototypes never change.

Usage:
    python tools/generate_patch_prototypes.py \
        --model runs/<run>/task-1/best.pt \
        --data data/VOC_15+5/task_1_cls_15/dataset.yaml \
        --output runs/<run>/task-1/patch_prototypes.pt

    # later stage: reuse the fixed PCA and append the task classes
    python tools/generate_patch_prototypes.py ... --load_hist runs/<run>/task-1/patch_prototypes.pt

    # PCA feasibility report (method_plan section 8.5): reconstruction deviations per
    # level / dimension / class, plus the prototype similarity distribution
    python tools/generate_patch_prototypes.py ... --report_dims 32 64 128 256 --report_similarity
"""

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.patch_proto import (
    EPS,
    HeadPatchSubspace,
    MultiCenterBank,
    RecordingDetectionLoss,
    extract_patches,
    head_input_channels,
    head_patch_radius,
    level_offsets,
    load_patch_artifact,
    save_patch_artifact,
    valid_patch_positions,
)
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.tal import make_anchors

PCA_CHUNK = 1024


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Generate detection-head patch prototypes")
    parser.add_argument("--model", required=True, help="Task checkpoint whose head defines the patch space")
    parser.add_argument("--data", required=True, help="Task dataset YAML (its train split is sampled)")
    parser.add_argument("--output", default=None, help="Output patch prototype artifact (.pt)")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Split to sample")
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only measure reconstruction/similarity against --load_hist; no artifact is written",
    )
    parser.add_argument("--load_hist", default=None, help="Previous task artifact; reuses its fixed PCA")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0, help="0 uses the whole train split")
    parser.add_argument("--pca_dims", default="128", help="Components per level, or comma separated per level")
    parser.add_argument("--pca_energy_ratio", type=float, default=0.0, help=">0 selects dims by explained variance")
    parser.add_argument("--no_compression", action="store_true", help="Store raw patch vectors (no PCA)")
    parser.add_argument("--patch_radius", type=int, default=-1, help="-1 derives it from the head; 0 gives 1x1 patches")
    parser.add_argument("--var_shrinkage", type=float, default=0.1, help="Shrinkage towards the class pooled variance")
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="DP-Means new-centre penalty as a fraction of the class total squared error",
    )
    parser.add_argument("--min_cluster", type=int, default=5)
    parser.add_argument("--min_centers", type=int, default=1)
    parser.add_argument("--max_centers", type=int, default=8)
    parser.add_argument("--merge_ratio", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max_samples_per_class", type=int, default=256)
    parser.add_argument("--center_ratio", type=float, default=0.5, help="GT centre region used for candidate anchors")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_dims", type=int, nargs="+", default=None)
    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--report_samples", type=int, default=1024, help="Patches per level kept for the report")
    parser.add_argument("--report_anchors", type=int, default=8, help="Confidence-sampled anchors per image")
    parser.add_argument("--report_similarity", action="store_true")
    return parser.parse_args()


def resolve_device(spec):
    """Resolve a device string the way the other tools do."""
    if str(spec).isdigit():
        return torch.device(f"cuda:{spec}")
    if "cuda" in str(spec) and not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(spec)


def build_loader(data_yaml, imgsz, batch_size, workers, split="train"):
    """Build the evaluation-mode (letterboxed, un-augmented) loader of one dataset split."""
    args = get_cfg(overrides={"imgsz": imgsz, "rect": False, "task": "detect", "batch": batch_size})
    data = check_det_dataset(data_yaml)
    if split not in data:
        raise KeyError(f"Dataset '{data_yaml}' has no {split!r} split")
    dataset = build_yolo_dataset(args, data[split], batch=batch_size, data=data, mode="val", rect=False, stride=32)
    loader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False, rank=-1)
    return loader, data


def class_mapping(dataset_names, model_names):
    """Map dataset class ids to model class ids by name; None when a class is unknown."""
    model_by_name = {name: index for index, name in model_names.items()}
    mapping = {}
    for index, name in dataset_names.items():
        if name not in model_by_name:
            return None
        mapping[int(index)] = model_by_name[name]
    return mapping


def predicted_positions(preds, shapes, anchors, levels, offsets, per_image, probe_classes=None):
    """Sample the model's most confident anchors per image and level.

    Used when the sampled dataset holds classes the model does not know (later-stage
    samples with the base-stage head), where GT-based representative positions do not
    exist. The predicted class of the anchor is reported alongside.

    Returns:
        (list[tuple[int, int, int, int, int]]): ``(image, level, row, col, class)`` tuples.
    """
    scores = preds["scores"].to(torch.float32).sigmoid()  # (B, nc, A)
    if probe_classes is not None and probe_classes.numel() > 0:
        scores = scores[:, probe_classes]
    confidence, predicted = scores.amax(dim=1), scores.argmax(dim=1)  # (B, A)
    if probe_classes is not None and probe_classes.numel() > 0:
        predicted = probe_classes[predicted]
    selected, _ = confidence.topk(per_image, dim=1)  # (B, K)
    indices = confidence.topk(per_image, dim=1).indices
    positions = []
    for image in range(scores.shape[0]):
        for flat in indices[image].tolist():
            level = int(levels[flat])
            height, width = shapes[level]
            local = flat - offsets[level]
            positions.append((image, level, local // width, local % width, int(predicted[image, flat])))
    return positions


def gt_boxes_xyxy(batch, imgsz):
    """Return the per-image GT boxes in model-input pixels and their classes."""
    boxes = batch["bboxes"].reshape(-1, 4).clone()
    scale = torch.tensor([imgsz, imgsz, imgsz, imgsz], device=boxes.device, dtype=boxes.dtype)
    boxes = boxes * scale
    xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=1)
    return xyxy, batch["batch_idx"].reshape(-1), batch["cls"].reshape(-1).long()


def anchor_grid(shapes, stride, device):
    """Return anchor pixel coordinates and the level index of every anchor."""
    anchors, stride_tensor = make_anchors(
        [torch.zeros(1, 1, height, width, device=device) for height, width in shapes], stride, 0.5
    )
    levels = torch.repeat_interleave(
        torch.arange(len(shapes), device=device),
        torch.tensor([height * width for height, width in shapes], device=device),
    )
    return anchors * stride_tensor, levels


def representative_positions(assignment, batch, imgsz, shapes, anchors, levels, center_ratio):
    """Return one representative anchor per GT instance and detection level.

    The anchor maximises the task-aligned score among the anchors assigned to that
    instance and lying inside the centre region of its box (method_plan section 4.3).

    Returns:
        (list[tuple[int, int, int, int, int]]): ``(image, level, row, col, class)`` tuples.
    """
    offsets = level_offsets(shapes)
    boxes, image_index, classes = gt_boxes_xyxy(batch, imgsz)
    selected = []
    for image in range(batch["img"].shape[0]):
        gt = (image_index == image).nonzero().flatten()
        if gt.numel() == 0:
            continue
        fg = assignment.fg_mask[image]
        if not bool(fg.any()):
            continue
        target = assignment.target_gt_idx[image]
        alpha = assignment.alpha[image]
        inside_centre = torch.zeros_like(fg)
        for position in range(gt.numel()):
            box = boxes[gt[position]]
            half_width = 0.5 * center_ratio * (box[2] - box[0])
            half_height = 0.5 * center_ratio * (box[3] - box[1])
            centre_x, centre_y = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            inside_centre |= (
                (anchors[:, 0] >= centre_x - half_width)
                & (anchors[:, 0] <= centre_x + half_width)
                & (anchors[:, 1] >= centre_y - half_height)
                & (anchors[:, 1] <= centre_y + half_height)
            )
        for position in range(gt.numel()):
            # target_gt_idx is the per-image position of the assigned GT, not the global index.
            assigned = fg & (target == position) & inside_centre
            if not bool(assigned.any()):
                continue
            for level in range(len(shapes)):
                candidates = assigned & (levels == level)
                if not bool(candidates.any()):
                    continue
                score = alpha.masked_fill(~candidates, -1.0)
                flat = int(score.argmax())
                local = flat - offsets[level]
                height, width = shapes[level]
                selected.append((image, level, local // width, local % width, int(classes[gt[position]])))
    return selected


def _flush(operator, pending):
    """Feed buffered patch vectors to the incremental PCA and clear the buffer."""
    if not pending:
        return
    operator.partial_fit(torch.cat(pending, dim=0))
    pending.clear()


def main():
    """Collect patches, fit the fixed PCA, cluster prototypes and save the artifact."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = YOLO(args.model).model.to(device).eval()
    head = model.model[-1]
    radius = head_patch_radius(head) if args.patch_radius < 0 else int(args.patch_radius)
    channels = head_input_channels(head)
    shapes_stride = head.stride
    if radius < 0:
        raise ValueError(f"patch_radius must be >= 0, got {radius}")

    # One warmup forward fixes the head anchors/strides and validates the input size.
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    # The criterion reads the loss gains from `model.args`, which a stripped checkpoint may store
    # as a plain dict; the assignment used here does not depend on the gain values.
    model.args = get_cfg(overrides={"imgsz": args.imgsz, "task": "detect", "rect": False})

    if not args.report_only and not args.output:
        raise ValueError("--output is required unless --report_only is set")
    if args.report_only and not args.load_hist:
        raise ValueError("--report_only needs --load_hist (the fixed PCA to measure)")
    if args.report_only and not (args.report_dims or args.report_similarity):
        raise ValueError("--report_only needs --report_dims and/or --report_similarity")
    loader, data = build_loader(args.data, args.imgsz, args.batch_size, args.workers, args.split)
    mapping = class_mapping(data["names"], model.names)
    mapping_ok = mapping is not None
    if not mapping_ok:
        if not args.report_only:
            raise KeyError(
                f"Dataset '{args.data}' holds classes unknown to the model; a prototype artifact needs a "
                f"model that knows every class (use --report_only for a pure measurement run)"
            )
        LOGGER.warning(
            f"Dataset '{args.data}' holds classes unknown to the model; sampling anchors by model confidence "
            f"and reporting the predicted class instead of the GT class"
        )
    criterion = RecordingDetectionLoss(model) if mapping_ok else None
    shapes = None
    anchors = levels = None

    history = None
    if args.load_hist:
        history_subspace, history_bank, history_meta = load_patch_artifact(args.load_hist)
        if history_subspace.radius != radius:
            raise ValueError(
                f"Historical artifact radius {history_subspace.radius} differs from the requested {radius}"
            )
        if list(history_subspace.channels) != channels:
            raise ValueError(
                f"Historical artifact head channels {list(history_subspace.channels)} differ from the model {channels}"
            )
        history_bank.bind_nc(head.nc)
        history = (history_subspace, history_bank, history_meta)
        existing = {int(class_id) for ids in history_bank.class_ids for class_id in ids.tolist()}
        LOGGER.info(f"Reusing the fixed PCA of '{args.load_hist}' with {len(existing)} known classes")

    requested = [int(part) for part in str(args.pca_dims).split(",")]
    if len(requested) == 1:
        requested = requested * len(channels)
    if len(requested) != len(channels):
        raise ValueError(f"pca_dims must give 1 or {len(channels)} values, got {args.pca_dims}")

    patch_size = (2 * radius + 1) ** 2
    operators = []
    if not args.no_compression and history is None:
        for dims in requested:
            if dims < 1:
                raise ValueError(f"pca_dims must be positive, got {dims}")
            operators.append(IncrementalPCAGPUWrapper(dims, device))
    pending = [[] for _ in channels]
    collected = [dict() for _ in channels]  # level -> class -> list of (n, D) chunks
    report_patches = [[] for _ in channels]
    report_classes = [[] for _ in channels]

    pbar = TQDM(loader, desc="Collecting patches")
    processed = 0
    for batch in pbar:
        images = batch["img"].to(device).float() / 255.0
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        batch["img"] = images
        with torch.no_grad():
            output = model(images)
            if criterion is not None:
                criterion(output, batch)  # records the task-aligned assignment
            feats = (output[1] if isinstance(output, tuple) else output)["feats"]
        batch_shapes = [(int(f.shape[-2]), int(f.shape[-1])) for f in feats]
        if shapes is None:
            shapes = batch_shapes
            anchors, levels = anchor_grid(shapes, shapes_stride, device)
        elif batch_shapes != shapes:
            raise RuntimeError(f"Batch feature shapes changed from {shapes} to {batch_shapes}")
        if criterion is not None:
            positions = representative_positions(
                criterion.assignment, batch, args.imgsz, shapes, anchors, levels, args.center_ratio
            )
        else:
            preds = output[1] if isinstance(output, tuple) else output
            positions = predicted_positions(
                preds,
                shapes,
                anchors,
                levels,
                level_offsets(shapes),
                args.report_anchors,
                probe_classes=torch.arange(model.model[-1].nc, device=device),
            )
        if positions:
            patch_levels = torch.tensor([item[1] for item in positions], device=device)
            images_idx = torch.tensor([item[0] for item in positions], device=device)
            rows = torch.tensor([item[2] for item in positions], device=device)
            cols = torch.tensor([item[3] for item in positions], device=device)
            patch_classes = torch.tensor([item[4] for item in positions], device=device)
            inside = torch.zeros(len(positions), dtype=torch.bool, device=device)
            for level in range(len(channels)):
                select = patch_levels == level
                height, width = shapes[level]
                inside[select] = valid_patch_positions(rows[select], cols[select], height, width, radius)
            keep = inside.nonzero().flatten()
            if keep.numel():
                patches = []
                for level in range(len(channels)):
                    select = keep[patch_levels[keep] == level]
                    if select.numel() == 0:
                        continue
                    level_patches = extract_patches(
                        feats,
                        level,
                        images_idx[select],
                        rows[select],
                        cols[select],
                        radius,
                    )
                    flat = level_patches.flatten(1).to(torch.float32)
                    patches.append((level, flat, patch_classes[select]))
                for level, flat, classes in patches:
                    if operators:
                        pending[level].append(flat)
                        buffered = sum(chunk.shape[0] for chunk in pending[level])
                        if buffered >= max(PCA_CHUNK, requested[level] * 4):
                            _flush(operators[level], pending[level])
                    for class_id in torch.unique(classes).tolist():
                        member = flat[classes == class_id]
                        bucket = collected[level].setdefault(int(class_id), [])
                        bucket.append(member.detach().cpu())
                    if args.report_dims or args.report_similarity:
                        for class_id in torch.unique(classes).tolist():
                            member = flat[classes == class_id]
                            budget = args.report_samples - sum(chunk.shape[0] for chunk in report_patches[level])
                            if budget > 0:
                                chunk = member[:budget].detach().cpu()
                                report_patches[level].append(chunk)
                                report_classes[level].append(
                                    torch.full((chunk.shape[0],), int(class_id), dtype=torch.long)
                                )
        processed += 1
        pbar.set_description(f"Collecting patches ({processed} batches)")
        if args.max_images and processed * args.batch_size >= args.max_images:
            break

    if not operators and history is None and not args.no_compression:
        raise RuntimeError("No patch was collected; check the dataset split and the assignment")
    for level, operator in enumerate(operators):
        _flush(operator, pending[level])
        if operator.operator.n_samples_seen_ < max(requested[level], 2):
            raise RuntimeError(
                f"Level {level} saw only {operator.operator.n_samples_seen_} patches, fewer than the "
                f"{requested[level]} requested PCA components; use a larger --max_images or fewer dimensions"
            )

    # Fixed PCA: fitted once at the base stage, reused verbatim afterwards.
    if history is not None:
        subspace = HeadPatchSubspace.from_state(history[0].state())
    elif args.no_compression:
        subspace = HeadPatchSubspace(
            means=[torch.zeros(channels[level] * patch_size) for level in range(len(channels))],
            components=None,
            channels=channels,
            radius=radius,
        )
    else:
        means, components = [], []
        for level, operator in enumerate(operators):
            dims = requested[level]
            if args.pca_energy_ratio > 0.0:
                ratio = torch.cumsum(operator.explained_variance_ratio_, dim=0)
                dims = int(torch.searchsorted(ratio, torch.tensor(args.pca_energy_ratio, device=ratio.device)).item()) + 1
                dims = max(1, min(dims, requested[level]))
            means.append(operator.mean_.detach().cpu())
            components.append(operator.components_[:dims].detach().cpu())
            LOGGER.info(
                f"Level {level}: PCA dims {dims}/{operator.components_.shape[0]} "
                f"(energy {float(torch.cumsum(operator.explained_variance_ratio_, 0)[dims - 1]):.4f})"
            )
        subspace = HeadPatchSubspace(means=means, components=components, channels=channels, radius=radius)
    subspace = subspace.to(device)

    if args.report_only:
        if args.report_dims:
            report = reconstruction_report(args, subspace, model, head, radius, channels, report_patches, report_classes)
            path = args.report_csv or f"{os.path.splitext(args.load_hist)[0]}.{Path(args.data).stem}.{args.split}.csv"
            write_report(path, report)
        if args.report_similarity:
            report_similarity(args, subspace, history[1], report_patches, report_classes)
        return

    # Per-class local components in the fixed codec.
    if history is not None:
        bank = MultiCenterBank.from_state(history[1].state())
    else:
        dims_per_level = subspace.dims
        bank = MultiCenterBank(
            class_ids=[torch.empty(0, dtype=torch.long) for _ in channels],
            means=[torch.empty(0, dims_per_level[level]) for level in range(len(channels))],
            variances=[torch.empty(0, dims_per_level[level]) for level in range(len(channels))],
            weights=[torch.empty(0) for _ in channels],
            nc=head.nc,
        )
    generator = torch.Generator().manual_seed(args.seed)
    counts = {}
    for level in range(len(channels)):
        for class_id in sorted(collected[level]):
            if class_id >= head.nc:
                raise ValueError(f"Collected class {class_id} is outside the model class space ({head.nc})")
            if history is not None and class_id in {
                int(value) for value in history[1].class_ids[level].tolist()
            }:
                raise ValueError(f"Class {class_id} already has prototypes at level {level}; refusing to duplicate")
            chunks = collected[level][class_id]
            if not chunks:
                continue
            samples = torch.cat(chunks, dim=0)
            if samples.shape[0] > args.max_samples_per_class:
                keep = torch.randperm(samples.shape[0], generator=generator)[: args.max_samples_per_class]
                samples = samples[keep]
            codes = subspace.encode(level, samples.to(device).reshape(-1, channels[level], 2 * radius + 1, 2 * radius + 1))
            codes = codes.detach().cpu()
            means, variances, weights = MultiCenterBank.cluster(
                codes,
                split_ratio=args.split_ratio,
                min_cluster=args.min_cluster,
                min_centers=args.min_centers,
                max_centers=args.max_centers,
                merge_ratio=args.merge_ratio,
                iters=args.iters,
                seed=args.seed,
            )
            if 0.0 < args.var_shrinkage < 1.0:
                pooled = codes.var(dim=0, unbiased=False)
                variances = (1.0 - args.var_shrinkage) * variances + args.var_shrinkage * pooled
            bank.append(level, class_id, means, variances.clamp_min(0.0), weights)
            counts[(level, class_id)] = int(means.shape[0])
            LOGGER.info(f"Level {level} class {class_id}: {means.shape[0]} components from {samples.shape[0]} patches")

    if bank.num_components == 0:
        raise RuntimeError("No prototype was built; the artifact would be unusable")

    stats = {
        "patch_count": {
            f"{level}:{class_id}": int(sum(chunk.shape[0] for chunk in collected[level][class_id]))
            for (level, class_id) in counts
        },
        "num_centers": {f"{level}:{class_id}": value for (level, class_id), value in counts.items()},
        "dims": subspace.dims,
        "no_compression": subspace.components is None,
        "covariance": "diag",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_patch_artifact(
        args.output,
        subspace,
        bank,
        history_nc=int(head.nc),
        class_names={int(key): str(value) for key, value in model.names.items()},
        stats=stats,
    )
    size = os.path.getsize(args.output)
    LOGGER.info(
        f"Saved {args.output}: {bank.num_components} components, {len(channels)} levels, "
        f"dims {subspace.dims}, {size / 1e6:.1f} MB"
    )

    if args.report_dims:
        report = reconstruction_report(args, subspace, model, head, radius, channels, report_patches, report_classes)
        path = args.report_csv or f"{os.path.splitext(args.output)[0]}.reconstruction.csv"
        write_report(path, report)
    if args.report_similarity:
        report_similarity(args, subspace, bank, report_patches, report_classes)


class IncrementalPCAGPUWrapper:
    """Thin adapter over `pca_on_gpu.IncrementalPCAonGPU` with a fixed component count."""

    def __init__(self, n_components, device):
        """Bind the operator to a device."""
        from pca_on_gpu import IncrementalPCAonGPU

        self.operator = IncrementalPCAonGPU(n_components=n_components, device=str(device))

    def partial_fit(self, samples):
        """Feed one chunk and return self."""
        self.operator.partial_fit(samples)
        return self

    @property
    def components_(self):
        """Fitted principal directions, ``(d, D)``."""
        return self.operator.components_

    @property
    def mean_(self):
        """Fitted patch mean, ``(D,)``."""
        return self.operator.mean_

    @property
    def explained_variance_ratio_(self):
        """Explained variance ratio of every component."""
        return self.operator.explained_variance_ratio_


def _head_forward(head, level, patches, center):
    """Return the classification and regression head output at the patch centre."""
    logits = head.cv3[level](patches)[:, :, center, center]
    boxes = head.cv2[level](patches)[:, :, center, center]
    return logits, boxes


def reconstruction_report(args, subspace, model, head, radius, channels, report_patches, report_classes):
    """Measure the patch reconstruction deviation of every candidate code dimension.

    Returns:
        (list[dict]): rows with ``level, dim, class, n, E_feat, E_cls, E_reg``.
    """
    if subspace.components is None:
        LOGGER.warning("No-compression artifact: the reconstruction report needs a fitted PCA")
        return []
    dims = [int(dim) for dim in args.report_dims]
    center = radius
    rows = []
    with torch.no_grad():
        for level in range(len(channels)):
            if not report_patches[level]:
                continue
            patches = torch.cat(report_patches[level], dim=0).to(next(model.parameters()).device)
            labels = torch.cat(report_classes[level], dim=0)
            patches = patches.reshape(-1, channels[level], 2 * radius + 1, 2 * radius + 1)
            reference_logits, reference_boxes = _head_forward(head, level, patches, center)
            reference_probs = reference_logits.sigmoid()
            for dim in dims:
                dim = min(dim, subspace.components[level].shape[0])
                truncated = HeadPatchSubspace(
                    means=[subspace.means[level]],
                    components=[subspace.components[level][:dim]],
                    channels=[channels[level]],
                    radius=radius,
                )
                codes = truncated.encode(0, patches)
                rebuilt = truncated.decode(0, codes)
                rebuilt_logits, rebuilt_boxes = _head_forward(head, level, rebuilt, center)
                relative = ((rebuilt.flatten(1) - patches.flatten(1)).norm(dim=1)
                            / patches.flatten(1).norm(dim=1).clamp_min(EPS))
                class_l1 = (rebuilt_logits.sigmoid() - reference_probs).abs().sum(dim=1)
                reg_err = F.smooth_l1_loss(rebuilt_boxes, reference_boxes, reduction="none", beta=1.0).mean(dim=1)
                rows.append(
                    {
                        "level": level,
                        "dim": dim,
                        "class": "all",
                        "n": int(patches.shape[0]),
                        "E_feat": float(relative.mean()),
                        "E_cls": float(class_l1.mean()),
                        "E_reg": float(reg_err.mean()),
                    }
                )
                for class_id in torch.unique(labels).tolist():
                    member = labels == class_id
                    rows.append(
                        {
                            "level": level,
                            "dim": dim,
                            "class": int(class_id),
                            "n": int(member.sum()),
                            "E_feat": float(relative[member].mean()),
                            "E_cls": float(class_l1[member].mean()),
                            "E_reg": float(reg_err[member].mean()),
                        }
                    )
    return rows


def write_report(path, rows):
    """Write the reconstruction report as CSV."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "dim", "class", "n", "E_feat", "E_cls", "E_reg"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info(f"Wrote reconstruction report to {path}")


def report_similarity(args, subspace, bank, report_patches, report_classes):
    """Report the prototype cosine similarity of in-distribution patches per level."""
    for level in range(bank.num_levels):
        if not report_patches[level] or bank.class_ids[level].numel() == 0:
            continue
        patches = torch.cat(report_patches[level], dim=0)
        labels = torch.cat(report_classes[level], dim=0)
        channels = subspace.channels[level]
        patches = patches.reshape(-1, channels, 2 * subspace.radius + 1, 2 * subspace.radius + 1).to(
            subspace.means[level].device
        )
        codes = subspace.encode(level, patches).to(bank.means[level].device)
        similarity, has = bank.max_cosine(level, labels.to(codes.device), codes)
        similarity = similarity[has]
        if similarity.numel() == 0:
            continue
        quantiles = torch.quantile(
            similarity, torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75], device=similarity.device)
        )
        LOGGER.info(
            f"Level {level} prototype similarity (n={similarity.numel()}): "
            f"p05={quantiles[0]:.3f} p10={quantiles[1]:.3f} p25={quantiles[2]:.3f} "
            f"p50={quantiles[3]:.3f} p75={quantiles[4]:.3f}"
        )


if __name__ == "__main__":
    main()
