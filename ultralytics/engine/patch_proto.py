"""Shared infrastructure for detection-head input patch prototypes.

A prototype lives in the patch space of the detection-head input feature ``X_s``: an
instance centred at anchor ``(i, j)`` of level ``s`` is represented by the
``(2r_s + 1) x (2r_s + 1)`` window around that anchor, where ``r_s`` is the receptive
radius of the head branches.  Prototypes are stored in a fixed, base-stage PCA subspace
plus a per-class set of local components (mean, diagonal variance, mixing weight).

`tools/generate_patch_prototypes.py` builds the artifact offline; the training-time
losses in `prad.py` / `prd.py` consume it read-only.
"""

from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss

PATCH_PROTO_VERSION = 1
EPS = 1e-8


def _branch_radius(branch):
    """Return the receptive radius of one detection-head branch for stride-1 outputs."""
    radius = 0
    for module in branch.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if tuple(module.stride) != (1, 1):
            raise ValueError(f"Detect branch conv must have stride 1, got stride {tuple(module.stride)}")
        if module.kernel_size[0] % 2 == 0 or module.kernel_size[1] % 2 == 0:
            raise ValueError(f"Detect branch conv must have odd kernels, got {tuple(module.kernel_size)}")
        radius += module.dilation[0] * (module.kernel_size[0] - 1) // 2
    return radius


def head_patch_radius(head):
    """Return the patch radius ``r = max(r_cls, r_reg)`` shared by every head level.

    Args:
        head (Detect): Detection head whose branches define the patch extent.

    Returns:
        (int): Patch radius; the patch is a ``(2r + 1) x (2r + 1)`` window.
    """
    if not isinstance(head, Detect):
        raise TypeError(f"Patch prototypes require a Detect head, got {type(head)}")
    radii = []
    for level in range(head.nl):
        radii.append(max(_branch_radius(head.cv3[level]), _branch_radius(head.cv2[level])))
    if len(set(radii)) != 1:
        raise ValueError(f"All Detect levels must share one patch radius, got {radii}")
    if radii[0] < 1:
        raise ValueError(f"Detect branches have no spatial receptive field (radius {radii[0]})")
    return radii[0]


def head_input_channels(head):
    """Return the detection-head input channels of every level."""
    channels = []
    for level in range(head.nl):
        convs = [module for module in head.cv3[level].modules() if isinstance(module, torch.nn.Conv2d)]
        if not convs:
            raise TypeError(f"Detect head level {level} has no convolution to read its input channels from")
        channels.append(int(convs[0].in_channels))
    return channels


def level_shapes(feats):
    """Return the ``(H, W)`` pair of every head input level."""
    return [(int(f.shape[-2]), int(f.shape[-1])) for f in feats]


def level_offsets(shapes):
    """Return cumulative anchor offsets of the head levels (level-major anchor order)."""
    offsets = [0]
    for height, width in shapes:
        offsets.append(offsets[-1] + height * width)
    return offsets


def anchor_levels(shapes, device):
    """Return the level index of every anchor, shaped ``(A,)``."""
    sizes = [height * width for height, width in shapes]
    return torch.repeat_interleave(
        torch.arange(len(sizes), device=device), torch.tensor(sizes, device=device)
    )


def valid_patch_positions(rows, cols, height, width, radius):
    """Return which anchor positions carry a full patch inside the feature map."""
    return (rows >= radius) & (rows < height - radius) & (cols >= radius) & (cols < width - radius)


def extract_patches(feats, level, batch_idx, rows, cols, radius):
    """Gather detection-head input patches centred on the given anchors.

    Args:
        feats (list[torch.Tensor]): Head input features, one ``(B, C, H, W)`` tensor per level.
        level (int): Level index addressed by ``rows``/``cols``.
        batch_idx (torch.Tensor): Image index per position, ``(K,)``.
        rows (torch.Tensor): Anchor row per position, ``(K,)``.
        cols (torch.Tensor): Anchor column per position, ``(K,)``.
        radius (int): Patch radius; the patch is ``(2r + 1)`` wide.

    Returns:
        (torch.Tensor): Patches shaped ``(K, C, 2r + 1, 2r + 1)``.
    """
    if batch_idx.numel() == 0:
        channels = feats[level].shape[1]
        size = 2 * radius + 1
        return feats[level].new_zeros((0, channels, size, size))
    # The unfold chain is a view; only the selected positions are copied.
    windows = F.pad(feats[level], (radius, radius, radius, radius)).unfold(2, 2 * radius + 1, 1).unfold(
        3, 2 * radius + 1, 1
    )
    patches = []
    for image in torch.unique(batch_idx).tolist():
        select = batch_idx == image
        picked = windows[image][:, rows[select], cols[select]]  # (C, K_i, k, k)
        patches.append(picked.permute(1, 0, 2, 3))
    return torch.cat(patches, dim=0)


class Assignment(NamedTuple):
    """Task-aligned assignment recorded by `RecordingDetectionLoss`."""

    fg_mask: torch.Tensor  # (B, A) bool
    target_gt_idx: torch.Tensor  # (B, A) long
    target_labels: torch.Tensor  # (B, A) long, assigned GT class
    alpha: torch.Tensor  # (B, A) float, normalized alignment score


class _AssignmentRecorder:
    """Proxy a criterion assigner and keep its last assignment on the side."""

    def __init__(self, assigner):
        object.__setattr__(self, "assigner", assigner)
        object.__setattr__(self, "fg_mask", None)
        object.__setattr__(self, "target_gt_idx", None)
        object.__setattr__(self, "target_labels", None)
        object.__setattr__(self, "alpha", None)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # keep the copy/pickle protocol off the wrapped assigner
        return getattr(object.__getattribute__(self, "assigner"), name)

    def __reduce__(self):
        return (_AssignmentRecorder, (self.assigner,))

    def __deepcopy__(self, memo):
        clone = _AssignmentRecorder.__new__(_AssignmentRecorder)
        memo[id(self)] = clone
        for name in ("assigner", "fg_mask", "target_gt_idx", "target_labels", "alpha"):
            object.__setattr__(clone, name, copy.deepcopy(object.__getattribute__(self, name), memo))
        return clone

    def __call__(self, *args, **kwargs):
        target_labels, _, target_scores, fg_mask, target_gt_idx = self.assigner(*args, **kwargs)
        object.__setattr__(self, "fg_mask", fg_mask)
        object.__setattr__(self, "target_gt_idx", target_gt_idx)
        object.__setattr__(self, "target_labels", target_labels)
        object.__setattr__(self, "alpha", target_scores.amax(dim=-1))
        return target_labels, _, target_scores, fg_mask, target_gt_idx


class RecordingDetectionLoss(v8DetectionLoss):
    """Detection criterion that exposes the task-aligned assignment it already computes.

    The prototype losses need the positive-anchor set assigned to each ground-truth
    instance (new-class mask, representative patch positions).  Recording the assigner
    output avoids a second assignment pass and leaves `ultralytics/utils/loss.py`
    untouched.
    """

    def __init__(self, model):
        """Bind the detection criterion and install the assignment recorder."""
        super().__init__(model)
        self.assignment_recorder = _AssignmentRecorder(self.assigner)
        self.assigner = self.assignment_recorder

    @property
    def assignment(self):
        """Return the assignment of the most recent forward pass."""
        recorder = self.assignment_recorder
        if recorder.fg_mask is None:
            raise RuntimeError("No task-aligned assignment recorded yet; run the loss before reading it")
        return Assignment(
            recorder.fg_mask,
            recorder.target_gt_idx,
            recorder.target_labels,
            recorder.alpha,
        )


def get_assignment(criterion):
    """Return the recorded assignment of `RecordingDetectionLoss`, unwrapping end-to-end criteria."""
    inner = getattr(criterion, "one2many", criterion)
    if not isinstance(inner, RecordingDetectionLoss):
        raise TypeError(
            f"Patch prototype losses need a RecordingDetectionLoss criterion, got {type(inner)}. "
            f"Disable end2end training for this method."
        )
    return inner.assignment


def new_class_mask(assignment, history_nc):
    """Return the anchors that the current task's new-class GT owns, ``(B, A)``."""
    return assignment.fg_mask & (assignment.target_labels >= history_nc)


class HeadPatchSubspace:
    """Fixed shared per-level PCA of the detection-head input patch space.

    Args:
        means (list[torch.Tensor]): Per-level patch means, ``(D_s,)``  where ``D_s = C_s (2r+1)^2``.
        components (list[torch.Tensor] | None): Per-level bases, ``(d_s, D_s)``; ``None``
            means no compression (identity codec, ``d_s = D_s``).
        channels (list[int]): Head input channels per level.
        radius (int): Patch radius shared by every level.
    """

    def __init__(self, means, components, channels, radius):
        """Validate and bind the per-level codec."""
        self.channels = [int(c) for c in channels]
        self.radius = int(radius)
        self.means = [mean.to(torch.float32) for mean in means]
        self.components = None if components is None else [comp.to(torch.float32) for comp in components]
        patch_size = (2 * self.radius + 1) ** 2
        if len(self.means) != len(self.channels):
            raise ValueError(f"Subspace needs one mean per level, got {len(self.means)} means for {len(self.channels)}")
        if self.components is not None and len(self.components) != len(self.means):
            raise ValueError(f"Subspace needs one base per level, got {len(self.components)}")
        for level, (mean, channels) in enumerate(zip(self.means, self.channels)):
            expected = channels * patch_size
            if mean.shape != (expected,):
                raise ValueError(f"Level {level} patch mean must be ({expected},), got {tuple(mean.shape)}")
            if self.components is not None and self.components[level].shape[1] != expected:
                raise ValueError(
                    f"Level {level} basis must have {expected} columns, got {self.components[level].shape[1]}"
                )

    @property
    def num_levels(self):
        """Return the number of detection levels."""
        return len(self.channels)

    @property
    def dims(self):
        """Return the code dimension of every level."""
        if self.components is None:
            patch_size = (2 * self.radius + 1) ** 2
            return [channels * patch_size for channels in self.channels]
        return [int(comp.shape[0]) for comp in self.components]

    def to(self, device):
        """Move the codec tensors to a device in place and return self."""
        self.means = [mean.to(device) for mean in self.means]
        if self.components is not None:
            self.components = [comp.to(device) for comp in self.components]
        return self

    def encode(self, level, patches):
        """Project patches ``(K, C, 2r+1, 2r+1)`` onto the level code, ``(K, d_s)``."""
        flat = patches.flatten(1).to(torch.float32)
        if self.components is None:
            return flat
        return (flat - self.means[level]) @ self.components[level].t()

    def decode(self, level, features):
        """Reconstruct patches ``(K, C, 2r+1, 2r+1)`` from level codes ``(K, d_s)``."""
        features = features.to(torch.float32)
        if self.components is None:
            flat = features
        else:
            flat = self.means[level] + features @ self.components[level]
        size = 2 * self.radius + 1
        return flat.reshape(-1, self.channels[level], size, size)

    def state(self):
        """Return a CPU-serializable snapshot."""
        return {
            "channels": list(self.channels),
            "radius": self.radius,
            "means": [mean.detach().cpu() for mean in self.means],
            "components": None if self.components is None else [comp.detach().cpu() for comp in self.components],
        }

    @classmethod
    def from_state(cls, state):
        """Rebuild a subspace from `state`."""
        return cls(
            means=state["means"],
            components=state["components"],
            channels=state["channels"],
            radius=state["radius"],
        )


class MultiCenterBank:
    """Per-class multi-centre prototypes with diagonal variances in a patch subspace.

    Args:
        class_ids (list[torch.Tensor]): Per-level prototype class ids, ``(N_s,)``.
        means (list[torch.Tensor]): Per-level component means, ``(N_s, d_s)``.
        variances (list[torch.Tensor]): Per-level diagonal variances, ``(N_s, d_s)``.
        weights (list[torch.Tensor]): Per-level mixing weights, normalized within a class.
        nc (int): Number of model classes the bank may hold.
    """

    def __init__(self, class_ids, means, variances, weights, nc):
        """Validate and bind the per-level component tables."""
        self.nc = int(nc)
        self.class_ids = [ids.to(torch.long) for ids in class_ids]
        self.means = [mean.to(torch.float32) for mean in means]
        self.variances = [var.to(torch.float32) for var in variances]
        self.weights = [weight.to(torch.float32) for weight in weights]
        if not len(self.class_ids) == len(self.means) == len(self.variances) == len(self.weights):
            raise ValueError("MultiCenterBank needs matching per-level class/mean/variance/weight lists")
        for level, ids in enumerate(self.class_ids):
            count = ids.numel()
            if self.means[level].shape[0] != count:
                raise ValueError(f"Level {level} has {count} class ids but {self.means[level].shape[0]} means")
            if self.variances[level].shape != self.means[level].shape:
                raise ValueError(f"Level {level} variance shape {tuple(self.variances[level].shape)} != mean shape")
            if self.weights[level].shape != ids.shape:
                raise ValueError(f"Level {level} weight shape {tuple(self.weights[level].shape)} != id shape")
            if count and (int(ids.min()) < 0 or int(ids.max()) >= self.nc):
                raise ValueError(f"Level {level} holds class ids outside [0, {self.nc})")

    @property
    def num_levels(self):
        """Return the number of detection levels."""
        return len(self.class_ids)

    @property
    def num_components(self):
        """Return the total number of local components."""
        return int(sum(ids.numel() for ids in self.class_ids))

    def bind_nc(self, nc):
        """Bind the bank to a consumer class space and validate every stored class id.

        The artifact records the class count of the model that built it; a later stage has a
        larger head, so the bank is bound to the consumer's count before use.
        """
        nc = int(nc)
        for level, ids in enumerate(self.class_ids):
            if ids.numel() and int(ids.max()) >= nc:
                raise ValueError(
                    f"Level {level} holds class {int(ids.max())}, outside the consumer class space [0, {nc})"
                )
        self.nc = nc
        return self

    def classes(self, level):
        """Return the sorted class ids present at one level."""
        if self.class_ids[level].numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.class_ids[level].device)
        return torch.unique(self.class_ids[level])

    def to(self, device):
        """Move the component tensors to a device in place and return self."""
        self.class_ids = [ids.to(device) for ids in self.class_ids]
        self.means = [mean.to(device) for mean in self.means]
        self.variances = [var.to(device) for var in self.variances]
        self.weights = [weight.to(device) for weight in self.weights]
        return self

    def state(self):
        """Return a CPU-serializable snapshot."""
        return {
            "nc": self.nc,
            "class_ids": [ids.detach().cpu() for ids in self.class_ids],
            "means": [mean.detach().cpu() for mean in self.means],
            "variances": [var.detach().cpu() for var in self.variances],
            "weights": [weight.detach().cpu() for weight in self.weights],
        }

    @classmethod
    def from_state(cls, state):
        """Rebuild a bank from `state`."""
        return cls(
            class_ids=state["class_ids"],
            means=state["means"],
            variances=state["variances"],
            weights=state["weights"],
            nc=state["nc"],
        )

    def append(self, level, class_id, means, variances, weights):
        """Append the components of one class to one level, keeping older classes untouched."""
        if int(class_id) < 0 or int(class_id) >= self.nc:
            raise ValueError(f"Class id {class_id} outside [0, {self.nc})")
        if means.shape[0] == 0:
            raise ValueError(f"Refusing to append an empty prototype set for class {class_id}")
        if self.class_ids[level].numel() and int(class_id) <= int(self.class_ids[level].max()):
            raise ValueError(
                f"Class {class_id} must be appended after the existing classes "
                f"(max {int(self.class_ids[level].max())}) of level {level}"
            )
        self.class_ids[level] = torch.cat([self.class_ids[level], torch.full((means.shape[0],), int(class_id), dtype=torch.long, device=means.device)])
        self.means[level] = torch.cat([self.means[level].to(means.device), means])
        self.variances[level] = torch.cat([self.variances[level].to(variances.device), variances])
        self.weights[level] = torch.cat([self.weights[level].to(weights.device), weights])

    def max_cosine(self, level, query_classes, features):
        """Return the best prototype cosine similarity per query and whether a prototype existed.

        Args:
            level (int): Detection level of the queries.
            query_classes (torch.Tensor): Class id per query, ``(K,)``.
            features (torch.Tensor): Codes of the queries, ``(K, d_s)``.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Similarities ``(K,)`` and a ``(K,)`` mask
                that is False where the query class has no prototype at this level.
        """
        if features.shape[0] == 0:
            return features.new_zeros(0), torch.zeros(0, dtype=torch.bool, device=features.device)
        if self.class_ids[level].numel() == 0:
            return features.new_zeros(features.shape[0]), torch.zeros(
                features.shape[0], dtype=torch.bool, device=features.device
            )
        prototypes = F.normalize(self.means[level], dim=1)
        queries = F.normalize(features, dim=1)
        similarity = queries @ prototypes.t()  # (K, N)
        allowed = self.class_ids[level].unsqueeze(0) == query_classes.unsqueeze(1)  # (K, N)
        similarity = similarity.masked_fill(~allowed, -2.0)
        best, _ = similarity.max(dim=1)
        return best, allowed.any(dim=1)

    def sample(self, level, count, generator, std_scale):
        """Draw `count` codes from the level's component mixture.

        Classes are sampled uniformly (so rare classes are not drowned out) and the local
        component is drawn from the class-internal mixing weights.

        Args:
            level (int): Detection level to sample from.
            count (int): Number of codes to draw.
            generator (torch.Generator): CPU generator, so every DDP rank draws its own set.
            std_scale (float): Multiplier on the diagonal standard deviation.

        Returns:
            (torch.Tensor | None): Codes ``(count, d_s)`` on the level device, or None when
                the level has no component.
        """
        ids = self.class_ids[level]
        classes = self.classes(level)
        if count <= 0 or classes.numel() == 0:
            return None
        device = ids.device
        dim = self.means[level].shape[1]
        draws, per_level_counts = [], []
        base, remainder = divmod(int(count), int(classes.numel()))
        for position, class_id in enumerate(classes.tolist()):
            take = base + (1 if position < remainder else 0)
            if take == 0:
                continue
            members = (ids == class_id).nonzero().flatten()
            weights = self.weights[level][members]
            weights = weights / weights.sum().clamp_min(EPS)
            picked = members[
                torch.multinomial(weights.detach().cpu(), take, replacement=True, generator=generator).to(members.device)
            ]
            mean = self.means[level][picked]
            std = self.variances[level][picked].clamp_min(EPS).sqrt()
            noise = torch.randn(take, dim, generator=generator).to(mean.device, mean.dtype)
            draws.append(mean + float(std_scale) * std * noise)
            per_level_counts.append(take)
        if not draws:
            return None
        return torch.cat(draws, dim=0).to(device)

    @staticmethod
    def _two_means(points):
        """Split `points` into two deterministic means."""
        mean = points.mean(dim=0)
        first = points[(points - mean).norm(dim=1).argmax()]
        second = points[(points - first).norm(dim=1).argmax()]
        centers = torch.stack([first, second])
        for _ in range(10):
            assign = torch.cdist(points, centers).argmin(dim=1)
            if not bool((assign == 0).any()) or not bool((assign == 1).any()):
                break
            updated = torch.stack([points[assign == 0].mean(dim=0), points[assign == 1].mean(dim=0)])
            converged = torch.allclose(updated, centers)
            centers = updated
            if converged:
                break
        return centers[0], centers[1]

    @staticmethod
    def cluster(samples, split_ratio=0.1, min_cluster=5, min_centers=1, max_centers=8, merge_ratio=0.5, iters=20, seed=0):
        """Cluster one class/level prototype set into local components.

        Minimizes the DP-Means objective ``sum_i ||h_i - mu_{a_i}||^2 + lambda * M`` by
        repeatedly applying the best two-means split whose squared-error reduction exceeds
        ``lambda = split_ratio * (class total squared error)``, so a unimodal class keeps one
        centre while a multimodal one splits until the gain stops paying for the penalty.
        Clusters below the size floor, centres closer than ``merge_ratio`` times their scale
        and components beyond the storage budget are dropped.

        Args:
            samples (torch.Tensor): Low-dimensional instance codes, ``(N, d)``.
            split_ratio (float): ``lambda`` as a fraction of the class total squared error.
            min_cluster (int): Minimum cluster size kept.
            min_centers (int): Minimum number of components kept per class/level.
            max_centers (int): Storage budget on components per class/level.
            merge_ratio (float): Centres closer than ``merge_ratio * sqrt(lambda)`` are merged.
            iters (float): Maximum number of splits considered.
            seed (int): Reserved for reproducibility of the caller's subsampling.

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Means ``(M, d)``,
                diagonal variances ``(M, d)`` and mixing weights ``(M,)``.
        """
        samples = samples.to(torch.float32).cpu()
        count = samples.shape[0]
        if count == 0:
            raise ValueError("Cannot cluster an empty prototype set")
        centers = [samples.mean(dim=0)]
        total_error = float(((samples - centers[0]) ** 2).sum())
        penalty = max(float(split_ratio), 0.0) * total_error
        merge_radius = float(merge_ratio) * math.sqrt(max(penalty, 0.0) / max(count, 1))
        for _ in range(max(1, int(iters))):
            if len(centers) >= max_centers:
                break
            matrix = torch.stack(centers)
            assign = torch.cdist(samples, matrix).argmin(dim=1)
            best = None
            for index in range(len(centers)):
                members = samples[assign == index]
                if members.shape[0] < max(2 * min_cluster, 4):
                    continue
                current = float(((members - members.mean(dim=0)) ** 2).sum())
                first, second = MultiCenterBank._two_means(members)
                side = torch.cdist(members, torch.stack([first, second])).argmin(dim=1)
                if not bool((side == 0).any()) or not bool((side == 1).any()):
                    continue
                split = float(
                    ((members[side == 0] - first) ** 2).sum() + ((members[side == 1] - second) ** 2).sum()
                )
                gain = current - split
                if gain > penalty and (best is None or gain > best[0]):
                    best = (gain, index, first, second)
            if best is None:
                break
            _, index, first, second = best
            centers = [center for position, center in enumerate(centers) if position != index] + [first, second]
        # Merge over-close centres, then enforce the size floor and the storage budget.
        while len(centers) > 1 and merge_radius > 0.0:
            matrix = torch.stack(centers)
            distances = torch.cdist(matrix, matrix)
            distances.fill_diagonal_(float("inf"))
            closest = float(distances.min())
            if closest >= merge_radius:
                break
            first, second = (int(i) for i in torch.nonzero(distances == closest)[0])
            merged = 0.5 * (centers[first] + centers[second])
            centers = [center for index, center in enumerate(centers) if index not in (first, second)] + [merged]
        while True:
            matrix = torch.stack(centers)
            assign = torch.cdist(samples, matrix).argmin(dim=1)
            sizes = torch.bincount(assign, minlength=len(centers))
            drop = {index for index, size in enumerate(sizes.tolist()) if size < min_cluster}
            if not drop or len(centers) - len(drop) < max(min_centers, 1):
                break
            centers = [center for index, center in enumerate(centers) if index not in drop]
        if len(centers) > max_centers:
            matrix = torch.stack(centers)
            assign = torch.cdist(samples, matrix).argmin(dim=1)
            sizes = torch.bincount(assign, minlength=len(centers))
            keep = torch.argsort(sizes, descending=True, stable=True)[:max_centers].sort().values.tolist()
            centers = [centers[index] for index in keep]
        matrix = torch.stack(centers)
        assign = torch.cdist(samples, matrix).argmin(dim=1)
        means, variances, weights = [], [], []
        for index in range(len(centers)):
            members = samples[assign == index]
            if members.shape[0] == 0:
                continue
            means.append(members.mean(dim=0))
            variances.append(members.var(dim=0, unbiased=False))
            weights.append(torch.tensor(float(members.shape[0]), dtype=torch.float32))
        if not means:
            raise RuntimeError("Prototype clustering dropped every component")
        means = torch.stack(means)
        variances = torch.stack(variances)
        weights = torch.stack(weights)
        return means, variances, weights / weights.sum()


def save_patch_artifact(path, subspace, bank, history_nc, class_names, stats=None):
    """Write the patch PCA and multi-centre prototypes to `path`."""
    torch.save(
        {
            "version": PATCH_PROTO_VERSION,
            "patch_radius": subspace.radius,
            "pca": subspace.state(),
            "levels": bank.state(),
            "history_nc": int(history_nc),
            "class_names": dict(class_names),
            "stats": {} if stats is None else dict(stats),
        },
        path,
    )


def load_patch_artifact(path):
    """Read a patch prototype artifact and return ``(subspace, bank, meta)``."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    version = artifact.get("version")
    if version != PATCH_PROTO_VERSION:
        raise ValueError(f"Unsupported patch prototype artifact version {version} in '{path}'")
    subspace = HeadPatchSubspace.from_state(artifact["pca"])
    bank = MultiCenterBank.from_state(artifact["levels"])
    if subspace.radius != int(artifact["patch_radius"]):
        raise ValueError(
            f"Patch artifact '{path}' stores radius {artifact['patch_radius']} but its codec has radius "
            f"{subspace.radius}"
        )
    meta = {key: value for key, value in artifact.items() if key not in {"pca", "levels"}}
    return subspace, bank, meta
