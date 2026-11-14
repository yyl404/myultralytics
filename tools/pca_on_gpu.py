"""
This file is based on IncrementalPCAonGPU.py by david.ng@mpg-alumni.de
Original repository: https://github.com/dnhkng/PCAonGPU.git

Modifications made by YYL for MyUltralytics

Original license:
MIT License

Copyright (c) 2023 David Ng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import contextlib
import math
from numbers import Integral, Real

import torch


class IncrementalPCAonGPU:
    """Incremental PCA with optional sample weights and CPU/CUDA execution.

    Full SVD is the default. ``gram=True`` selects a wide-matrix Gram solver
    with full-SVD fallback; ``lowrank=True`` selects randomized SVD, controlled
    by ``lowrank_q``, ``lowrank_niter`` and ``lowrank_seed``. These modes are
    mutually exclusive. Gram matrices square the condition number.

    ``device=None`` selects CUDA when available, otherwise CPU. ``fit`` transfers
    one input batch at a time. ``copy=False`` permits first-batch centering in
    place; ``transform`` always preserves input. ``whiten=True`` divides projected
    coordinates by the square root of the learned explained variance.

    ``stats_dtype`` selects float32/float64 accumulation (including the matching
    complex dtype for complex means). By default CPU statistics use float64 and
    CUDA statistics retain the input precision. ``svd_driver`` is CUDA-only.
    ``allow_tf32`` and ``matmul_precision`` temporarily control matrix products.

    See README.md for weight semantics, dtype conversion and approximation limits.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        whiten: bool = False,
        device=None,
        copy: bool = True,
        batch_size: int | None = None,
        svd_driver: str | None = None,
        lowrank: bool = False,
        lowrank_q: int | None = None,
        lowrank_niter: int = 4,
        lowrank_seed: int | None = None,
        gram: bool = False,
        stats_dtype: torch.dtype | None = None,
        ensure_contiguous: bool = True,
        gram_eps: float = 1e-7,
        allow_tf32: bool | None = None,
        matmul_precision: str | None = None,
        deterministic_flip: bool = True,
    ):
        self.whiten = whiten
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_components = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.svd_driver = svd_driver

        self.lowrank = lowrank
        self.lowrank_q = lowrank_q
        self.lowrank_niter = lowrank_niter
        self.lowrank_seed = lowrank_seed

        self.gram = gram
        self.stats_dtype = stats_dtype
        self.ensure_contiguous = ensure_contiguous
        self.gram_eps = gram_eps

        self.allow_tf32 = allow_tf32
        self.matmul_precision = matmul_precision
        self.deterministic_flip = deterministic_flip

        self._reset_fit_state()
        self._validate_parameters()

    def _reset_fit_state(self):
        """Remove all state learned from earlier calls to fit."""
        learned_attributes = (
            "components_",
            "singular_values_",
            "mean_",
            "var_",
            "explained_variance_",
            "explained_variance_ratio_",
            "noise_variance_",
            "mean_proj_",
            "n_components_",
            "batch_size_",
            "n_features_",
            "n_samples_seen_",
            "weight_sum_",
            "weight_square_sum_",
        )
        for attribute in learned_attributes:
            self.__dict__.pop(attribute, None)

        # Workspace for the augmented matrix; it must not survive a refit on a
        # different shape, device, or dtype.
        self._x_aug_work: torch.Tensor | None = None

    def _validate_parameters(self):
        for name in ("n_components", "batch_size", "lowrank_q"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        if self.svd_driver not in (None, "gesvd", "gesvdj", "gesvda"):
            raise ValueError("Invalid svd_driver.")
        if self.svd_driver is not None and self.device.type != "cuda":
            raise ValueError("svd_driver requires a CUDA device.")
        if self.matmul_precision not in (None, "highest", "high", "medium"):
            raise ValueError("Invalid matmul_precision.")
        if (
            not isinstance(self.lowrank_niter, Integral)
            or isinstance(self.lowrank_niter, bool)
            or self.lowrank_niter < 0
        ):
            raise ValueError("lowrank_niter must be a nonnegative integer.")
        if (
            not isinstance(self.gram_eps, Real)
            or isinstance(self.gram_eps, bool)
            or not math.isfinite(float(self.gram_eps))
            or self.gram_eps <= 0
        ):
            raise ValueError("gram_eps must be finite and strictly positive.")
        if self.stats_dtype not in (None, torch.float32, torch.float64):
            raise ValueError(
                "stats_dtype must be torch.float32, torch.float64, or None."
            )
        if self.lowrank and self.gram:
            raise ValueError(
                "lowrank and gram are mutually exclusive. Set only one to True."
            )
        if (
            self.lowrank_q is not None
            and self.n_components is not None
            and self.lowrank_q < self.n_components
        ):
            raise ValueError("lowrank_q must be >= n_components.")

    @contextlib.contextmanager
    def _matmul_context(self):
        """Temporarily apply the requested matrix multiplication precision."""
        if self.matmul_precision is None and self.allow_tf32 is None:
            yield
            return
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if self.allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            if self.matmul_precision is not None:
                torch.set_float32_matmul_precision(self.matmul_precision)
            yield
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def _svd_fn_full(self, X):
        _, S, Vh = torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)
        return S, Vh

    def _svd_lowrank(self, X):
        q = min(self.lowrank_q or 2 * self.n_components_, min(X.shape))
        if q < self.n_components_:
            raise ValueError("lowrank_q must be >= n_components_.")
        with torch.random.fork_rng(enabled=self.lowrank_seed is not None):
            if self.lowrank_seed is not None:
                torch.manual_seed(self.lowrank_seed)
            _, S, V = torch.svd_lowrank(X, q=q, niter=self.lowrank_niter)
        return S, V.mH

    def _svd_gram(self, X):
        """Return leading singular pairs, or None when full SVD is safer."""
        rows, features = X.shape
        if rows > features:
            return None
        gram = X @ X.mH
        loading = torch.finfo(X.dtype).eps * rows * gram.diagonal().real.abs().max()
        gram.diagonal().add_(loading.clamp_min(self.gram_eps**2))
        try:
            _, vectors = torch.linalg.eigh(gram)
        except torch.linalg.LinAlgError:
            return None
        # Recover values from the original matrix, excluding diagonal loading.
        projected = vectors[:, -self.n_components_ :].flip(1).mH @ X
        S = torch.linalg.vector_norm(projected, dim=1)
        if not torch.isfinite(S).all() or (S <= self.gram_eps).any():
            return None
        return S, projected / S[:, None]

    def _decompose(self, X):
        """Return retained singular values, component rows, and discarded energy."""
        approximate = False
        with self._matmul_context():
            if self.lowrank:
                result = self._svd_lowrank(X)
                approximate = True
            elif self.gram:
                result = self._svd_gram(X)
                approximate = result is not None
            else:
                result = None
            S, Vh = self._svd_fn_full(X) if result is None else result
        k = self.n_components_
        residual = (
            (X.abs().square().sum() - S[:k].square().sum()).clamp_min(0)
            if approximate
            else S[k:].square().sum()
        )
        S, Vh = S[:k].clone(), Vh[:k].clone()
        if self.deterministic_flip:
            pivots = Vh.gather(1, Vh.abs().argmax(dim=1, keepdim=True))
            # Fix complex phase as well as real sign; left vectors are unused.
            Vh *= torch.sgn(pivots).conj()
        return S, Vh, residual

    @staticmethod
    def _input_tensor(X):
        X = torch.as_tensor(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError("X must be a nonempty 2D input.")
        return X

    def _prepare(self, X, dtype=None, check_input=True):
        X = self._input_tensor(X)
        valid_dtypes = (torch.float32, torch.float64, torch.complex64, torch.complex128)
        if dtype is None:
            if hasattr(self, "components_"):
                dtype = self.components_.dtype
            else:
                dtype = X.dtype if X.dtype in valid_dtypes else torch.float32
        if dtype not in valid_dtypes:
            raise ValueError(
                "dtype must be float32, float64, complex64, or complex128."
            )
        if X.is_complex() and not dtype.is_complex:
            raise ValueError("Cannot cast complex input to a real dtype.")
        if hasattr(self, "components_"):
            if X.shape[1] != self.n_features_:
                raise ValueError("Number of features does not match the fitted model.")
            if dtype != self.components_.dtype:
                raise ValueError(
                    "dtype must match the fitted model; call fit to change it."
                )
        X = X.to(device=self.device, dtype=dtype)
        if check_input and not bool(torch.isfinite(X).all()):
            raise ValueError("X must contain only finite values.")
        return X.contiguous() if self.ensure_contiguous else X

    @staticmethod
    def _weights(weights, n):
        if weights is None:
            return None
        w = torch.as_tensor(weights)
        if w.is_complex() or w.ndim > 2 or (w.ndim == 2 and w.shape[1] != 1):
            raise ValueError(
                "Weights must be real scalars, vectors, or column vectors."
            )
        w = w.reshape(-1)
        if w.numel() == 1:
            w = w.expand(n)
        if w.numel() != n:
            raise ValueError("Weights must have one entry per sample.")
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("Weights must be finite and nonnegative.")
        return w

    def _augmented_matrix(self, X, weights, batch_mean, mass):
        """Combine retained scatter, the centered batch, and the mean correction."""
        center = batch_mean.to(X.dtype)
        scale = weights.sqrt().to(X.real.dtype)[:, None]
        if not hasattr(self, "components_"):
            centered = X - center if self.copy else X.sub_(center)
            return centered * scale

        k = self.n_components_
        rows, features = X.shape
        size = k + rows + 1
        work = self._x_aug_work
        if work is None or work.shape[0] < size:
            # Feature count and dtype are fixed during partial_fit; fit clears this.
            self._x_aug_work = X.new_empty(size, features)
        matrix = self._x_aug_work[:size]
        torch.mul(self.components_, self.singular_values_[:, None], out=matrix[:k])
        torch.sub(X, center, out=matrix[k:-1])
        matrix[k:-1].mul_(scale)
        correction = math.sqrt(self.weight_sum_ * mass / (self.weight_sum_ + mass))
        matrix[-1].copy_((self.mean_ - batch_mean) * correction)
        return matrix

    def _merge_statistics(self, mean, scatter, mass):
        """Merge population scatter using the difference between batch means."""
        if not hasattr(self, "components_"):
            return mean, scatter
        previous_mean = self.mean_.to(mean.dtype)
        previous_scatter = self.var_.to(scatter.dtype) * self.weight_sum_
        total = self.weight_sum_ + mass
        delta = mean - previous_mean
        mean = previous_mean + delta * (mass / total)
        correction = delta.abs().square() * (self.weight_sum_ * mass / total)
        return mean, previous_scatter + scatter + correction

    @torch.inference_mode()
    def fit(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Reset and fit. CPU data and weights are transferred one batch at a time.

        Weights are relative importance weights; see README for variance normalization.
        dtype=None preserves supported input dtypes. check_input=False skips finite-X
        checking only; shape, weight and dtype invariants are always checked.
        """
        self._reset_fit_state()
        self._validate_parameters()
        X = self._input_tensor(X)
        weights = self._weights(X_weights, X.shape[0])
        if weights is not None and not bool((weights > 0).any()):
            raise ValueError("At least one sample must have positive weight.")
        self.batch_size_ = self.batch_size or 5 * X.shape[1]
        if self.gram and self.batch_size is None and self.n_components is not None:
            self.batch_size_ = max(
                self.n_components, X.shape[1] - self.n_components - 1
            )
        if self.n_components is not None and self.batch_size_ < self.n_components:
            raise ValueError("batch_size must be >= n_components.")
        for start in range(0, X.shape[0], self.batch_size_):
            batch = slice(start, start + self.batch_size_)
            self.partial_fit(
                X[batch],
                check_input=check_input,
                X_weights=None if weights is None else weights[batch],
                dtype=dtype,
            )
        return self

    @torch.inference_mode()
    def partial_fit(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Update from one batch. A zero-total-weight batch is a no-op.

        The first positive-weight batch needs at least n_components rows.
        Subsequent batches may be smaller. Existing state fixes feature count and dtype.
        """
        self._validate_parameters()
        X = self._prepare(X, dtype, check_input)
        rows, features = X.shape
        w = self._weights(X_weights, rows)
        real_dtype = self.stats_dtype or X.real.dtype
        if self.stats_dtype is None and not X.is_cuda:
            real_dtype = torch.float64
        stat_dtype = (
            {torch.float32: torch.complex64, torch.float64: torch.complex128}[
                real_dtype
            ]
            if X.is_complex()
            else real_dtype
        )
        xs = X.to(stat_dtype)
        w = (
            torch.ones(rows, device=X.device, dtype=real_dtype)
            if w is None
            else w.to(device=X.device, dtype=real_dtype)
        )
        mass = w.sum().item()
        square_mass = w.square().sum().item()
        if not math.isfinite(mass) or not math.isfinite(square_mass):
            raise ValueError("Weight totals overflow the statistics dtype.")
        if mass == 0:
            return self
        first = not hasattr(self, "components_")
        k = (self.n_components or min(rows, features)) if first else self.n_components_
        if k > features or (first and k > rows):
            raise ValueError(
                "n_components must not exceed features or first batch rows."
            )
        batch_mean = (xs * w[:, None]).sum(0) / mass
        batch_ss = ((xs - batch_mean).abs().square() * w[:, None]).sum(0)
        old_mass = 0 if first else self.weight_sum_
        total = old_mass + mass
        total_square = square_mass + (0 if first else self.weight_square_sum_)
        mean, ss = self._merge_statistics(batch_mean, batch_ss, mass)
        matrix = self._augmented_matrix(X, w, batch_mean, mass)
        self.n_components_ = k
        S, components, residual = self._decompose(matrix)
        # Reliability-weighted unbiased covariance; equals n - 1 for unit weights.
        dof = max(0.0, total - total_square / total)
        S2 = S.square()
        variance = S2 / dof if dof > 0 else torch.zeros_like(S2)
        energy = ss.sum()
        ratio = S2 / energy if energy > 0 else torch.zeros_like(S2)
        self.components_ = components
        self.singular_values_ = S
        self.mean_ = mean
        self.var_ = ss / total
        self.n_features_ = features
        self.n_samples_seen_ = rows + (0 if first else self.n_samples_seen_)
        self.weight_sum_ = total
        self.weight_square_sum_ = total_square
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = ratio
        self.mean_proj_ = self.mean_.to(X.dtype) @ self.components_.mH
        discarded = min(matrix.shape) - k
        self.noise_variance_ = (
            residual / (dof * discarded)
            if discarded > 0 and dof > 0
            else S.new_zeros(())
        )
        return self

    @torch.inference_mode()
    def transform(self, X, check_input=True, *, dtype=None):
        """Project onto learned components, optionally whitening. Input is preserved."""
        if not hasattr(self, "components_"):
            raise ValueError("Model must be fitted before transforming data.")
        X = self._prepare(X, dtype, check_input)
        with self._matmul_context():
            result = X @ self.components_.mH - self.mean_proj_
        if self.whiten:
            scale = self.explained_variance_.sqrt()
            result = result / scale.clamp_min(torch.finfo(scale.dtype).eps)
        return result

    def fit_transform(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Fit and project the original data, including when copy=False."""
        X = self._input_tensor(X)
        original = X.clone() if not self.copy else X
        self.fit(X, check_input=check_input, X_weights=X_weights, dtype=dtype)
        return self.transform(original, check_input=check_input, dtype=dtype)


# ---------------------------------------------------------------------------
# Cache serialization used by tools/pca.py. The upstream class keeps learned
# statistics as plain attributes; these helpers snapshot/restore them so the
# artifact can move between CPU/GPU and be extended across tasks.
# ---------------------------------------------------------------------------

_SERIALIZED = (
    "components_",
    "singular_values_",
    "mean_",
    "var_",
    "explained_variance_",
    "explained_variance_ratio_",
    "noise_variance_",
    "mean_proj_",
    "n_components_",
    "batch_size_",
    "n_features_",
    "n_samples_seen_",
    "weight_sum_",
    "weight_square_sum_",
)


def _state_methods(cls):
    def to(self, device):
        self.device = torch.device(device)
        for name in _SERIALIZED:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))
        return self

    def state_dict(self):
        state = {}
        for name in _SERIALIZED:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            state[name] = value
        return state

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        return self.to(self.device)

    cls.to = to
    cls.state_dict = state_dict
    cls.load_state_dict = load_state_dict
    return cls


IncrementalPCAonGPU = _state_methods(IncrementalPCAonGPU)


class UncenteredPCAonGPU:
    """Accumulate an uncentered input Gram matrix and expose its eigensystem.

    ``explained_variance_`` holds the eigenvalues of the non-centered second moment
    ``E[x x^T]`` (the Gram matrix divided by the number of sampled positions), so
    ``raw`` ESPReg weights are the same quantity as the drift identity uses.
    """

    def __init__(self, n_components, device="cuda"):
        self.n_components = n_components
        self.device = torch.device(device if "cuda" not in device or torch.cuda.is_available() else "cpu")
        self.gram_matrix = None
        self.n_samples_seen_ = 0
        self.components_ = None
        self.explained_variance_ = None

    def to(self, device):
        """Move all tensor state to the given device and update self.device."""
        self.device = torch.device(device)
        for attr in ("gram_matrix", "components_", "explained_variance_"):
            value = getattr(self, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
        return self

    def state_dict(self):
        """Return a CPU-serializable snapshot of the operator state.

        The Gram matrix is omitted: it is exactly reconstructible from the stored
        eigen-decomposition and would otherwise double the artifact size. Tensor
        attributes are detached and moved to CPU so the snapshot can be loaded on
        any device; the device itself is not serialized.
        """
        state = {}
        for key, value in self.__dict__.items():
            if key in {"device", "gram_matrix"}:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            state[key] = value
        return state

    def load_state_dict(self, state):
        """Restore a snapshot produced by state_dict(), moving tensors to this operator's device."""
        for key, value in state.items():
            setattr(self, key, value)
        self.to(self.device)
        if self.gram_matrix is None:
            if self.components_ is None or self.explained_variance_ is None:
                raise RuntimeError("Uncentered PCA state has neither a Gram matrix nor an eigen-decomposition")
            scale = self.explained_variance_ * self.n_samples_seen_
            self.gram_matrix = (self.components_ * scale.unsqueeze(1)).T @ self.components_
        return self

    def _validate_data(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.n_components:
            raise ValueError(
                f"Expected samples shaped (N, {self.n_components}), got {tuple(x.shape)}"
            )
        return x.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def partial_fit(self, x, check_input=True):
        """Accumulate X^T X for samples shaped (N, input_dim)."""
        x = self._validate_data(x) if check_input else x.to(self.device, dtype=torch.float32)
        batch_gram = x.T @ x
        self.gram_matrix = batch_gram if self.gram_matrix is None else self.gram_matrix + batch_gram
        self.n_samples_seen_ += x.shape[0]
        self.components_ = None
        self.explained_variance_ = None
        return self

    def fit(self, x, check_input=True):
        self.gram_matrix = None
        self.n_samples_seen_ = 0
        self.partial_fit(x, check_input=check_input)
        self.finalize()
        return self

    @torch.no_grad()
    def finalize(self):
        """Compute the reference SVD and normalize the spectrum to a second moment."""
        if self.gram_matrix is None or self.n_samples_seen_ == 0:
            raise RuntimeError("Cannot finalize an empty uncentered covariance")
        _, singular_values, right_vectors = torch.svd(self.gram_matrix, some=False)
        self.explained_variance_ = singular_values / self.n_samples_seen_
        self.components_ = right_vectors.T.contiguous()
        return self


class ExactMomentPCAonGPU:
    """Collect feature vectors and compute their exact non-centered second moment.

    Vectors are stored on CPU; the Gram matrix is accumulated on the GPU in row
    chunks during ``finalize``, so peak GPU memory is O(chunk_size * D + D^2)
    instead of O(N * D). The spectrum is the exact E[x x^T] of the collected
    vectors (no batch averaging).
    """

    def __init__(self, n_components, device="cuda", chunk_size=16384):
        self.n_components = n_components
        self.device = torch.device(device if "cuda" not in device or torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.vectors = []
        self.n_samples_seen_ = 0
        self.gram_matrix = None
        self.components_ = None
        self.explained_variance_ = None

    def to(self, device):
        """Move the learned Gram matrix to the given device."""
        self.device = torch.device(device)
        if isinstance(self.gram_matrix, torch.Tensor):
            self.gram_matrix = self.gram_matrix.to(self.device)
        return self

    def state_dict(self):
        """Return a CPU-serializable snapshot (the Gram is reconstructed on load)."""
        return {
            "n_components": self.n_components,
            "n_samples_seen_": self.n_samples_seen_,
            "components_": None if self.components_ is None else self.components_.detach().cpu(),
            "explained_variance_": (
                None if self.explained_variance_ is None else self.explained_variance_.detach().cpu()
            ),
        }

    def load_state_dict(self, state):
        """Restore a snapshot and rebuild the historical Gram matrix."""
        for key, value in state.items():
            setattr(self, key, value)
        if self.components_ is not None and self.explained_variance_ is not None and self.gram_matrix is None:
            scale = self.explained_variance_.to(self.device) * self.n_samples_seen_
            components = self.components_.to(self.device)
            self.gram_matrix = (components * scale.unsqueeze(1)).T @ components
        return self

    def _validate_data(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.n_components:
            raise ValueError(f"Expected samples shaped (N, {self.n_components}), got {tuple(x.shape)}")
        return x

    @torch.no_grad()
    def partial_fit(self, x, check_input=True):
        """Store one batch of samples on CPU for the exact final decomposition."""
        x = self._validate_data(x) if check_input else x
        self.vectors.append(x.detach().to("cpu", dtype=torch.float32))
        self.n_samples_seen_ += x.shape[0]
        self.components_ = None
        self.explained_variance_ = None
        return self

    @torch.no_grad()
    def finalize(self):
        """Accumulate the Gram matrix in row chunks and decompose it."""
        if self.n_samples_seen_ == 0:
            raise RuntimeError("Cannot finalize an exact-moment PCA without samples")
        gram = self.gram_matrix
        if gram is None:
            gram = torch.zeros(self.n_components, self.n_components, device=self.device)
        for chunk in self.vectors:
            for start in range(0, chunk.shape[0], self.chunk_size):
                rows = chunk[start : start + self.chunk_size].to(self.device)
                gram = gram + rows.T @ rows
        self.gram_matrix = gram
        values, vectors = torch.linalg.eigh(gram)
        values = values.flip(0).clamp_min(0)
        vectors = vectors.flip(1)
        self.explained_variance_ = values / self.n_samples_seen_
        self.components_ = vectors.T.contiguous()
        self.vectors.clear()
        return self
