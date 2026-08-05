---
name: pytorch-coding-style
description: >-
  Reusable coding style for PyTorch training/research code: fail-fast error
  policy, tensor shape discipline, helper/class layout, frozen-model and
  no_grad usage, optional loss composition, and DDP-safe patterns. Use when
  writing or reviewing PyTorch modules, trainers, losses, or batch pipelines.
---

# PyTorch Coding Style

Prefer clarity over cleverness. Make tensor contracts, control flow, and failure
modes obvious to the next reader (and to yourself three months later).

## 1. Error handling (classify first)

Default: **fail-fast**. Do NOT catch or soften unexpected errors.
Only use special handling for the two cases below.

### A. Latent / deferred errors → validate early

When a bad state would not fail immediately but would cause a cryptic,
unrelated error later (shape mismatch, missing keys, wrong dtype, invalid path,
unsupported module type, etc.):

- Check at the earliest boundary where the invariant is known.
- Raise a clear exception with debug context (expected vs got, id/path/key/type).
- Do NOT silently coerce, skip, or return `None`.

```python
# Good: fail early with context
head = model.model[-1]
if not isinstance(head, (DetectHead, OBBHead)):
    raise TypeError(f"Unsupported head type: {type(head)}")
```

### B. Per-item failures in large loops → isolate + warn + continue

Only when ALL are true:

1. Processing many independent items (samples, files, batches, indices).
2. One item failure should not abort the whole loop.
3. The raw exception would lack the item index/id needed to debug.

Then:

- `try/except` around **that item only** (narrow scope).
- Log a warning with: index/id, brief reason, and original exception type/message
  (traceback at debug level if useful).
- Skip that item and continue.
- Do NOT wrap the entire loop/pipeline in one broad `try/except`.
- If failure rate is extreme (empty result / too many skips), fail hard at the end.

### C. Everything else → fail-fast

- No bare `except:` / broad `except Exception:` that swallows.
- No “just in case” guards that hide bugs.
- If you must wrap, use `raise ... from e` and keep the original cause.

When unsure which category: choose **C** (fail-fast).

---

## 2. Module layout

- Put **pure helpers** (no trainer state) as module-level functions above the class.
- Prefix internal helpers with `_`; keep public helpers named for their effect
  (`get_*`, `merge_*`, `compute_*`).
- Prefer a small helper over inlining a multi-branch transform in the training loop.
- Optional / cached inputs: accept an optional precomputed value to avoid
  duplicate forward passes (`pred=None` → compute only if missing).

---

## 3. Docstrings and tensor contracts

- Docstrings state **what** and **shape contracts**, not narrative history.
- Annotate tensor shapes in Args/Returns and at non-obvious lines:

```python
# (B, C, H, W), (N, 4), (B, A), list of (num_boxes, 6)
decoded = raw_output[0]  # (bs, 4+nc, num_anchors)
```

- Name intermediates after their role (`gt_boxes_img`, `keep_mask`, `cls_start`),
  not `tmp` / `x2`.
- Prefer keyword args at call sites when arity is high or meanings are easy to swap.

---

## 4. Tensor and device hygiene

- Move inputs to device at the **boundary** (batch entry, loaded checkpoint,
  constructed buffers); keep inner math on already-placed tensors.
- Empty results use explicit empty tensors with correct **shape, dtype, and device**
  — not `None`, not “sometimes missing keys”:

```python
if len(boxes) == 0:
    boxes = torch.empty((0, n_cols), device=ref.device, dtype=ref.dtype)
```

- Keep device/dtype consistent across `cat` / `stack` / `full` siblings.
- For numerically sensitive ops (KL, log, probs): cast to `float32`, clamp with
  a small `eps`, then cast back to the working dtype if needed.
- Guard divisions with `clamp_min(eps)` (or equivalent) when the denominator can
  be zero under valid data.

---

## 5. Autograd, eval mode, and frozen modules

- Frozen reference nets: `deepcopy` (or load) → `.eval()` → `requires_grad_(False)`.
- Inference on frozen nets runs under `torch.no_grad()`.
- If you temporarily switch a submodule to `.eval()` for a loss/probe, **restore**
  `.train()` before returning to the main training path.
- Do not let teacher/reference forwards leak into the student graph unless that
  is an explicit design choice.

---

## 6. Training-loop composition

- Gate optional features with clear config flags; set them up once in setup,
  not ad hoc deep inside the step.
- Compose losses explicitly: compute each term → scale → add to total → append
  a scalar to the logged loss vector. Prefer readable blocks over one mega-expression.
- Pair resource lifecycles: register hooks / buffers at a known start boundary;
  remove / release them at the matching end boundary (e.g. epoch).
- When subclassing or patching a long base method, mark extension blocks so diffs
  stay scannable (short `BEGIN/END` or equivalent comments) — keep markers local
  to the changed region, not file-wide noise.

---

## 7. Control flow and branching

- Branch on **type or format at the boundary** (`isinstance`, format flag), then
  keep each branch linear.
- Treat “no GT / no detections / empty cache entry” as first-class empty-tensor
  paths, not afterthoughts.
- Skip empty optional slots with an explicit condition and `continue`; do not
  invent fake placeholders that later confuse shapes.
- Prefer reusing an already-loaded cache/object over loading it twice when two
  features share the same artifact.

---

## 8. Distributed and wrappers

- Guard logging, validation, and plotting with rank checks (`rank in {-1, 0}` or
  project equivalent).
- When a model may be DDP-wrapped, unwrap before name-based parameter access
  (`model.module` if present).
- Broadcast control flags that must stay synchronized across ranks; do not assume
  side effects on rank 0 are visible elsewhere.

---

## 9. Logging vs exceptions

- Use **info** for expected lifecycle events (freeze layer X, start epoch, …).
- Use **warning** when recovering a surprising but handled state (e.g. forcing
  `requires_grad=True` on a float param that was frozen unexpectedly), or for
  category-B per-item skips.
- Do not replace a hard invariant violation with a log line and continued execution.

---

## 10. What not to do

- Silent `except: pass` / vague `"failed"` prints without traceback or re-raise.
- Broad `try` around a whole epoch/pipeline “for safety”.
- Returning `None` from tensor APIs that callers will index or `cat`.
- Magical shape transforms without comments or named slices.
- Leaving modules in `.eval()` after a temporary switch.
- Catching autograd/shape bugs that should fail immediately in development.
