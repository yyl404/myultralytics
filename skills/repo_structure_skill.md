---
name: myultralytics-repo-structure
description: >-
  Enforces the myultralytics repository layout for scripts, ultralytics
  extensions, tools, and datasets. Use when adding or moving training scripts,
  eval scripts, trainers, models, blocks, tools, datasets, or any paths that
  cross directories in this project.
---

# Repository Structure

Follow these layout rules when creating, moving, or referencing files in this repo.

## Directory roles

| Path | Allowed content |
|------|-----------------|
| `scripts/` | Shell orchestration only (train / create-split / eval). No Python or other implementation code. |
| `ultralytics/` | Code that extends or patches Ultralytics (trainers, models, blocks, etc.). |
| `tools/` | Standalone utilities that do **not** subclass Ultralytics objects. |
| `data/` | Datasets (real dirs or symlinks); one subdirectory per dataset. |
| `skills/` | Skills for the Coding Agent to modify the repo code automatically |

## `scripts/` layout

One entry point per action. Do **not** add per-dataset / per-model / per-method copies.

```text
scripts/
├── train.sh                       # dataset × model × method
├── eval.sh
├── create.sh                      # CIL splits only
├── feature_drift.sh
├── analyze.sh
├── run_incremental.sh
├── lib/experiment.sh              # dataset / model resolution
└── model_adapters/ultralytics.sh
```

Example:

```bash
bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg
bash scripts/eval.sh runs/<run>
bash scripts/create.sh voc 15_5
```

Rules:
- Put **all** launch / scheduling scripts under `scripts/`.
- Do **not** put implementation code under `scripts/`.
- Register a new dataset or model family in `scripts/lib/experiment.sh`, not as a new directory tree.
- Keep eval model-agnostic: `scripts/eval.sh` takes a run directory.

## Code placement

### Ultralytics extensions → `ultralytics/`

If the code reuses or subclasses Ultralytics objects (new trainer / model / block, or patches to existing ones):

- Place it next to the inherited or reused object (same hierarchy level), **or**
- Edit the existing Ultralytics source file in place.

### Standalone utilities → `tools/`

If the code does not reuse Ultralytics class hierarchy, put it under `tools/`. Examples:

- Per-layer feature PCA on an existing model
- Per-layer parameter importance matrices
- Dataset merging / fusion utilities

## Datasets → `data/`

- Store each dataset under `data/<dataset_name>/`.
- Symlinks are allowed.

## Path references

- Cross-directory references **must** use relative paths.
- **Never** hardcode absolute paths.
