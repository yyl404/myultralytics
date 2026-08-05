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

Organize by dataset → class-incremental split → baseline model:

```text
scripts/
└── <dataset>/                    # e.g. voc
    └── <split>/                  # e.g. 15_5
        ├── <baseline>/           # e.g. yolov8
        │   └── train_*.sh        # one script per IOD method
        ├── create_<dataset>_<split>.sh
        └── eval.sh               # model-agnostic evaluation
```

Example:

```text
scripts/
└── voc/
    └── 15_5/
        ├── yolov8/
        │   ├── train_naive.sh
        │   ├── train_pseudo_label.sh
        │   └── ...
        ├── create_voc_15_5.sh
        └── eval.sh
```

Rules:
- Put **all** launch / scheduling scripts under `scripts/`.
- Do **not** put implementation code under `scripts/`.
- Keep eval scripts model-agnostic at the split level (`scripts/<dataset>/<split>/eval.sh`).
- Keep split-creation scripts at the split level (`create_<dataset>_<split>.sh`).

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
