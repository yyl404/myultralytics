---
name: myultralytics-scripts-structure
description: >-
  Documents scripts/ path naming, file roles, and shell content templates for
  incremental-learning experiments (class-incremental and task-incremental).
  Use when adding a dataset split, baseline model config, train/eval/create/
  feature-drift script, or model adapter.
---

# Scripts Structure

`scripts/` holds **shell orchestration only**. Implementation lives in `tools/`
and `ultralytics/`. Always invoke scripts from the **repo root** so relative
paths like `scripts/...`, `data/...`, and `tools/...` resolve.

## Incremental settings

Two settings share the same script layout and `run_incremental.sh` loop; they
differ mainly in how tasks are defined and whether cumulative eval data exists.

| Setting | What each task adds | Typical data layout | `create_*.sh` | Cumulative eval |
|---------|---------------------|---------------------|---------------|-----------------|
| **Class-incremental (CIL)** | New classes from one source dataset | `data/<DATASET>_<n1>+<n2>+.../task_*` | Required | Yes (`task_1`, `task_1-2`, …) |
| **Task-incremental (TIL)** | A whole domain / sub-dataset | Pre-bundled dirs under `data/<DATASET>/` | Omit (already split) | Optional; omit when none exist |

OdinW-13 is TIL: each subdomain under `data/OdinW-13-yolo/` is one task.
Task order is **lexicographic (C locale)** over subdirectory names. There are
**no** stage-cumulative datasets—eval only covers individual tasks seen so far.

## Directory layout

```text
scripts/
├── run_incremental.sh                 # shared task loop (model-agnostic)
├── model_adapters/
│   └── <framework>.sh                 # e.g. ultralytics.sh
└── <dataset>/                         # voc | coco | voc-tiny | odinw-13 | ...
    └── <split>/                       # see Path naming
        ├── create_<dataset>_<split>.sh  # CIL only (omit for pre-bundled TIL)
        ├── eval.sh
        ├── feature_drift.sh
        └── <baseline>/                # e.g. yolov8 | yoloe-v8
            ├── config.sh
            └── train_<method>.sh
```

Example (CIL):

```text
scripts/voc/15_5/
├── create_voc_15_5.sh
├── eval.sh
├── feature_drift.sh
└── yolov8/
    ├── config.sh
    ├── train_naive.sh
    ├── train_pseudo_label.sh
    └── train_pseudo_label+dist+espreg.sh
```

Example (TIL, pre-bundled):

```text
scripts/odinw-13/13/
├── eval.sh                 # no create_*.sh; no cumulative eval
├── feature_drift.sh
├── yolov8/
└── yoloe-v8/
```

## Path naming

| Segment | Convention | Examples |
|---------|------------|----------|
| `<dataset>` | lowercase dataset family; variants use `-` | `voc`, `coco`, `voc-tiny`, `odinw-13` |
| `<split>` (CIL) | per-task class counts joined by `_` | `15_5`, `10_10`, `19_1`, `10_2_2_2_2_2`, `40_40`, `70_10` |
| `<split>` (TIL) | task-count or protocol id (not class counts) | `13` (OdinW-13, 13 lex-ordered tasks) |
| `<baseline>` | detector family name (not a specific size) | `yolov8`, `yoloe-v8` |
| `train_<method>.sh` | method id; compound methods join with `+` | `train_pseudo_label+ewc.sh` |

### Split path vs data directory

**Class-incremental**

- Script path uses `_`: `scripts/voc/10_2_2_2_2_2/`
- Data directory uses `+` and an uppercase family prefix: `data/VOC_10+2+2+2+2+2/`
- Tiny / variant families keep the variant in the data name: `data/VOC-TINY_15+5/`

```text
scripts/<dataset>/<n1>_<n2>_.../  →  data/<DATASET>_<n1>+<n2>+.../
```

where `<DATASET>` is `VOC`, `COCO`, `VOC-TINY`, etc.

**Task-incremental (pre-bundled)**

- Script path: `scripts/<dataset>/<split>/` (e.g. `scripts/odinw-13/13/`)
- Data: one subdirectory per task under `data/<DATASET>/`, each with its own yaml
  (often `data.yaml`, not `dataset.yaml`)
- Task order in `TASK_DATASETS` must match the documented protocol (for OdinW-13:
  lexicographic / `LC_ALL=C` sort of subdirectory names)
- Do **not** invent `+`-joined class-count data dirs or run `create_incremental_dataset.py`

```text
scripts/odinw-13/13/  →  data/OdinW-13-yolo/<TaskName>/data.yaml
```

Prefer repo-relative paths under `data/` (symlink to storage such as
`/hy-tmp/data/...` if needed). Never hardcode absolute paths in scripts.

## File roles

| File | Location | Role |
|------|----------|------|
| `config.sh` | `<baseline>/` | Shared vars for all methods of this dataset/split/baseline |
| `train_<method>.sh` | `<baseline>/` | Thin launcher: source config → set `METHOD` / `OUTPUT_DIR` → source `run_incremental.sh` |
| `create_<dataset>_<split>.sh` | `<split>/` | Build CIL split under `data/`; **omit** for pre-bundled TIL |
| `eval.sh` | `<split>/` | Model-agnostic eval over a finished run directory |
| `feature_drift.sh` | `<split>/` | Compare two checkpoints on task-1 data of this split |
| `run_incremental.sh` | `scripts/` | Validate adapter + loop tasks |
| `model_adapters/<framework>.sh` | `scripts/model_adapters/` | Framework-specific train / artifact hooks |

## Content templates

### `config.sh`

Export (or assign) everything the adapter and orchestrator need. Keep method-agnostic.

Required / expected variables:

```bash
#!/bin/bash

MODEL_ADAPTER="scripts/model_adapters/ultralytics.sh"
MODEL_ID="yolov8x"                 # size/variant used in run naming
MODEL_CONFIG="yolov8x.yaml"
MODEL_WEIGHTS="yolov8x-cls.pt"     # optional; omit for from-scratch
DATASET_FAMILY="voc"               # voc | coco | odinw | ...
TASK_DATASETS=(
    "data/VOC_15+5/task_1_cls_15/dataset.yaml"
    "data/VOC_15+5/task_2_cls_5/dataset.yaml"
)
OUTPUT_PREFIX="runs/yolov8x_VOC_15+5_pretrained-from-yolov8x-cls"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"                  # training device; may be a multi-GPU list, e.g. "0,1"
```

TIL example (`TASK_DATASETS` = lex-ordered OdinW-13 domains):

```bash
DATASET_FAMILY="odinw"
TASK_DATASETS=(
    "data/OdinW-13-yolo/AerialMaritimeDrone/data.yaml"
    "data/OdinW-13-yolo/Aquarium/data.yaml"
    # ... remaining tasks in LC_ALL=C lexicographic order ...
    "data/OdinW-13-yolo/thermalDogsAndPeople/data.yaml"
)
OUTPUT_PREFIX="runs/yolov8x_OdinW-13-yolo_pretrained-from-yolov8x-cls"
```

Rules:
- `TASK_DATASETS` order = task order (`task-1`, `task-2`, …).
- Paths must be repo-relative; never absolute.
- `OUTPUT_PREFIX` should encode model, dataset (and split when applicable), and init scheme; `train_*.sh` appends `_${METHOD}`.
- Override knobs via env defaults (`"${EPOCHS:-100}"`) so callers can change them without editing the file.
- `DEVICE` is passed to training (`tools/train.py`) and may name multiple GPUs (comma-separated, e.g. `0,1`) for DDP. Artifact tools (`compute_importance.py`, `pca.py`, `generate_prototypes.py`) always run single-device on `TOOL_DEVICE`, which the adapter derives as the first entry of `DEVICE` (overridable via env).
- Optional: `TASK_FREEZE_LAYERS`, `EXTRA_TRAIN_ARGS`, method-specific weights (`EWC_LOSS_WEIGHT`, …) if needed beyond adapter defaults.

### `train_<method>.sh`

Keep every method script to this shape; do not inline training logic.

```bash
#!/bin/bash

set -euo pipefail

source "scripts/<dataset>/<split>/<baseline>/config.sh"
METHOD="<method>"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
```

`<method>` must match the adapter’s supported set **and** the filename after `train_`.

Supported Ultralytics methods today:

| `METHOD` / filename suffix | Notes |
|----------------------------|--------|
| `naive` | Fine-tune only |
| `bpf` | BPF trainer |
| `pseudo_label` | Pseudo-label baseline |
| `pseudo_label+ewc` | + EWC |
| `pseudo_label+l2` | + L2 |
| `pseudo_label+espreg` | + EspReg |
| `pseudo_label+dist+espreg` | + distillation + EspReg |
| `pseudo_label+nsgp` | + NSGP |
| `pseudo_label+nsgp+repre` | + NSGP + RePRE |

### `create_<dataset>_<split>.sh`

**CIL only.** Call `tools/create_incremental_dataset.py` with class counts matching the split name.

```bash
#!/bin/bash
set -euo pipefail
python tools/create_incremental_dataset.py \
    --source_cfg data/<SOURCE>/<name>.yaml \
    --output_dir data/<DATASET>_<n1>+<n2>+... \
    --n_classes <n1> <n2> ... \
    --workers "${WORKERS:-8}"
```

Rules:
- Filename: `create_<dataset>_<split>.sh` with the same `<dataset>` and `_`-joined `<split>` as the directory.
- `--n_classes` must equal the split segments in order (`15_5` → `15 5`; `10_2_2_2_2_2` → `10 2 2 2 2 2`).
- `--output_dir` must use the `+` data naming above.
- **Do not** add a create script for pre-bundled TIL datasets (e.g. OdinW-13).

### `eval.sh`

Model-agnostic for the split: take a finished run dir, evaluate each `task-k/best.pt`
on individual task datasets seen so far, optionally on cumulative datasets, then
summarize tables.

```bash
#!/bin/bash
# Usage: bash eval.sh [OUTPUT_DIR]

OUTPUT_DIR="${1:-runs/<default_run>}"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE=0

TASK_DATASETS=( ... )          # same task yamls as config.sh
CUMULATIVE_DATASETS=( ... )    # CIL: task_1, task_1-2, ... ; omit for TIL without cumulatives

# loop: convert class ids → tools/eval.py → generate_eval_tables.py
#      → summarize_cumulative_task_map.py   # only when cumulative CSVs exist
```

Rules:
- Lives at split level (not under `<baseline>/`).
- Hardcode this split’s `TASK_DATASETS` (and `CUMULATIVE_DATASETS` when present); do not source `config.sh`.
- Expect models at `$OUTPUT_DIR/task-$k/best.pt`.
- Write results under `$OUTPUT_DIR/evaluation_results/`.
- **CIL:** evaluate individual + cumulative; call `generate_eval_tables.py` and `summarize_cumulative_task_map.py`.
- **TIL without cumulatives (e.g. OdinW-13):** evaluate only individual tasks; call `generate_eval_tables.py` (it skips the cumulative table when no cumulative CSVs exist); **do not** call `summarize_cumulative_task_map.py`.

### `feature_drift.sh`

Fixed task-1 dataset for the split; checkpoints are CLI args.

```bash
#!/bin/bash
set -euo pipefail

MODEL1="${1:?Pass the task-1 model checkpoint path}"
MODEL2="${2:?Pass the task-2 model checkpoint path}"
SAVE_PATH="${3:-$(dirname "$MODEL2")/feature_drift_task1_to_task2.json}"

python tools/feature_drift.py \
    --data "data/<DATASET>_<split+>/task_1_cls_<n1>/dataset.yaml" \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --save_path "$SAVE_PATH"
```

For TIL, `--data` points at the first task yaml (OdinW-13: `data/OdinW-13-yolo/AerialMaritimeDrone/data.yaml`).

### `run_incremental.sh` + model adapters

Do not duplicate the task loop in per-method scripts. Contract:

1. Caller sets `MODEL_ADAPTER`, `METHOD`, `OUTPUT_DIR`, and `TASK_DATASETS` (usually via `config.sh` + `train_*.sh`).
2. `run_incremental.sh` sources the adapter and requires:
   - `model_adapter_validate`
   - `model_adapter_initialize`
   - `model_adapter_prepare_task`
   - `model_adapter_train_task`
   - `model_adapter_finalize_task`
3. Per task it sets `TASK_ID`, `DATASET_PATH`, `TASK_DIR`, `PREVIOUS_TASK_DIR` and calls prepare → train → finalize.
4. Optional resume: `START_TASK` (1-based), plus adapter-specific env vars (`RESUME_CHECKPOINT`, …).

New frameworks: add `scripts/model_adapters/<framework>.sh` implementing the five functions; point `MODEL_ADAPTER` at it from `config.sh`.

The same loop serves CIL and TIL: only `TASK_DATASETS` contents and eval differ.

## Checklist: adding a new CIL split

1. Create `scripts/<dataset>/<split>/`.
2. Add `create_<dataset>_<split>.sh` → produces `data/<DATASET>_<n1>+.../`.
3. Add `eval.sh` and `feature_drift.sh` with that split’s yaml paths (including cumulatives).
4. Add `<baseline>/config.sh` with matching `TASK_DATASETS` and `OUTPUT_PREFIX`.
5. Add one `train_<method>.sh` per method (template above).
6. Keep all paths relative to the repo root.

## Checklist: adding a new TIL / pre-bundled dataset

1. Ensure `data/<DATASET>/` exists (real dir or symlink); each task is a subdirectory with a yaml.
2. Create `scripts/<dataset>/<split>/` (no `create_*.sh`).
3. List tasks in the protocol order (OdinW-13: lexicographic) in `config.sh` and `eval.sh`.
4. Write `eval.sh` without cumulative blocks / without `summarize_cumulative_task_map.py` when no cumulative data exists.
5. Add `feature_drift.sh` pointing at the first task yaml.
6. Add each `<baseline>/` (`config.sh` + `train_*.sh`) as for CIL.
7. Keep all paths relative to the repo root.

## Checklist: adding a new method

1. Implement training/artifact behavior in `tools/` / `ultralytics/` (not under `scripts/`).
2. Extend the adapter’s `METHOD` case list and prepare/finalize branches.
3. Add `train_<method>.sh` under each split/baseline that should expose it (`METHOD` string = filename suffix).
