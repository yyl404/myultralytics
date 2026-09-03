---
name: myultralytics-scripts-structure
description: >-
  Documents the unified scripts/ experiment entry points (train / eval / create /
  feature_drift / detect / analyze), the three-yaml-sequence experiment contract
  (train / per-task eval / cumulative eval), dataset and model resolution, and
  the incremental task loop. Use when adding a dataset family, model family, IOD
  method, or changing experiment orchestration.
---

# Scripts Structure

`scripts/` holds **shell orchestration only**. Implementation lives in `tools/`
and `ultralytics/`. Entry points `cd` to the repo root, so they can be launched
from anywhere; relative paths like `data/...` and `tools/...` are repo-relative.

Do **not** add per-dataset, per-split, per-model, or per-method launcher copies.
One train/eval/create/feature_drift script covers every combination.

## Layout

```text
scripts/
├── train.sh                       # dataset × model × method, or --tasks yaml sequence
├── eval.sh                        # model-agnostic; manifest → dataset/split → run-name inference
├── create.sh                      # CIL split builder
├── feature_drift.sh               # task-1 feature drift between two ckpts
├── detect.sh                      # per-task inference over a yaml sequence
├── analyze.sh                     # vis / confusion-matrix tools
├── run_incremental.sh             # shared task loop (model-agnostic)
├── libexec/
│   └── experiment.sh              # dataset / model / output resolution
└── model_adapters/
    └── ultralytics.sh             # train + artifact hooks
```

## Yaml sequences (the core contract)

Every entry point ultimately resolves **three yaml sequences**:

| Sequence | Source flag | Role | Default |
|----------|-------------|------|---------|
| `TASK_DATASETS` | `--tasks` | Incremental train data, one yaml per task | from `--dataset/--split` |
| `EVAL_DATASETS` | `--eval-tasks` | Per-task eval data, one yaml per task | the train sequence |
| `CUMULATIVE_DATASETS` | `--cumulative` | Cumulative eval data, one yaml per task | registered CIL splits only |

`--cumulative` is optional: when no cumulative sequence exists, cumulative eval is
skipped (gated on the sequence being non-empty, not on the CIL/TIL label).
Explicit sequences are validated fail-fast: files must exist and counts must
match the task count. A single comma-separated argument also works
(`--tasks a.yaml,b.yaml`). `--tag NAME` overrides the auto-derived `DATA_TAG`.

A yaml-list flag consumes every following non-flag argument, so put the list
first and pass everything after it as `--flags` (e.g.
`--tasks t1.yaml t2.yaml --model yolo26 --method naive`).

The data-split two-level layout is just one way to generate these sequences
(`create.sh` + the registered families in `experiment_load_dataset`).

`train.sh` persists the resolved sequences to the run directory
(`task_yamls.txt`, `eval_yamls.txt`, `cumulative_yamls.txt`, `experiment.meta`),
so `eval.sh runs/<run>` and `detect.sh --run runs/<run>` need no dataset flags.
`experiment.meta` also carries `EVAL_IOU_THRESHOLD` (e.g. coco's 0.75).

## Incremental settings

Two registered settings share `run_incremental.sh`; they differ in how tasks are
defined and whether cumulative eval data exists.

| Setting | What each task adds | Typical data layout | `create.sh` | Cumulative eval |
|---------|---------------------|---------------------|-------------|-----------------|
| **Class-incremental (CIL)** | New classes from one source dataset | `data/<DATASET>_<n1>+<n2>+.../task_*` | Required | Yes (`task_1`, `task_1-2`, …) |
| **Task-incremental (TIL)** | A whole domain / sub-dataset | Pre-bundled dirs under `data/<DATASET>/` | Omit | Optional; omit when none exist |

OdinW-13 is TIL: each subdomain under `data/OdinW-13-yolo/` is one task.
Task order is **lexicographic (C locale)** over subdirectory names.

Head expansion (`tools/expand_model_head.py`) keeps existing class ids and their
order in the detection head, and appends classes not yet known to the model after
them in their dataset order. A class already known to the model keeps its existing
id — overlapping annotations in the new dataset are aligned to that id by class
name (`tools/convert_dataset_class_ids.py`), never duplicated. Task, eval, and
replay datasets are converted into the current model's class-id space by name
before training/eval, so every DDP rank shares the same class space.

Every `best.pt` also carries its incremental history as a module attribute
`incremental_history` (`[{"task": k, "names": [...]}]`, one entry per stage):
task 1 is stamped by `tools/train.py` at save time, later stages are appended by
`expand_model_head.py`. Eval reads the stage class spaces from the checkpoint
itself, never assuming the eval-time task datasets match the training ones.

## Unified commands

Identity knobs: **dataset + split**, or an explicit **`--tasks`** yaml sequence
(with optional `--eval-tasks` / `--cumulative` / `--tag`), plus **model** and
**method**. Flags or positionals are equivalent (with `--tasks`, positionals are
just model and method).

```bash
# Train
bash scripts/train.sh --dataset voc-tiny --split 15_5 --model yolo26 --method pseudo_label+dist+espreg
bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg
bash scripts/train.sh --tasks t1.yaml t2.yaml --eval-tasks e1.yaml e2.yaml \
    --cumulative c1.yaml c2.yaml --model yolo26 --method naive

# Create a CIL split (TIL has no create step)
bash scripts/create.sh voc 15_5
bash scripts/create.sh voc-tiny 15_5

# Eval a finished run (sequences recovered from the run manifest, else inferred
# from the folder name when omitted)
bash scripts/eval.sh runs/yolo26m_VOC-TINY_15+5_pretrained-from-yoloe-26m-seg_pseudo_label+dist+espreg

# Feature drift on task-1 images (registered split or --tasks)
bash scripts/feature_drift.sh voc-tiny 15_5 runs/<run>/task-1/best.pt runs/<run>/task-2/best.pt
bash scripts/feature_drift.sh --tasks t1.yaml t2.yaml --model1 runs/<run>/task-1/best.pt --model2 runs/<run>/task-2/best.pt
```

`train.sh --` passes extra flags to `tools/train.py`.

## Dataset / split resolution (`libexec/experiment.sh`)

Known dataset families (add new ones as a `case` branch in `experiment_load_dataset`):

| `--dataset` | Setting | Data tag | Source yaml | Default epochs |
|-------------|---------|----------|-------------|----------------|
| `voc` | CIL | `VOC_<n1>+<n2>+...` | `data/VOC-YOLO/VOC.yaml` | 100 |
| `voc-tiny` | CIL | `VOC-TINY_<n1>+<n2>+...` | subsample of VOC, then `data/VOC-TINY-YOLO/VOC.yaml` | 10 |
| `coco` | CIL | `COCO_<n1>+<n2>+...` | `data/coco-yolo/coco.yaml` | 12 |
| `odinw-13` | TIL | `OdinW-13-yolo` | pre-bundled; split must be `13` | 100 |

CIL `--split` is underscore-joined class counts and maps to `+` in the data
directory. Any positive integer sequence is allowed without adding a script:

```text
--dataset voc --split 15_5     →  data/VOC_15+5/
--dataset voc --split 10_2_2_2_2_2  →  data/VOC_10+2+2+2+2+2/
--dataset coco --split 70_10   →  data/COCO_70+10/
```

Task yamls (derived, not hardcoded per split):

- Per-task: `data/<TAG>/task_<k>_cls_<nk>/dataset.yaml`
- Cumulative: task 1 reuses the per-task yaml; later tasks use
  `task_1-<k>_cls_<sum>/dataset.yaml`

Registered splits fill all three sequences at once: train = eval = per-task
yamls, cumulative as above. TIL tasks are discovered with `LC_ALL=C` sort of
`data/<TAG>/*/` and `<dir>/data.yaml`, with no cumulative sequence.

COCO eval adds `--iou_threshold 0.75`. Override with `EVAL_IOU_THRESHOLD`.

Prefer repo-relative paths under `data/` (symlink to storage such as
`/hy-tmp/data/...` if needed). Never hardcode absolute paths.

## Model resolution

`--model` is a family, optionally with a size suffix (`yolo26m`, `yolov8x`,
`yoloe-v8l`). `--size n|s|m|l|x` overrides the suffix.

Default size: `m` on `voc-tiny`, `l` for `yoloe-v8`, otherwise `x`.

| Family | Config yaml | Default weights (size m) | Default weights (size x / l) |
|--------|-------------|--------------------------|------------------------------|
| `yolo26` | `yolo26{size}.yaml` | `yoloe-26m-seg.pt` | `yolo26{size}.pt` |
| `yolov8` | `yolov8{size}.yaml` | `yoloe-v8m-seg.pt` | `yolov8{size}-cls.pt` |
| `yoloe-v8` | `yolov8{size}.yaml` | `yoloe-v8{size}-seg.pt` | `yoloe-v8{size}-seg.pt` |

`yoloe-v8` keeps `MODEL_ID=yolov8{size}` so run names stay compatible with
existing checkpoints. `--weights FILE` / `--from-scratch` override init.

### YOLO26 special train hyps

Applied automatically unless `YOLO26_DEFAULT_HYPS=0`:

- **Every yolo26 run:** `--end2end False` (train/val on one2many + NMS).
- **voc-tiny yolo26 additionally:** AdamW, `lr0=0.001`, `warmup_bias_lr=0.0`,
  `mosaic=0.5`, `freeze=10`.

Each of those values is overridable via the matching env var (`END2END`,
`OPTIMIZER`, `LR0`, `WARMUP_BIAS_LR`, `MOSAIC`, `FREEZE`).

## Method string

`--method` is a `+`-joined list of components. The adapter does **not** restrict
combinations: an unimplemented component fails when resolved; an incompatible
one fails when its artifacts are used.

| Component | Effect |
|-----------|--------|
| `naive` | No extra flags (plain fine-tune) |
| `bpf` | BPF trainer (exclusive trainer branch) |
| `pseudo_label` | Pseudo-label baseline |
| `ewc` | + EWC (importance artifact) |
| `l2` | + L2 |
| `dist` | + distillation |
| `espreg` | + EspReg (PCA artifact) |
| `nsgp` | + NSGP (implies EWC + PCA artifacts) |
| `repre` | + RePRE (prototypes artifact) |
| `replay` | + experience replay |

Output dir (override with `--output`):

```text
runs/${MODEL_ID}_${DATA_TAG}_pretrained-from-${WEIGHTS_STEM}_${METHOD}
runs/${MODEL_ID}_${DATA_TAG}_fromscratch_${METHOD}
```

## File roles

| File | Role |
|------|------|
| `train.sh` | Parse identity knobs or `--tasks` → resolve sequences → write manifest → source `run_incremental.sh` |
| `eval.sh` | Per `task-k/best.pt`, convert class ids → `tools/eval.py` → tables |
| `create.sh` | CIL only: optional subsample (voc-tiny) then `create_incremental_dataset.py` |
| `feature_drift.sh` | `tools/feature_drift.py` on `TASK_DATASETS[0]` |
| `detect.sh` | `tools/detect.py` per task yaml, class ids aligned to the model |
| `analyze.sh` | Dispatch to vis / confusion-matrix tools |
| `libexec/experiment.sh` | Dataset, model, output, and yaml-sequence resolution + manifest IO |
| `run_incremental.sh` | Validate adapter + loop tasks |
| `model_adapters/<framework>.sh` | Framework-specific train / artifact hooks |

Eval always writes the individual table; it adds cumulative tables and
`final_cumulative_task_mAP.csv` whenever a cumulative sequence exists, plus the
per-stage mAP matrix sequence from `tools/cumulative_stage_map.py`
(`model_<k>_eval_cumulative_stage_mAP.csv` per model and the combined
`cumulative_stage_mAP_sequence.csv`), which groups cumulative per-class rows by
each checkpoint's own `incremental_history`.

Trainer/validator/predictor intermediates live inside the run dir, never under
`runs/detect/`: entry points absolutize `OUTPUT_DIR` (a relative ultralytics
`--project` would be re-rooted at `runs/detect/`), so training logs land in
`task-<k>/train/` and eval logs in `evaluation_results/model_*_eval_*/val/`.
Reruns clear these intermediate dirs first (skipped when `RESUME_CHECKPOINT` is
set); final artifacts (`best.pt`, CSVs) are overwritten in place.

## Env knobs

| Variable | Meaning | Default |
|----------|---------|---------|
| `EPOCHS` | Per-task epochs | Dataset family default |
| `BATCH_SIZE` / `IMGSZ` / `WORKERS` / `DEVICE` | Train/eval device and loader | 16 / 640 / 8 / 0 |
| `START_TASK` / `END_TASK` | Partial incremental run | 1 / last task |
| `DIST_LOSS_WEIGHT` / `DIST_TOPK` / `ESPREG_LOSS_WEIGHT` | Method weights | adapter defaults (100 / 1 / 100) |
| `YOLO26_DEFAULT_HYPS` | Set `0` to skip yolo26 extras | 1 |
| `TINY_FRACTION` / `SEED` | voc-tiny subsample | 0.25 / 0 |

`DEVICE` may be a multi-GPU list for training (`0,1`). Artifact tools use
`TOOL_DEVICE` (first GPU of `DEVICE` unless overridden).

## `run_incremental.sh` + model adapters

Do not duplicate the task loop. Contract:

1. Caller sets `MODEL_ADAPTER`, `METHOD`, `OUTPUT_DIR`, and `TASK_DATASETS`
   (`train.sh` via `libexec/experiment.sh`).
2. `run_incremental.sh` sources the adapter and requires:
   `model_adapter_validate`, `model_adapter_initialize`,
   `model_adapter_prepare_task`, `model_adapter_train_task`,
   `model_adapter_finalize_task`.
3. Per task it sets `TASK_ID`, `DATASET_PATH`, `TASK_DIR`, `PREVIOUS_TASK_DIR`
   and calls prepare → train → finalize.
4. Optional resume: `START_TASK` (1-based), plus adapter env (`RESUME_CHECKPOINT`, …).

New frameworks: add `scripts/model_adapters/<framework>.sh` implementing the
five functions.

## Checklist: adding a new CIL dataset family

1. Add a `case` in `experiment_load_dataset` (`DATASET_FAMILY`, `DATA_TAG` prefix,
   `SOURCE_CFG`, default epochs).
2. No new train/eval/create scripts. `create.sh voc-style <n>_<m>` just works
   once the family is registered.
3. Keep paths repo-relative.

## Checklist: adding a new TIL / pre-bundled dataset

1. Ensure `data/<DATASET>/` exists; each task is a subdirectory with a yaml.
2. Add a `case` in `experiment_load_dataset` with `INCREMENTAL_SETTING=til`,
   `TASK_YAML_NAME`, and the protocol split id.
3. Eval automatically skips cumulative tables when no cumulative sequence exists.

## Checklist: running an arbitrary incremental protocol

1. No repo change needed: pass `--tasks` (plus optional `--eval-tasks`,
   `--cumulative`, `--tag`) to `train.sh`; sequences are validated and written
   to the run manifest.
2. `eval.sh runs/<run>` / `detect.sh --run runs/<run>` recover the sequences
   from the manifest; no dataset flags needed afterwards.

## Checklist: adding a new method

1. Implement training/artifact behavior in `tools/` / `ultralytics/` (not under `scripts/`).
2. Extend the adapter’s component handling (`method_has` branches in prepare/finalize,
   plus `model_adapter_check_method_components` for the new token).
3. Update `experiment_known_method_components`. Do not add a new `train_*.sh`.

## Checklist: adding a new model family

1. Add a branch in `experiment_parse_model_spec` / `experiment_load_model`
   (yaml name, default weights, optional extra train args).
2. Put model-specific hyps in `experiment_load_model` (see yolo26), not in a
   per-dataset script copy.
