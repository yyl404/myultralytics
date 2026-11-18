---
name: myultralytics-scripts-structure
description: >-
  Documents the unified scripts/ experiment entry points (train / eval / create /
  feature_drift / analyze), the decoupled yaml-sequence contract
  (explicit train sequence; independent eval sequences; full model × dataset
  matrices), dataset and model resolution, the incremental task loop, and the
  scripts/voc/tiny/15+5 demo. Use when adding a dataset family, model family,
  IOD method, or changing experiment orchestration.
---

# Scripts Structure

`scripts/` holds **shell orchestration only**. Implementation lives in `tools/`
and `ultralytics/`. Entry points `cd` to the repo root, so they can be launched
from anywhere; relative paths like `data/...` and `tools/...` are repo-relative.

The IOD pipeline ends at evaluation: there is no predict entry and no
class-agnostic (`object`) eval pass. `eval.sh` writes class-aware cells only.
Dataset class-id alignment happens at load time (see below), never by writing a
converted dataset tree.

Do **not** add per-dataset, per-split, per-model, or per-method launcher copies.
One train/eval/create/feature_drift script covers every combination. The
sanctioned exceptions are the demo under `scripts/voc/tiny/15+5/` (see below),
which only composes the unified entry points and holds its tunables in
`common.sh`, and the fixed CVPR-27 revision bundle under `scripts/cvpr27/`
(one shared driver + three machine queues; it composes the same entry points).

## Layout

```text
scripts/
├── train.sh                       # --tasks yaml sequence × model × method
├── eval.sh                        # run dir × explicit eval yaml sequences
├── create.sh                      # CIL split builder (the ONLY --dataset/--split entry)
├── feature_drift.sh               # task-1 feature drift between two ckpts
├── dataset_similarity.sh          # N x N image-domain feature similarity over a yaml sequence
├── analyze.sh                     # vis / confusion-matrix tools
├── run_incremental.sh             # shared task loop (model-agnostic)
├── offline_experiment.sh          # one-command run: reuse base model -> train -> eval -> PCA report -> txt summary
├── libexec/
│   └── experiment.sh              # dataset / model / output resolution
├── model_adapters/
│   └── ultralytics.sh             # train + artifact hooks
├── voc/tiny/15+5/                 # VOC-TINY 15+5 demo (common.sh + train/eval/similarity/pipeline)
└── cvpr27/                        # fixed 3-machine revision queues (common.sh + machine1/2/3 + README)
```

## The yaml-sequence contract (train / eval decoupled)

Training and evaluation are fully decoupled; both take explicit yaml sequences
and nothing else:

| Entry | Sequence flag | Role |
|-------|---------------|------|
| `train.sh` | `--tasks` (required) | Incremental train data, one yaml per task |
| `eval.sh` | `--tasks` (required) | Per-task eval data, any yaml sequence |
| `eval.sh` | `--cumulative` (optional) | Cumulative eval data, any yaml sequence |

- Training never receives eval datasets. It produces `task-1`, `task-2`, …
  under the run directory and stops there.
- The eval matrix is built strictly from reality: every `task-*/best.pt` found
  under the run directory × every eval yaml given, per-task and cumulative
  sequences each as a full cross product. Train and eval sequences do not have
  to match in order, kind, or length. A cell whose classes are disjoint from
  the model's class space yields an empty per-class CSV and shows as `N/A`.
- A yaml-list flag consumes every following non-flag argument, so put the list
  first and pass everything after it as `--flags` (e.g.
  `--tasks t1.yaml t2.yaml --model yolo26 --method naive`). A single
  comma-separated argument also works (`--tasks a.yaml,b.yaml`).
- Each yaml is evaluated on its `test` split by default, falling
  back to `val` when no `test:` key exists (`--split` overrides).
- `--dataset/--split` (registered dataset families) is accepted **only** by
  `create.sh`, for slicing a full single-stage dataset into class-incremental
  task datasets. It is not a train/eval entry.

Head expansion (`tools/expand_model_head.py`) keeps existing class ids and their
order in the detection head, and appends classes not yet known to the model after
them in their dataset order. A class already known to the model keeps its existing
id — overlapping annotations in the new dataset are aligned to that id by class
name, never duplicated. Task, eval, and replay datasets are aligned to the current
model's class-id space at load time: `tools/make_class_mapped_yaml.py` writes a
tiny YAML with the target names and a `class_id_map`, and
`ClassMappedYOLODataset` remaps labels after the shared label cache is read. Every
DDP rank reads the same YAML and cache, so the class space is identical.

Decode-config consistency: `end2end` / `agnostic_nms` / `max_det` are Python
attributes on the Detect head, not state_dict entries, so any yaml rebuild
(`yolo26*.yaml` defaults to `end2end: True`) would silently revert them. The
pipeline keeps them consistent everywhere: head expansion copies all three from
the source checkpoint's head onto the expanded model; every "yaml rebuild +
weight transfer" path (trainer `setup_model`, the pre-build inside
`Model.train`, the distillation teacher rebuild) inherits them from the source
model in `BaseModel.load` (`ultralytics/nn/tasks.py`), and explicit train args
still win afterwards in `set_model_attributes`; the AntiForget/BPF frozen
teacher and reference models get the current train args applied via
`_apply_train_head_args` (`ultralytics/engine/anti_forget.py`) and a
teacher/student `end2end` mismatch raises immediately with expected vs actual.
Eval uses the checkpoint's pickled attributes unless explicit args are
passed through `--`.

Per-stage aggregation at eval time (`tools/stage_task_map.py`) takes the stage
class spaces from the per-task eval yaml sequence: stage j is the class space
of the j-th `--tasks` eval yaml.

## Unified commands

```bash
# Train (yaml sequence × model × method; --tag overrides run naming)
bash scripts/train.sh --tasks t1.yaml t2.yaml --tag my-exp --model yolo26m --method naive
bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26 --method pseudo_label+dist+espreg

# Create a CIL split (the only place --dataset/--split exists)
bash scripts/create.sh voc 15_5
bash scripts/create.sh voc-tiny 15_5

# Eval a finished run on any yaml sequences (full models × datasets matrix)
bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml --cumulative c1.yaml c2.yaml

# Feature drift on task-1 images
bash scripts/feature_drift.sh --tasks t1.yaml t2.yaml --model1 runs/<run>/task-1/best.pt --model2 runs/<run>/task-2/best.pt

# N x N image-domain feature similarity over a dataset sequence
bash scripts/dataset_similarity.sh --tasks t1.yaml t2.yaml --weights yoloe-26m-seg.pt
```

`train.sh --` passes extra flags to `tools/train.py`; `eval.sh --` forwards to
`tools/eval.py` (e.g. `-- --agnostic_nms True`).

## Demo: VOC-TINY 15+5 (`scripts/voc/tiny/15+5/`)

A minimal end-to-end example and the template for new incremental datasets:

```text
scripts/voc/tiny/15+5/
├── common.sh       # the ONLY file to edit when migrating, and only its first
│                   # block: DATA_ROOT / DATA_TAG plus the three yaml sequences
│                   # TRAIN_YAMLS (incremental train), EVAL_YAMLS (independent
│                   # eval), CUMULATIVE_YAMLS (cumulative eval, () to skip)
├── train.sh        # checks TRAIN_YAMLS exist (hint: create the dataset), then scripts/train.sh
├── eval.sh         # scripts/eval.sh on EVAL_YAMLS + CUMULATIVE_YAMLS
├── similarity.sh   # scripts/dataset_similarity.sh on TRAIN_YAMLS (standalone)
└── pipeline.sh     # train → eval
```

The stage scripts resolve their own directory at runtime (`DEMO_DIR`) and walk
up to the repo root (the directory holding `scripts/train.sh`), so copying the
whole folder to any depth under the repo gives a working demo for a new
dataset. Every demo copy under `scripts/` (voc-tiny / voc, 15+5 / joint)
shares byte-identical stage scripts, and their `common.sh` files differ only
in the first dataset block — migrate by editing just that block. The three
yaml sequences are plain bash arrays read from `common.sh` only (no
environment or command-line override); the train, independent-eval, and
cumulative-eval sequences are fully decoupled and may differ in content,
order, and length. The scalar knobs further down (`MODEL`, `WEIGHTS`,
`METHOD`, `RUN_DIR`, decode knobs, `EXTRA_*_ARGS`,
`EPOCHS`, `DEVICE`) can
still be overridden per launch with an environment variable of the same name,
e.g. `EPOCHS=1 RUN_DIR=runs/smoke bash <dir>/pipeline.sh`.

Conventions demonstrated: `yolo26m` + `yoloe-26m-seg.pt`; decode modes
(`end2end` / `agnostic_nms` / `max_det`) are prominent knobs in `common.sh`
(`END2END` / `AGNOSTIC_NMS` / `MAX_DET`, defaults `False` / `True` / `300`)
forwarded to train/eval via `DECODE_ARGS` in every `EXTRA_*_ARGS`;
small-data hyps (AdamW, lr0, mosaic, freeze) passed explicitly after `--` via
`EXTRA_TRAIN_ARGS` in `common.sh`; eval on the `test` split (`val`
fallback) in EVAL_YAMLS then CUMULATIVE_YAMLS order; `similarity.sh` runs the
N x N dataset image-domain feature similarity on TRAIN_YAMLS with the
pretrained backbone (`SIMILARITY_WEIGHTS`, defaults to `WEIGHTS`).

## Dataset / split resolution (`libexec/experiment.sh`, create.sh only)

Known dataset families (add new ones as a `case` branch in `experiment_load_dataset`):

| `--dataset` | Setting | Data tag | Source yaml |
|-------------|---------|----------|-------------|
| `voc` | CIL | `VOC_<n1>+<n2>+...` | `data/VOC-YOLO/VOC.yaml` |
| `voc-tiny` | CIL | `VOC-TINY_<n1>+<n2>+...` | subsample of VOC, then `data/VOC-TINY-YOLO/VOC.yaml` |
| `coco` | CIL | `COCO_<n1>+<n2>+...` | `data/coco-yolo/coco.yaml` |
| `odinw-13` | TIL | `OdinW-13-yolo` | pre-bundled; split must be `13` (no create step) |

CIL `--split` is underscore-joined class counts and maps to `+` in the data
directory. Any positive integer sequence is allowed without adding a script:

```text
--dataset voc --split 15_5     →  data/VOC_15+5/
--dataset voc --split 10_2_2_2_2_2  →  data/VOC_10+2+2+2+2+2/
--dataset coco --split 70_10   →  data/COCO_70+10/
```

`create.sh` produces per-task yamls `task_<k>_cls_<nk>/dataset.yaml` plus
cumulative yamls `task_1-<k>_cls_<sum>/dataset.yaml` (task 1 reuses the
per-task yaml) — pass them explicitly to train/eval afterwards.

COCO eval wants `--iou-threshold 0.75` on `eval.sh` for the extra AP column.

Prefer repo-relative paths under `data/` (symlink to storage such as
`/hy-tmp/data/...` if needed). Never hardcode absolute paths.

## Model resolution

`--model` is a family, optionally with a size suffix (`yolo26m`, `yolov8x`,
`yoloe-v8l`). `--size n|s|m|l|x` overrides the suffix.

Default size: `l` for `yoloe-v8`, otherwise `x`.

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

Dataset-specific fine-tune hyps are NOT auto-applied anymore (train does not
know the dataset). Pass them explicitly after `--`, e.g. the voc-tiny set:
`--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5 --freeze 10`
(see `EXTRA_TRAIN_ARGS` in `scripts/voc/tiny/15+5/common.sh`).

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
| `dist` | + distillation (KL on top-k teacher channels) |
| `lwf` | + YOLO LwF self-distillation (teacher = previous task model) |
| `espreg` | + EspReg (PCA artifact) |
| `nsgp` | + NSGP (implies EWC + PCA artifacts) |
| `repre` | + RePRE (prototypes artifact) |
| `replay` | + experience replay |
| `prad` | + prototype-reliability-aware YOLO LwF (`lwf_mode=prad`, patch-prototype artifact) |
| `prd` | + all-class patch replay distillation (patch-prototype artifact) |

Output dir (override with `--output`):

```text
runs/${MODEL_ID}_${DATA_TAG}_pretrained-from-${WEIGHTS_STEM}_${METHOD}
runs/${MODEL_ID}_${DATA_TAG}_fromscratch_${METHOD}
```

`DATA_TAG` comes from `--tag`, else from the `+`-joined parent directory names
of the train yamls.

## File roles

| File | Role |
|------|------|
| `train.sh` | Parse `--tasks` + model/method → resolve → source `run_incremental.sh` |
| `eval.sh` | Discover `task-*/best.pt` → per (model, yaml) write a class-map YAML → `tools/eval.py` (class-aware) → tables + `tools/stage_task_map.py` |
| `create.sh` | CIL only: optional subsample (voc-tiny) then `create_incremental_dataset.py` |
| `feature_drift.sh` | `tools/feature_drift.py` on `TASK_DATASETS[0]` |
| `dataset_similarity.sh` | `tools/dataset_similarity.py` over the whole `--tasks` sequence |
| `analyze.sh` | Dispatch to vis / confusion-matrix tools |
| `libexec/experiment.sh` | create recipe, model/output resolution, yaml/split/model helpers |
| `run_incremental.sh` | Validate adapter + loop tasks |
| `offline_experiment.sh` | Reuse a trained base model, run the incremental stages, evaluate, measure the patch-PCA reconstruction and write one txt summary |
| `model_adapters/<framework>.sh` | Framework-specific train / artifact hooks (importance, pca_cache, repre prototypes, `patch_prototypes.pt`, replay dataset) |

Eval always writes the individual matrix table; it adds the cumulative matrix
table whenever a cumulative sequence is given. `tools/stage_task_map.py` then
aggregates every `model_<k>_eval_*.csv` into per-stage mAP tables
(`<name>_stage_mAP.csv`) grouped by the class spaces of the per-task eval yaml
sequence, plus the combined `stage_mAP_sequence.csv`.

Trainer/validator intermediates live inside the run dir, never under
`runs/detect/`: entry points absolutize `OUTPUT_DIR` (a relative ultralytics
`--project` would be re-rooted at `runs/detect/`), so training logs land in
`task-<k>/train/` and eval logs in `evaluation_results/model_*_eval_*/`.
Reruns clear these intermediate dirs first (skipped when `RESUME_CHECKPOINT` is
set); final artifacts (`best.pt`, CSVs) are overwritten in place.

## Env knobs

| Variable | Meaning | Default |
|----------|---------|---------|
| `EPOCHS` | Per-task epochs | 100 |
| `BATCH_SIZE` / `IMGSZ` / `WORKERS` / `DEVICE` | Train/eval device and loader | 16 / 640 / 8 / 0 |
| `START_TASK` / `END_TASK` | Partial incremental run | 1 / last task |
| `DIST_LOSS_WEIGHT` / `DIST_TOPK` / `ESPREG_LOSS_WEIGHT` | Method weights | adapter defaults (100 / 1 / 100) |
| `ESPREG_WEIGHT_MODE` / `ESPREG_FORM` / `ESPREG_SCOPE` | ESPReg spectrum (`raw`/`nt`), reduction (`original`/`frobenius`) and layer scope (`all`/`backbone`) | nt / original / all |
| `PRAD_FG_THRESHOLD` / `PRAD_BG_THRESHOLD` / `PRAD_CONF_POWER` | prad teacher confidence thresholds and exponent | 0.5 / 0.1 / 1.0 |
| `PRAD_PROTO_THRESHOLD` / `PRAD_PROTO_TEMPERATURE` | prad prototype-reliability sigmoid center and width | 0.5 / 0.1 |
| `PRAD_BG_WEIGHT` / `PRAD_SPLIT_NORMALIZE` / `PRAD_MAX_FG` | prad background weight, foreground/background normalization, candidate cap | 1.0 / True / 512 |
| `PRAD_CLS_LOSS_WEIGHT` / `PRAD_REG_LOSS_WEIGHT` | prad classification and regression weights | 4.0 / 0.5 |
| `PRD_SAMPLES` / `PRD_STD_SCALE` / `PRD_LOSS_WEIGHT` / `PRD_REG_LOSS_WEIGHT` | prd patch draws, sampling spread, classification and regression weights | 64 / 1.0 / 1.0 / 0.5 |
| `PATCH_PCA_DIMS` / `PATCH_PROTO_MAX_CENTERS` / `PATCH_PROTO_SPLIT_RATIO` | patch-prototype artifact PCA dims, centre budget, DP-Means split ratio | 128 / 8 / 0.2 |
| `PATCH_PROTO_MAX_SAMPLES` / `PATCH_PROTO_SEED` | patches kept per class for clustering, artifact seed | 256 / 0 |
| `YOLO26_DEFAULT_HYPS` | Set `0` to skip yolo26 extras | 1 |
| `TINY_FRACTION` / `SEED` | voc-tiny subsample (create.sh) | 0.1 / 0 |

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

1. Add a `case` in `experiment_load_dataset` (`INCREMENTAL_SETTING=cil`,
   `DATA_TAG` prefix, `SOURCE_CFG`).
2. No new train/eval/create scripts. `create.sh <family> <n>_<m>` just works
   once the family is registered; pass the generated yamls to train/eval.
3. Keep paths repo-relative.

## Checklist: adding a new TIL / pre-bundled dataset

1. Ensure `data/<DATASET>/` exists; each task is a subdirectory with a yaml.
2. No registration needed: pass `--tasks data/<DATASET>/*/data.yaml` (and
   optional `--cumulative`) to train/eval.

## Checklist: running an arbitrary incremental protocol

1. No repo change needed: pass `--tasks` to `train.sh`, and any (possibly
   different) `--tasks` / `--cumulative` sequences to `eval.sh`.
2. Copy `scripts/voc/tiny/15+5/` as a starting point and edit only the dataset
   block of the copied `common.sh`.

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
