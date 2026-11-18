# CVPR-27 revision experiments (3 machines)

This directory contains the one-command pipelines for the revision experiments in
`ESPReg-CVPR27/revision_plan.md`, split across three machines with identical
hardware. Every run uses the paper's protocol:

- detector: **YOLOv8m**, backbone initialized from **`yolov8m-cls.pt`**
- 100 epochs per stage, batch 16, **SGD** `lr0=0.01`, momentum `0.937`, weight decay `5e-4`
- one shared **naive stage-1** checkpoint per split; every YOLO row starts from it
- final metric: final model on the **20-class union test set**
  (`Old` = Task_1, `New` = Task_2, `All` = all 20 classes, `Avg = (Old + New) / 2`)

## Experiment matrix

| Machine | Runs | Purpose |
|---|---|---|
| 1 (local) | stage-1 15+5, stage-1 19+1, then `15+5 ESP-YOLO` | produces the shared stage-1 checkpoints; 15+5 main row (CAD top-1) |
| 2 (offline) | `15+5 CAD top-3`, `top-5`, `all` | CAD distillation-channel ablation |
| 3 (offline) | `19+1 ESP-YOLO`, `EWC`, `NSGP`, `NSGP-RePRE`, `BPF` | 19+1 rows and same-backbone baselines |

`15+5 ESP-YOLO` is `pseudo_label+espreg+dist` with `DIST_TOPK=1` and is also the
CAD top-1 row. The total is **9 task-2 runs + 2 stage-1 runs**.

## Requirements

Put the same `yolov8m-cls.pt` at the repository root on all three machines:

```
<repo>/yolov8m-cls.pt
```

The scripts abort with a clear message if it is missing. They do **not** download
it. `YOLOV8M_CLS=/absolute/path/to/yolov8m-cls.pt` overrides the path.

All three machines must run the **same code revision**. These scripts set
`REFERENCE_MODEL_MODE=previous`, which relies on the `get_dist_loss` teacher/student
class-count fix in `ultralytics/engine/anti_forget.py` and the adapter change in
`scripts/model_adapters/ultralytics.sh`. Copy the repository (or apply those two
changes) to the offline machines, not just this directory.

The 15+5 split must already exist under `data/VOC_15+5/`. The 19+1 split is
created automatically on the machines that need it:

```
bash scripts/create.sh voc 19_1
```

The scripts run this once if `data/VOC_19+1/` is missing.

## Stage-1 sharing

Machine 1 trains both stage-1 checkpoints and prints the exact files:

```
runs/_cvpr27_stage1/VOC_15+5/task-1-best.pt
runs/_cvpr27_stage1/VOC_19+1/task-1-best.pt
```

Copy each file to the **same path** on the machines that need it:

```bash
# after the 15+5 stage-1 is printed
scp runs/_cvpr27_stage1/VOC_15+5/task-1-best.pt <machine2>:/root/myultralytics/runs/_cvpr27_stage1/VOC_15+5/task-1-best.pt

# after the 19+1 stage-1 is printed
scp runs/_cvpr27_stage1/VOC_19+1/task-1-best.pt <machine3>:/root/myultralytics/runs/_cvpr27_stage1/VOC_19+1/task-1-best.pt
```

Machine 2 requires the 15+5 file; machine 3 requires the 19+1 file. If the file
is missing, the offline script exits immediately with the exact path; copy it and
rerun the script. The offline scripts never train stage-1 themselves.

## Run

```bash
# machine 1
bash scripts/cvpr27/machine1_local.sh

# machine 2 (copy the directory and the 15+5 stage-1)
bash scripts/cvpr27/machine2_offline.sh

# machine 3 (copy the directory and the 19+1 stage-1)
bash scripts/cvpr27/machine3_offline.sh

# show the queue without running anything
PLAN_ONLY=1 bash scripts/cvpr27/machine2_offline.sh
```

Useful environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | `100` | epochs per stage |
| `BATCH_SIZE` | `16` | batch size |
| `DEVICE` | `0` | GPU id |
| `WORKERS` | `8` | dataloader workers |
| `YOLOV8M_CLS` | `<repo>/yolov8m-cls.pt` | classification checkpoint |
| `FORCE` | `0` | `1` retrains/re-evaluates existing runs |
| `PLAN_ONLY` | `0` | `1` prints the queue and exits |

Each run is resumable: an existing `task-2/best.pt` skips training, and an
existing `evaluation_results/stage_mAP_sequence.csv` skips eval. Set
`FORCE=1` to redo a run.

For a smoke test, `EPOCHS=1` writes the same shared stage-1 path as the real run.
Delete `runs/_cvpr27_stage1/` (or use `FORCE=1` on machine 1) before starting the
real 100-epoch queue.

## Outputs

Each run directory:

```
runs/yolov8m_<split>_pretrained-from-yolov8m-cls_<tag>/
├── task-1/best.pt -> runs/_cvpr27_stage1/.../task-1-best.pt
├── task-2/best.pt
├── evaluation_results/
│   ├── individual_datasets_eval.csv
│   ├── cumulative_datasets_eval.csv
│   └── stage_mAP_sequence.csv
└── summary.txt
```

Per-machine result table:

```
results/cvpr27/machine1.csv
results/cvpr27/machine2.csv
results/cvpr27/machine3.csv
```

Columns: `machine,split,method,tag,dist_topk,seed,old50,new50,all50,avg50,old95,new95,all95,run_dir`.
The `*50` columns are mAP50 in percent, `*95` are mAP50-95 in percent. Copy the
three CSVs back together to assemble the revision tables.

## Implementation notes

- **Common base is Pseudo Label only.** Every row in this bundle uses
  `pseudo_label`; no `lwf` component is added. The old `lwf+ewc*` results and
  scripts under `results/` / `scripts/voc/all/10+10/` are prior 10+10 experiments
  and are not part of this matrix. The paper's existing 15+5
  EWC/NSGP/NSGP-RePRE/BPF rows are not re-run here; for a fully PL-only 15+5
  table they need four new runs.
- **Shared teacher for PL and CAD.** The runs set `REFERENCE_MODEL_MODE=previous`,
  so pseudo-labeling and CAD use the pre-expansion (historical-class) model.
  `ultralytics/engine/anti_forget.py::get_dist_loss` now accepts a teacher with
  fewer classes than the expanded student; `DIST_TOPK=-1` therefore means **all
  historical channels** (15 on 15+5, 19 on 19+1), matching the paper's CAD table,
  not all 20 channels of the expanded head.
- **Baselines share the stage-1 checkpoint.** BPF starts at task-2 from the same
  naive stage-1; it does not run its future-aware stage-1. This follows the paper's
  "all YOLO rows share the same stage-1 checkpoint".
- **CAD ablation.** `top-1` is the ESP-YOLO main row on machine 1; machine 2 adds
  `top-3`, `top-5`, and `all`.
- **Single seed.** The requested matrix is single-seed (`seed=0`) for every row.
- **Artifacts.** EWC importance, NSGP/ESPReg PCA, and RePRE prototypes are generated
  automatically by the adapter from the shared stage-1 when a run starts; they are
  cached under that run's `task-1/`.
