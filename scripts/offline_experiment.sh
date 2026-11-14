#!/bin/bash
# Offline incremental-experiment driver for one run directory.
#
# Reuses an already trained base-stage model (task 1), trains only the incremental stages of
# one method, evaluates every stage (per-task and cumulative, class-aware and class-agnostic),
# measures the patch-prototype PCA reconstruction fidelity on every stage sample set, and
# writes a single plain-text summary of the whole run.
#
# The driver composes the unified entry points (scripts/train.sh, scripts/eval.sh) and the
# standalone tools; it contains no experiment logic of its own.
#
# Usage (run from anywhere; all paths are repo-relative):
#   bash scripts/offline_experiment.sh \
#     --tasks data/VOC_5+5+5+5/task_1_cls_5/dataset.yaml \
#             data/VOC_5+5+5+5/task_2_cls_5/dataset.yaml \
#             data/VOC_5+5+5+5/task_3_cls_5/dataset.yaml \
#             data/VOC_5+5+5+5/task_4_cls_5/dataset.yaml \
#     --base-model runs/<baseline-run>/task-1/best.pt \
#     --output runs/offline_voc_5+5+5+5 \
#     --summary results/offline_voc_5+5+5+5.txt
#
#   # ... or let the driver slice the split from data/VOC-YOLO first:
#   bash scripts/offline_experiment.sh --dataset voc --split 5_5_5_5 \
#     --output runs/offline_voc_5+5+5+5 --summary results/offline_voc_5+5+5+5.txt
#
# The base model is linked as <output>/task-1/best.pt; its patch-prototype and ESPReg PCA
# artifacts are rebuilt from it on the first incremental stage, so a machine that only holds
# the base checkpoint needs no other artifact.
#
# Options:
#   --tasks YAML [YAML ...]   Incremental task sequence, one yaml per task (first = base task).
#                             Omit it together with --dataset/--split to build the split on the fly.
#   --dataset FAMILY          CIL dataset family to slice from its source yaml (voc, voc-tiny, ...)
#   --split N_N_...           Class counts per task (e.g. 5_5_5_5); with --dataset this creates
#                             data/<TAG>/ via scripts/create.sh when the task yamls are missing
#   --overwrite-dataset       Rebuild the split even when the task yamls already exist
#   --base-model CKPT         Trained task-1 checkpoint to reuse (default: the naive baseline's
#                             task-1 checkpoint of the same tag, when --dataset/--split are given)
#   --base-data YAML          Base-task dataset yaml (default: the first --tasks yaml)
#   --output DIR              Run directory (required)
#   --summary FILE            Text summary path (default: <output>/summary.txt)
#   --model SPEC              Model family/size (default: yolo26m)
#   --weights FILE            Init weights of the (reused) base stage (default: yolo26m-cls.pt)
#   --method STR              '+'-joined method components (default: prad+prd+espreg)
#   --epochs N --batch N --device D --workers N
#   --train-base              Train the base stage when --base-model is missing or not found
#   --prepare-only            Only build the split and resolve the plan, then exit (pre-flight check)
#   --skip-eval / --skip-report   Skip the evaluation / the PCA reconstruction report
#   --extra-args "..."        Extra flags forwarded to tools/train.py; defaults to the project protocol
#                             "--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5"
#
# Env: the method knobs of scripts/model_adapters/ultralytics.sh still apply; this driver
#      exports the defaults of the prototype method unless they are already set.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The tools under tools/ import the `ultralytics` package; put the repository root on the
# import path so the run works without an editable install of the package.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_YAMLS=()
DATASET=""
SPLIT=""
OVERWRITE_DATASET=0
BASE_MODEL=""
BASE_DATA=""
OUTPUT_DIR=""
SUMMARY=""
MODEL_SPEC="yolo26m"
WEIGHTS="yolo26m-cls.pt"
METHOD="prad+prd+espreg"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
SKIP_EVAL=0
SKIP_REPORT=0
PREPARE_ONLY=0
TRAIN_BASE=0
EXTRA_ARGS="${EXTRA_ARGS:---optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5}"

die() { echo "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        --tasks)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASK_YAMLS+=("$1")
                shift
            done
            ;;
        --dataset) DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --split) SPLIT="${2:?--split needs a value}"; shift 2 ;;
        --overwrite-dataset) OVERWRITE_DATASET=1; shift ;;
        --base-model) BASE_MODEL="${2:?--base-model needs a value}"; shift 2 ;;
        --base-data) BASE_DATA="${2:?--base-data needs a value}"; shift 2 ;;
        --output) OUTPUT_DIR="${2:?--output needs a value}"; shift 2 ;;
        --summary) SUMMARY="${2:?--summary needs a value}"; shift 2 ;;
        --model) MODEL_SPEC="${2:?--model needs a value}"; shift 2 ;;
        --weights) WEIGHTS="${2:?--weights needs a value}"; shift 2 ;;
        --method) METHOD="${2:?--method needs a value}"; shift 2 ;;
        --epochs) EPOCHS="${2:?--epochs needs a value}"; shift 2 ;;
        --batch) BATCH_SIZE="${2:?--batch needs a value}"; shift 2 ;;
        --workers) WORKERS="${2:?--workers needs a value}"; shift 2 ;;
        --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
        --extra-args) EXTRA_ARGS="${2:-}"; shift 2 ;;
        --train-base) TRAIN_BASE=1; shift ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --skip-report) SKIP_REPORT=1; shift ;;
        --*) die "Unknown option: $1" ;;
        *) die "Unexpected positional argument: $1" ;;
    esac
done

# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

if [[ -n "$DATASET" || -n "$SPLIT" ]]; then
    [[ -n "$DATASET" && -n "$SPLIT" ]] || die "--dataset and --split must be given together"
    experiment_load_dataset "$DATASET" "$SPLIT"
    [[ "$INCREMENTAL_SETTING" == "cil" ]] || die "Dataset '$DATASET' has no create step; pass --tasks instead"
    [[ -n "$SOURCE_CFG" ]] || die "No source yaml configured for dataset '$DATASET'"
    [[ -f "$SOURCE_CFG" ]] || die "Source dataset yaml not found: $SOURCE_CFG"

    # One yaml per incremental stage, named by the split counts (task 1 keeps its own yaml).
    derived=()
    for index in "${!CLASS_COUNTS[@]}"; do
        stage=$((index + 1))
        derived+=("${DATA_ROOT}/task_${stage}_cls_${CLASS_COUNTS[$index]}/dataset.yaml")
    done
    if (( OVERWRITE_DATASET == 1 )) || [[ ! -f "${derived[0]}" ]]; then
        echo "Creating the '$DATASET $SPLIT' split from $SOURCE_CFG"
        bash scripts/create.sh --dataset "$DATASET" --split "$SPLIT"
    fi
    if (( ${#TASK_YAMLS[@]} == 0 )); then
        TASK_YAMLS=("${derived[@]}")
    fi
    if [[ -z "$BASE_MODEL" ]]; then
        BASE_MODEL="runs/${MODEL_SPEC}_${DATA_TAG}_pretrained-from-${WEIGHTS%.pt}_naive/task-1/best.pt"
    fi
fi

(( ${#TASK_YAMLS[@]} >= 2 )) || die "Need --tasks (or --dataset/--split) with the base task and at least one incremental task"
for yaml in "${TASK_YAMLS[@]}"; do
    [[ -f "$yaml" ]] || die "Task yaml not found: $yaml (create the split with scripts/create.sh --dataset/--split)"
done
[[ -n "$OUTPUT_DIR" ]] || die "Need --output (the run directory)"
if [[ -z "$BASE_MODEL" ]]; then
    die "Need --base-model (a trained task-1 checkpoint) or --train-base"
fi
if [[ ! -f "$BASE_MODEL" ]]; then
    (( TRAIN_BASE == 1 )) || die "Base model not found: $BASE_MODEL (pass --train-base to train it first)"
    echo "Base model not found; training the base stage on ${TASK_YAMLS[0]}"
    EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" WORKERS="$WORKERS" \
        bash scripts/train.sh --tasks "${TASK_YAMLS[0]}" --model "$MODEL_SPEC" --weights "$WEIGHTS" \
            --method naive --output "$OUTPUT_DIR" -- $EXTRA_ARGS
    BASE_MODEL="$OUTPUT_DIR/task-1/best.pt"
    [[ -f "$BASE_MODEL" ]] || die "Base-stage training produced no checkpoint at $BASE_MODEL"
fi
BASE_DATA="${BASE_DATA:-${TASK_YAMLS[0]}}"
[[ -f "$BASE_DATA" ]] || die "Base dataset yaml not found: $BASE_DATA"
SUMMARY="${SUMMARY:-${OUTPUT_DIR}/summary.txt}"

# Defaults of the prototype method; an explicit environment value always wins.
export ESPREG_WEIGHT_MODE="${ESPREG_WEIGHT_MODE:-raw}"
export ESPREG_FORM="${ESPREG_FORM:-frobenius}"
export ESPREG_SCOPE="${ESPREG_SCOPE:-all}"
export ESPREG_LOSS_WEIGHT="${ESPREG_LOSS_WEIGHT:-0.01}"
export PRAD_PROTO_THRESHOLD="${PRAD_PROTO_THRESHOLD:-0.3}"
export PRAD_CLS_LOSS_WEIGHT="${PRAD_CLS_LOSS_WEIGHT:-30.0}"
export PRAD_REG_LOSS_WEIGHT="${PRAD_REG_LOSS_WEIGHT:-1.5}"
export PRD_LOSS_WEIGHT="${PRD_LOSS_WEIGHT:-30.0}"
export PRD_REG_LOSS_WEIGHT="${PRD_REG_LOSS_WEIGHT:-5.0}"
export PATCH_PCA_DIMS="${PATCH_PCA_DIMS:-128}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/task-1"
ln -sfn "$(readlink -f "$BASE_MODEL")" "$OUTPUT_DIR/task-1/best.pt"

# Cumulative eval sequence: every "task_1-<k>" dataset of the same family, ordered by task count.
candidate_dir="$(dirname "${TASK_YAMLS[0]}")"
family_dir="$(dirname "$candidate_dir")"
CUMULATIVE_YAMLS=("$BASE_DATA")
while IFS= read -r path; do
    CUMULATIVE_YAMLS+=("$path")
done < <(find "$family_dir" -maxdepth 2 -name dataset.yaml -path "*/task_1-*" | sort -V)

echo "=========================================="
echo "Offline incremental experiment"
echo "  tasks   : ${#TASK_YAMLS[@]}  base model: $BASE_MODEL"
echo "  method  : $METHOD"
echo "  output  : $OUTPUT_DIR"
echo "  summary : $SUMMARY"
echo "  epochs  : $EPOCHS  batch=$BATCH_SIZE  device=$DEVICE"
echo "  cumul.  : ${#CUMULATIVE_YAMLS[@]} eval yamls"
echo "  tasks   : ${TASK_YAMLS[*]}"
echo "=========================================="

if (( PREPARE_ONLY == 1 )); then
    echo "prepare-only: split ready, base model linked, no training launched"
    exit 0
fi

START_TASK=2 EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" WORKERS="$WORKERS" \
    bash scripts/train.sh --tasks "${TASK_YAMLS[@]}" --model "$MODEL_SPEC" --weights "$WEIGHTS" \
        --method "$METHOD" --output "$OUTPUT_DIR" -- $EXTRA_ARGS

if [[ "$SKIP_EVAL" == "0" ]]; then
    bash scripts/eval.sh "$OUTPUT_DIR" --tasks "${TASK_YAMLS[@]}" --cumulative "${CUMULATIVE_YAMLS[@]}"
fi

if [[ "$SKIP_REPORT" == "0" ]]; then
    mkdir -p "$OUTPUT_DIR/pca_reconstruction"
    base_model="$OUTPUT_DIR/task-1/best.pt"
    base_artifact="$OUTPUT_DIR/task-1/patch_prototypes.pt"
    if [[ -f "$base_artifact" ]]; then
        : > "$OUTPUT_DIR/pca_reconstruction/similarity.txt"
        for yaml in "${TASK_YAMLS[@]}"; do
            stage="$(basename "$(dirname "$yaml")")"
            python tools/generate_patch_prototypes.py \
                --model "$base_model" --data "$yaml" \
                --load_hist "$base_artifact" --report_only --split val \
                --report_dims 16 32 64 128 --report_similarity --report_samples 512 \
                --max_images 300 --batch_size 8 --workers "$WORKERS" --device "$DEVICE" \
                --report_csv "$OUTPUT_DIR/pca_reconstruction/$stage.csv" \
                >> "$OUTPUT_DIR/pca_reconstruction/report.log" 2>&1
            grep -E "^Level [0-9]+ prototype similarity" "$OUTPUT_DIR/pca_reconstruction/report.log" \
                >> "$OUTPUT_DIR/pca_reconstruction/similarity.txt"
        done
    else
        echo "No patch prototype artifact under $OUTPUT_DIR/task-1; skipping the PCA report" >&2
    fi
fi

python tools/summarize_run.py --run "$OUTPUT_DIR" --output "$SUMMARY"
echo "=========================================="
echo "Summary written to $SUMMARY"
echo "Copy that file back for analysis."
echo "=========================================="
