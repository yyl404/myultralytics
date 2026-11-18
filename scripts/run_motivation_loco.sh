#!/bin/bash
# Full LoCo (Low Co-occurrence) motivation pipeline on VOC.
#
# LoCo clusters frequently co-occurring classes into the same incremental stage
# and assigns every image to exactly one stage, so no training image is shared
# between stages. This script:
#
#   1. builds VOC_LoCo_10+10 and VOC_LoCo_5+5+5+5 (scripts/create.sh voc-loco);
#   2. runs motivation experiment 1 (stage class counts and sample overlap);
#   3. trains naive and YOLO-LwF incrementally on both protocols and one Joint
#      model on the full 20-class LoCo dataset;
#   4. evaluates every model on the per-stage and cumulative test sets;
#   5. runs motivation experiment 2 (teacher high-confidence errors and their
#      severity on the trained models) and experiment 3 (incremental-model
#      errors on the history stage);
#   6. writes a formatted Markdown summary that also extracts the reference
#      values recorded in experiments.md (tools/loco_report.py).
#
# The run is fully offline: it only reads local data and checkpoints, and it
# disables Ultralytics auto-install/online checks. Training is skipped for tasks
# whose best.pt already exists unless FORCE_TRAIN=1, so a failed run can be
# resumed by rerunning the same command.
#
# Usage:
#   bash scripts/run_motivation_loco.sh
#   EPOCHS=1 DEVICE=0 bash scripts/run_motivation_loco.sh
#   FORCE_TRAIN=1 bash scripts/run_motivation_loco.sh
#
# Env knobs:
#   MODEL, WEIGHTS, FROM_SCRATCH   model family and initialization (default yolo26m / yolo26m-cls.pt)
#   EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE   train/eval resources
#   SEED                           LoCo image assignment and training seed (default 0)
#   FORCE_TRAIN                    set 1 to retrain every task (default 0)
#   FORCE_CREATE                   set 1 to rebuild both LoCo datasets (default 0; pair with FORCE_TRAIN=1)
#   END2END, AGNOSTIC_NMS, MAX_DET decode mode (default False / True / 300)
#   CONF_MIN, CONF_MAX, CONF_STEP  error-rate confidence sweep (default 0.05 / 0.95 / 0.05)
#   VIS_CONF, PREDICT_CONF         example and dump confidence (default 0.25 / 0.01)
#   IOU_THRESHOLD                  GT matching IoU (default 0.5)
#   SUMMARY_DIR                    report directory (default runs/<model>_VOC_LoCo_motivation)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Offline execution: never auto-install packages or query the update service.
export YOLO_OFFLINE=1
export YOLO_AUTOINSTALL=false
export MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
MODEL="${MODEL:-yolo26m}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"
if [[ "$FROM_SCRATCH" == "1" ]]; then
    WEIGHTS=""
else
    WEIGHTS="${WEIGHTS:-yolo26m-cls.pt}"
fi
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
# Eval / predict run on one device even when training uses a multi-GPU list.
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE%%,*}}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_CREATE="${FORCE_CREATE:-0}"
END2END="${END2END:-False}"
AGNOSTIC_NMS="${AGNOSTIC_NMS:-True}"
MAX_DET="${MAX_DET:-300}"
CONF_MIN="${CONF_MIN:-0.05}"
CONF_MAX="${CONF_MAX:-0.95}"
CONF_STEP="${CONF_STEP:-0.05}"
VIS_CONF="${VIS_CONF:-0.25}"
PREDICT_CONF="${PREDICT_CONF:-0.01}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"

export SEED WORKERS EPOCHS BATCH_SIZE IMGSZ DEVICE

TAG_TWO="VOC_LoCo_10+10"
TAG_FOUR="VOC_LoCo_5+5+5+5"
TAG_JOINT="VOC_LoCo_joint"
DATA_TWO="data/${TAG_TWO}"
DATA_FOUR="data/${TAG_FOUR}"
SOURCE_CFG="data/VOC-YOLO/VOC.yaml"

TWO_STAGE_YAMLS=(
    "$DATA_TWO/task_1_cls_10/dataset.yaml"
    "$DATA_TWO/task_2_cls_10/dataset.yaml"
)
TWO_CUMULATIVE_YAMLS=(
    "$DATA_TWO/task_1_cls_10/dataset.yaml"
    "$DATA_TWO/task_1-2_cls_20/dataset.yaml"
)
FOUR_STAGE_YAMLS=(
    "$DATA_FOUR/task_1_cls_5/dataset.yaml"
    "$DATA_FOUR/task_2_cls_5/dataset.yaml"
    "$DATA_FOUR/task_3_cls_5/dataset.yaml"
    "$DATA_FOUR/task_4_cls_5/dataset.yaml"
)
FOUR_CUMULATIVE_YAMLS=(
    "$DATA_FOUR/task_1_cls_5/dataset.yaml"
    "$DATA_FOUR/task_1-2_cls_10/dataset.yaml"
    "$DATA_FOUR/task_1-3_cls_15/dataset.yaml"
    "$DATA_FOUR/task_1-4_cls_20/dataset.yaml"
)

run_prefix() {
    local tag="$1" method="$2" weights_name
    if [[ -n "$WEIGHTS" ]]; then
        weights_name="$(basename "$WEIGHTS")"
        echo "runs/${MODEL}_${tag}_pretrained-from-${weights_name%.pt}_${method}"
    else
        echo "runs/${MODEL}_${tag}_fromscratch_${method}"
    fi
}

TWO_NAIVE_RUN="$(run_prefix "$TAG_TWO" naive)"
TWO_LWF_RUN="$(run_prefix "$TAG_TWO" lwf)"
FOUR_NAIVE_RUN="$(run_prefix "$TAG_FOUR" naive)"
FOUR_LWF_RUN="$(run_prefix "$TAG_FOUR" lwf)"
JOINT_RUN="$(run_prefix "$TAG_JOINT" naive)"
JOINT_EVAL_TWO="$(run_prefix "${TAG_JOINT}10" naive)"
JOINT_EVAL_FOUR="$(run_prefix "${TAG_JOINT}4" naive)"
SUMMARY_DIR="${SUMMARY_DIR:-runs/${MODEL}_VOC_LoCo_motivation}"
SUMMARY_MD="${SUMMARY_DIR}/summary.md"
LOG_DIR="${SUMMARY_DIR}/logs"

if [[ -n "$WEIGHTS" ]]; then
    WEIGHT_ARGS=(--weights "$WEIGHTS")
else
    WEIGHT_ARGS=(--from-scratch)
fi
TRAIN_EXTRA=(
    --optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5
    --seed "$SEED"
    --end2end "$END2END" --agnostic_nms "$AGNOSTIC_NMS" --max_det "$MAX_DET"
)
PREDICT_EXTRA=(
    --end2end "$END2END" --agnostic_nms "$AGNOSTIC_NMS" --max_det "$MAX_DET"
)

usage() {
    sed -n '2,40p' "$0"
}

if (( $# > 0 )); then
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (only --help is accepted)"
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Preflight: local data, checkpoints, and Python packages only.
# ---------------------------------------------------------------------------
require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1${2:+ ($2)}"
}

require_dir() {
    [[ -d "$1" ]] || die "Required directory not found: $1${2:+ ($2)}"
}

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "VOC LoCo motivation pipeline"
echo "  model    : ${MODEL}  weights=${WEIGHTS:-<from scratch>}"
echo "  seed     : ${SEED}  epochs=${EPOCHS}  batch=${BATCH_SIZE}  device=${DEVICE}"
echo "  report   : ${SUMMARY_MD}"
echo "=========================================="

run_logged() {
    local name="$1"
    shift
    echo ""
    echo "----- ${name} -----"
    echo ">>> $*"
    "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

require_dir "data/VOC-YOLO" "VOC dataset"
for split in train val test; do
    require_dir "data/VOC-YOLO/images/${split}" "VOC images"
    require_dir "data/VOC-YOLO/labels/${split}" "VOC labels"
done
require_file "$SOURCE_CFG" "VOC dataset yaml"
require_file "experiments.md" "reference report"
if [[ -n "$WEIGHTS" ]]; then
    require_file "$WEIGHTS" "initial checkpoint (set FROM_SCRATCH=1 to train without one)"
fi
require_file "scripts/create.sh" "dataset creation entry"
require_file "scripts/train.sh" "training entry"
require_file "scripts/eval.sh" "evaluation entry"
require_file "scripts/analyze.sh" "analysis entry"
require_file "tools/loco_report.py" "report tool"

python - <<'PY'
import cv2
import matplotlib
import numpy
import torch
from ultralytics.utils import YAML
PY
echo "Preflight OK"

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
create_dataset() {
    local split_id="$1" data_root="$2" first_yaml="$3"
    if [[ "$FORCE_CREATE" != "1" && -f "$first_yaml" ]]; then
        require_file "${data_root}/loco_partition.json" \
            "LoCo partition metadata; rerun with FORCE_CREATE=1 to rebuild the dataset"
        echo "Dataset already present: ${data_root}"
        return
    fi
    if [[ "$FORCE_CREATE" != "1" && -e "$data_root" ]]; then
        die "Directory ${data_root} exists but ${first_yaml} is missing; remove it or rerun with FORCE_CREATE=1"
    fi
    local -a overwrite_args=()
    if [[ "$FORCE_CREATE" == "1" ]]; then
        overwrite_args=(--overwrite)
    fi
    run_logged "create_${split_id}" bash scripts/create.sh voc-loco "$split_id" ${overwrite_args[@]+"${overwrite_args[@]}"}
    require_file "$first_yaml" "created LoCo stage yaml"
    require_file "${data_root}/loco_partition.json" "LoCo partition metadata"
}

echo ""
echo "=========================================="
echo "Stage 1/6: build LoCo datasets"
echo "=========================================="
create_dataset "10_10" "$DATA_TWO" "${TWO_STAGE_YAMLS[0]}"
create_dataset "5_5_5_5" "$DATA_FOUR" "${FOUR_STAGE_YAMLS[0]}"
for yaml in "${TWO_STAGE_YAMLS[@]}" "${TWO_CUMULATIVE_YAMLS[@]}" \
            "${FOUR_STAGE_YAMLS[@]}" "${FOUR_CUMULATIVE_YAMLS[@]}"; do
    require_file "$yaml" "LoCo dataset yaml"
done

# ---------------------------------------------------------------------------
# Motivation experiment 1: stage class distribution and sample overlap
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Stage 2/6: stage statistics"
echo "=========================================="
run_logged "stats_two" bash scripts/analyze.sh stage_stats \
    --stages "${TWO_STAGE_YAMLS[@]}" \
    --source_cfg "$SOURCE_CFG" \
    --split train \
    --output_dir "${TWO_NAIVE_RUN}/stage_stats"
run_logged "stats_four" bash scripts/analyze.sh stage_stats \
    --stages "${FOUR_STAGE_YAMLS[@]}" \
    --source_cfg "$SOURCE_CFG" \
    --split train \
    --output_dir "${FOUR_NAIVE_RUN}/stage_stats"

# ---------------------------------------------------------------------------
# Incremental training
# ---------------------------------------------------------------------------
train_run() {
    local name="$1" tag="$2" run="$3" method="$4" first_task="$5"
    shift 5
    local -a yamls=("$@")
    local total=${#yamls[@]}
    local start=$((total + 1))
    if [[ "$FORCE_TRAIN" == "1" ]]; then
        start=$first_task
    else
        local index
        for ((index = first_task - 1; index < total; index++)); do
            if [[ ! -f "${run}/task-$((index + 1))/best.pt" ]]; then
                start=$((index + 1))
                break
            fi
        done
        if (( start > total )); then
            echo "Training already complete: ${run}"
            return
        fi
    fi
    run_logged "train_${name}" env START_TASK="$start" bash scripts/train.sh \
        --tasks "${yamls[@]}" \
        --model "$MODEL" \
        --method "$method" \
        "${WEIGHT_ARGS[@]}" \
        --tag "$tag" \
        --output "$run" \
        -- "${TRAIN_EXTRA[@]}"
}

train_lwf() {
    local name="$1" tag="$2" run="$3" naive_run="$4"
    shift 4
    local -a yamls=("$@")
    local total=${#yamls[@]}
    local needs_training=0
    if [[ "$FORCE_TRAIN" == "1" || ! -f "${run}/task-1/best.pt" ]]; then
        needs_training=1
    else
        local index
        for ((index = 2; index <= total; index++)); do
            if [[ ! -f "${run}/task-${index}/best.pt" ]]; then
                needs_training=1
                break
            fi
        done
    fi
    if (( needs_training == 0 )); then
        echo "Training already complete: ${run}"
        return
    fi
    # YOLO-LwF does not distill at task 1; both methods share the naive task-1 checkpoint.
    require_file "${naive_run}/task-1/best.pt" "naive task-1 checkpoint"
    mkdir -p "${run}/task-1"
    cp -f "${naive_run}/task-1/best.pt" "${run}/task-1/best.pt"
    train_run "$name" "$tag" "$run" lwf 2 "${yamls[@]}"
}

train_joint() {
    local run="$1" yaml="$2"
    if [[ "$FORCE_TRAIN" != "1" && -f "${run}/task-1/best.pt" ]]; then
        echo "Joint model already trained: ${run}"
        return
    fi
    run_logged "train_joint" bash scripts/train.sh \
        --tasks "$yaml" \
        --model "$MODEL" \
        --method naive \
        "${WEIGHT_ARGS[@]}" \
        --tag "$TAG_JOINT" \
        --output "$run" \
        -- "${TRAIN_EXTRA[@]}"
}

echo ""
echo "=========================================="
echo "Stage 3/6: incremental training"
echo "=========================================="
train_run "two_naive" "$TAG_TWO" "$TWO_NAIVE_RUN" naive 1 "${TWO_STAGE_YAMLS[@]}"
train_run "four_naive" "$TAG_FOUR" "$FOUR_NAIVE_RUN" naive 1 "${FOUR_STAGE_YAMLS[@]}"
train_lwf "two_lwf" "$TAG_TWO" "$TWO_LWF_RUN" "$TWO_NAIVE_RUN" "${TWO_STAGE_YAMLS[@]}"
train_lwf "four_lwf" "$TAG_FOUR" "$FOUR_LWF_RUN" "$FOUR_NAIVE_RUN" "${FOUR_STAGE_YAMLS[@]}"
train_joint "$JOINT_RUN" "${FOUR_CUMULATIVE_YAMLS[3]}"

require_file "${TWO_NAIVE_RUN}/task-1/best.pt" "trained checkpoint"
require_file "${TWO_NAIVE_RUN}/task-2/best.pt" "trained checkpoint"
require_file "${TWO_LWF_RUN}/task-1/best.pt" "trained checkpoint"
require_file "${TWO_LWF_RUN}/task-2/best.pt" "trained checkpoint"
require_file "${FOUR_LWF_RUN}/task-1/best.pt" "trained checkpoint"
for stage in 2 3 4; do
    require_file "${FOUR_NAIVE_RUN}/task-${stage}/best.pt" "trained checkpoint"
    require_file "${FOUR_LWF_RUN}/task-${stage}/best.pt" "trained checkpoint"
done
require_file "${JOINT_RUN}/task-1/best.pt" "Joint checkpoint"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
eval_two() {
    local name="$1" run="$2"
    run_logged "eval_${name}" env DEVICE="$EVAL_DEVICE" bash scripts/eval.sh \
        --run "$run" \
        --tasks "${TWO_STAGE_YAMLS[@]}" \
        --cumulative "${TWO_CUMULATIVE_YAMLS[@]}"
}

eval_four() {
    local name="$1" run="$2"
    run_logged "eval_${name}" env DEVICE="$EVAL_DEVICE" bash scripts/eval.sh \
        --run "$run" \
        --tasks "${FOUR_STAGE_YAMLS[@]}" \
        --cumulative "${FOUR_CUMULATIVE_YAMLS[@]}"
}

# A Joint eval directory exposes the shared Joint checkpoint as task-1/best.pt
# so the protocol-specific evaluation writes into a separate result tree.
prepare_joint_eval() {
    local eval_dir="$1"
    require_file "${JOINT_RUN}/task-1/best.pt" "Joint checkpoint"
    mkdir -p "${eval_dir}/task-1"
    rm -f "${eval_dir}/task-1/best.pt"
    ln -s "$(python -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' \
        "${JOINT_RUN}/task-1/best.pt" "${eval_dir}/task-1")" "${eval_dir}/task-1/best.pt"
}

echo ""
echo "=========================================="
echo "Stage 4/6: evaluation"
echo "=========================================="
eval_two "two_naive" "$TWO_NAIVE_RUN"
eval_two "two_lwf" "$TWO_LWF_RUN"
eval_four "four_naive" "$FOUR_NAIVE_RUN"
eval_four "four_lwf" "$FOUR_LWF_RUN"
prepare_joint_eval "$JOINT_EVAL_TWO"
prepare_joint_eval "$JOINT_EVAL_FOUR"
eval_two "joint_two" "$JOINT_EVAL_TWO"
eval_four "joint_four" "$JOINT_EVAL_FOUR"

# ---------------------------------------------------------------------------
# Motivation experiment 2: teacher high-confidence errors
# ---------------------------------------------------------------------------
teacher_errors() {
    local name="$1" run="$2" target_yaml="$3"
    shift 3
    local -a stage_yamls=("$@")
    local out="${run}/teacher_errors/task_2"
    local images
    images="$(dirname "$target_yaml")/images/train"
    require_dir "$images" "teacher stage images"
    rm -rf "${out}/predictions" "${out}/examples"
    mkdir -p "$out"
    run_logged "teacher_${name}_predict" python tools/predict.py \
        --model "${run}/task-1/best.pt" \
        --images "$images" \
        --save_path "${out}/predictions" \
        --project "$out" \
        --device "$EVAL_DEVICE" \
        --conf "$PREDICT_CONF" \
        "${PREDICT_EXTRA[@]}"
    run_logged "teacher_${name}_analyze" bash scripts/analyze.sh teacher_errors \
        --predictions "${out}/predictions" \
        --model "${run}/task-1/best.pt" \
        --stages "${stage_yamls[@]}" \
        --target_stage 2 \
        --source_cfg "$SOURCE_CFG" \
        --split train \
        --iou_threshold "$IOU_THRESHOLD" \
        --conf_min "$CONF_MIN" \
        --conf_max "$CONF_MAX" \
        --conf_step "$CONF_STEP" \
        --vis_dir "${out}/examples" \
        --vis_conf "$VIS_CONF" \
        --output_dir "$out"
}

echo ""
echo "=========================================="
echo "Stage 5/6: teacher and incremental-model error analyses"
echo "=========================================="
teacher_errors "two" "$TWO_NAIVE_RUN" "${TWO_STAGE_YAMLS[1]}" "${TWO_STAGE_YAMLS[@]}"
teacher_errors "four" "$FOUR_NAIVE_RUN" "${FOUR_STAGE_YAMLS[1]}" "${FOUR_STAGE_YAMLS[@]}"

# ---------------------------------------------------------------------------
# Experiments 2 (severity) and 3: model errors on the history stage
# ---------------------------------------------------------------------------
analyze_incremental() {
    local name="$1" model="$2" images="$3" out="$4"
    shift 4
    local -a stage_yamls=("$@")
    require_file "$model" "analysis checkpoint"
    require_dir "$images" "analysis images"
    local -a future_args=()
    if (( ${#stage_yamls[@]} > 2 )); then
        future_args=(--future_stages 3 4)
    fi
    rm -rf "${out}/predictions" "${out}/examples"
    mkdir -p "$out"
    run_logged "history_${name}_predict" python tools/predict.py \
        --model "$model" \
        --images "$images" \
        --save_path "${out}/predictions" \
        --project "$out" \
        --device "$EVAL_DEVICE" \
        --conf "$PREDICT_CONF" \
        "${PREDICT_EXTRA[@]}"
    run_logged "history_${name}_analyze" bash scripts/analyze.sh incremental_errors \
        --predictions "${out}/predictions" \
        --model "$model" \
        --stages "${stage_yamls[@]}" \
        --history_stages 1 \
        ${future_args[@]+"${future_args[@]}"} \
        --source_cfg "$SOURCE_CFG" \
        --split test \
        --iou_threshold "$IOU_THRESHOLD" \
        --conf_min "$CONF_MIN" \
        --conf_max "$CONF_MAX" \
        --conf_step "$CONF_STEP" \
        --vis_dir "${out}/examples" \
        --vis_conf "$VIS_CONF" \
        --output_dir "$out"
}

analyze_protocol() {
    local protocol="$1" run_naive="$2" run_lwf="$3" joint_eval="$4" cum2_yaml="$5"
    shift 5
    local -a stage_yamls=("$@")
    local test1_images test2_images cum2_images
    test1_images="$(dirname "${stage_yamls[0]}")/images/test"
    test2_images="$(dirname "${stage_yamls[1]}")/images/test"
    cum2_images="$(dirname "$cum2_yaml")/images/test"

    local -a models=(
        "base|${run_naive}/task-1/best.pt|${run_naive}/history_errors-base"
        "naive|${run_naive}/task-2/best.pt|${run_naive}/history_errors"
        "lwf|${run_lwf}/task-2/best.pt|${run_lwf}/history_errors"
        "joint|${joint_eval}/task-1/best.pt|${joint_eval}/history_errors"
    )
    local -a test_specs=(
        "stage-1|${test1_images}"
        "stage-2|${test2_images}"
        "cum-2|${cum2_images}"
    )
    local model_spec test_spec model_key model_path out_root test_key images
    for model_spec in "${models[@]}"; do
        IFS='|' read -r model_key model_path out_root <<< "$model_spec"
        for test_spec in "${test_specs[@]}"; do
            IFS='|' read -r test_key images <<< "$test_spec"
            analyze_incremental "${protocol}_${model_key}_${test_key}" "$model_path" "$images" \
                "${out_root}/on-${test_key}" "${stage_yamls[@]}"
        done
    done
}

analyze_protocol "two" "$TWO_NAIVE_RUN" "$TWO_LWF_RUN" "$JOINT_EVAL_TWO" \
    "${TWO_CUMULATIVE_YAMLS[1]}" "${TWO_STAGE_YAMLS[@]}"
analyze_protocol "four" "$FOUR_NAIVE_RUN" "$FOUR_LWF_RUN" "$JOINT_EVAL_FOUR" \
    "${FOUR_CUMULATIVE_YAMLS[1]}" "${FOUR_STAGE_YAMLS[@]}"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Stage 6/6: formatted summary"
echo "=========================================="
run_logged "report" python tools/loco_report.py \
    --two_root "$DATA_TWO" \
    --four_root "$DATA_FOUR" \
    --run_two_naive "$TWO_NAIVE_RUN" \
    --run_two_lwf "$TWO_LWF_RUN" \
    --run_four_naive "$FOUR_NAIVE_RUN" \
    --run_four_lwf "$FOUR_LWF_RUN" \
    --joint_eval_two "$JOINT_EVAL_TWO" \
    --joint_eval_four "$JOINT_EVAL_FOUR" \
    --reference "experiments.md" \
    --seed "$SEED" \
    --output "$SUMMARY_MD"

echo ""
echo "=========================================="
echo "LoCo motivation pipeline complete"
echo "  report : ${SUMMARY_MD}"
echo "  logs   : ${LOG_DIR}"
echo "=========================================="
