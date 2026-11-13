#!/bin/bash
# Shared parameters for the motivation experiments, sourced by the sibling scripts.
# To migrate to another dataset or split, edit only the first two blocks below;
# the stage scripts stay unchanged. Paths are repo-relative: the stage scripts cd
# to the repo root.
#
# Every scalar can be overridden for one launch with an environment variable of
# the same name, e.g.:
#   EPOCHS=1 RUN_DIR_TWO=runs/smoke bash <this-dir>/pipeline.sh

# ============================================================================
# Two-stage protocol. One dataset yaml per stage, in stage order; the same
# sequence drives training, the stage statistics and the teacher analysis.
# ============================================================================
TWO_STAGE_ROOT="data/VOC_10+10"
TWO_STAGE_TAG="VOC_10+10"
TWO_STAGE_YAMLS=(
    "$TWO_STAGE_ROOT/task_1_cls_10/dataset.yaml"
    "$TWO_STAGE_ROOT/task_2_cls_10/dataset.yaml"
)

# ============================================================================
# Multi-stage protocol (four stages, same class order as TWO_STAGE_YAMLS).
# ============================================================================
FOUR_STAGE_ROOT="data/VOC_5+5+5+5"
FOUR_STAGE_TAG="VOC_5+5+5+5"
FOUR_STAGE_YAMLS=(
    "$FOUR_STAGE_ROOT/task_1_cls_5/dataset.yaml"
    "$FOUR_STAGE_ROOT/task_2_cls_5/dataset.yaml"
    "$FOUR_STAGE_ROOT/task_3_cls_5/dataset.yaml"
    "$FOUR_STAGE_ROOT/task_4_cls_5/dataset.yaml"
)

# Protocol selector for the stage scripts: two, four, or all.
PROTOCOL="${PROTOCOL:-all}"

# Full class-space annotation used by the statistics and teacher analysis.
SOURCE_CFG="data/VOC-YOLO/VOC.yaml"

# `scripts/create.sh` recipes for the protocols above (dataset / split id).
TWO_STAGE_SPLIT_ID="10_10"
FOUR_STAGE_SPLIT_ID="5_5_5_5"

# Run identity.
MODEL="${MODEL:-yolo26m}"
WEIGHTS="${WEIGHTS:-yolo26m-cls.pt}"
METHOD="${METHOD:-naive}"
# Run directory of one protocol and one method, matching the naming of
# scripts/train.sh so every stage script resolves the same checkpoint.
motivation_run_dir() {
    local tag="$1" method="$2"
    if [[ -n "$WEIGHTS" ]]; then
        echo "runs/${MODEL}_${tag}_pretrained-from-${WEIGHTS%.pt}_${method}"
    else
        echo "runs/${MODEL}_${tag}_fromscratch_${method}"
    fi
}

if [[ -z "${RUN_DIR_TWO:-}" ]]; then
    RUN_DIR_TWO="$(motivation_run_dir "$TWO_STAGE_TAG" "$METHOD")"
fi
if [[ -z "${RUN_DIR_FOUR:-}" ]]; then
    RUN_DIR_FOUR="$(motivation_run_dir "$FOUR_STAGE_TAG" "$METHOD")"
fi

# ============================================================================
# Decode / NMS mode knobs, forwarded to train / predict / analysis.
# ============================================================================
END2END="${END2END:-False}"
AGNOSTIC_NMS="${AGNOSTIC_NMS:-True}"
MAX_DET="${MAX_DET:-300}"
export END2END

DECODE_ARGS="--end2end $END2END --agnostic_nms $AGNOSTIC_NMS --max_det $MAX_DET"

# Extra args forwarded to tools/train.py and tools/predict.py (everything after
# `--`). Overriding an EXTRA_*_ARGS env var replaces the whole string.
if [[ -z "${EXTRA_TRAIN_ARGS:-}" ]]; then
    EXTRA_TRAIN_ARGS="--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5 $DECODE_ARGS"
fi
read -ra EXTRA_TRAIN_ARGS <<< "$EXTRA_TRAIN_ARGS"
EXTRA_PREDICT_ARGS="${EXTRA_PREDICT_ARGS:-$DECODE_ARGS}"
read -ra EXTRA_PREDICT_ARGS <<< "$EXTRA_PREDICT_ARGS"

# Teacher analysis: stage 2 is the incremental stage seen by the task-1 teacher;
# GT comes from the full class-space annotation of the analysis split.
TARGET_STAGE="${TARGET_STAGE:-2}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-train}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
CONF_MIN="${CONF_MIN:-0.05}"
CONF_MAX="${CONF_MAX:-0.95}"
CONF_STEP="${CONF_STEP:-0.05}"
# Confidence threshold for the typical error visualizations (Ultralytics default).
VIS_CONF="${VIS_CONF:-0.25}"
# Predictions are dumped below the lowest swept threshold so every threshold
# in the sweep is backed by the same dump.
PREDICT_CONF="${PREDICT_CONF:-0.01}"

# Incremental-model analysis: each method below is trained on the protocol and
# then run on the history-stage images (two-stage protocol only).
INCREMENTAL_METHODS=("naive" "lwf")
# Stage whose checkpoint is analyzed, and the stages whose classes count as history.
INCREMENTAL_STAGE="${INCREMENTAL_STAGE:-2}"
INCREMENTAL_HISTORY_STAGES="${INCREMENTAL_HISTORY_STAGES:-1}"
# Stage whose images the model is run on: the first history stage by default.
INCREMENTAL_IMAGES_STAGE="${INCREMENTAL_IMAGES_STAGE:-${INCREMENTAL_HISTORY_STAGES%% *}}"
INCREMENTAL_SPLIT="${INCREMENTAL_SPLIT:-test}"

# Train knobs (read from the environment by scripts/train.sh).
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export IMGSZ="${IMGSZ:-640}"
export WORKERS="${WORKERS:-8}"
export DEVICE="${DEVICE:-0}"
