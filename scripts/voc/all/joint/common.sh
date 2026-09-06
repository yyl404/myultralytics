#!/bin/bash
# Shared parameters for this incremental demo, sourced by the sibling scripts.
# To migrate to another incremental dataset or split, edit only the first
# block of this file; everything below it is an identical template in every
# demo copy. Paths are repo-relative: the stage scripts cd to the repo root.
#
# The dataset block is read from this file only — the yaml sequences cannot be
# set from the command line or the environment. Every scalar further down can
# also be overridden for one launch with an environment variable of the same
# name, e.g.:
#   EPOCHS=1 MODEL=yolo26x RUN_DIR=runs/smoke bash <this-dir>/pipeline.sh

# ============================================================================
# Dataset / split — the ONLY block to edit when migrating.
#   DATA_ROOT          root directory of this incremental dataset
#   DATA_TAG           tag used in the run directory name
#   TRAIN_YAMLS        incremental TRAIN sequence: one task dataset yaml per
#                      task, in task order (train.sh / similarity.sh)
#   EVAL_YAMLS         independent EVAL sequence: datasets each checkpoint is
#                      evaluated on one by one; may differ from TRAIN_YAMLS in
#                      content, order, and length (eval.sh / predict.sh)
#   CUMULATIVE_YAMLS   cumulative EVAL sequence (optional); set to () to skip
#                      cumulative evaluation (eval.sh / predict.sh)
# ============================================================================
DATA_ROOT="data/VOC-YOLO"
DATA_TAG="VOC_15+5"

TRAIN_YAMLS=(
    "$DATA_ROOT/VOC.yaml"
)
EVAL_YAMLS=(
    "$DATA_ROOT/task_1_cls_15/dataset.yaml"
    "$DATA_ROOT/task_2_cls_5/dataset.yaml"
)
CUMULATIVE_YAMLS=(
    "$DATA_ROOT/task_1_cls_15/dataset.yaml"
    "$DATA_ROOT/task_1-2_cls_20/dataset.yaml"
)

# Run identity.
MODEL="${MODEL:-yolo26m}"
WEIGHTS="${WEIGHTS:-yoloe-26m-seg.pt}"
METHOD="${METHOD:-naive}"
# Default output dir is composed from the identity above (same naming rule as
# scripts/train.sh); override RUN_DIR to use a fixed location instead.
if [[ -z "${RUN_DIR:-}" ]]; then
    if [[ -n "$WEIGHTS" ]]; then
        RUN_DIR="runs/${MODEL}_${DATA_TAG}_pretrained-from-${WEIGHTS%.pt}_${METHOD}"
    else
        RUN_DIR="runs/${MODEL}_${DATA_TAG}_fromscratch_${METHOD}"
    fi
fi

# Model used for the predict stage (any task-k/best.pt).
PREDICT_MODEL="${PREDICT_MODEL:-$RUN_DIR/task-2/best.pt}"

# Pretrained backbone weights for the similarity stage (similarity.sh);
# defaults to the same weights used to initialize training.
SIMILARITY_WEIGHTS="${SIMILARITY_WEIGHTS:-$WEIGHTS}"

# ============================================================================
# Decode / NMS mode knobs — DEFAULTS BELOW, edit here to switch all stages.
#   END2END       False = one-to-many + NMS (default) | True = NMS-free one-to-one
#   AGNOSTIC_NMS  True = class-agnostic NMS (default) | False = per-class NMS
#   MAX_DET       Max detections kept per image (default: 300)
# Forwarded to train / eval / predict via DECODE_ARGS in the EXTRA_*_ARGS
# below. scripts/train.sh additionally reads the exported END2END for its
# yolo26 default hyps, so training stays consistent even when EXTRA_TRAIN_ARGS
# is overridden from the environment.
# ============================================================================
END2END="${END2END:-False}"
AGNOSTIC_NMS="${AGNOSTIC_NMS:-True}"
MAX_DET="${MAX_DET:-300}"
export END2END

DECODE_ARGS="--end2end $END2END --agnostic_nms $AGNOSTIC_NMS --max_det $MAX_DET"

# Extra args forwarded to tools/train.py / tools/eval.py / tools/predict.py
# (everything after `--`). Overriding an EXTRA_*_ARGS env var replaces the
# whole string, decode knobs included.
if [[ -z "${EXTRA_TRAIN_ARGS:-}" ]]; then
    EXTRA_TRAIN_ARGS="--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5 --freeze 10 $DECODE_ARGS"
fi
read -ra EXTRA_TRAIN_ARGS <<< "$EXTRA_TRAIN_ARGS"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-$DECODE_ARGS}"
read -ra EXTRA_EVAL_ARGS <<< "$EXTRA_EVAL_ARGS"
EXTRA_PREDICT_ARGS="${EXTRA_PREDICT_ARGS:-$DECODE_ARGS}"
read -ra EXTRA_PREDICT_ARGS <<< "$EXTRA_PREDICT_ARGS"

# Train knobs (read from the environment by scripts/train.sh).
export EPOCHS="${EPOCHS:-10}"
export DEVICE="${DEVICE:-0}"
