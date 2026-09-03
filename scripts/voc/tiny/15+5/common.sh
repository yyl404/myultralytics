#!/bin/bash
# Shared parameters for this incremental demo, sourced by the sibling scripts.
# To migrate to another incremental dataset or setting, edit only this file.
# Paths are repo-relative: the stage scripts cd to the repo root first.
#
# Every value can also be overridden for one launch with an environment
# variable of the same name — scalars directly, lists as space-separated
# strings. Examples:
#   EPOCHS=1 MODEL=yolo26x RUN_DIR=runs/smoke bash <this-dir>/pipeline.sh
#   TASK_YAMLS="data/A/t1.yaml data/A/t2.yaml" bash <this-dir>/train.sh

DATA_ROOT="${DATA_ROOT:-data/VOC-TINY_15+5}"

# Incremental task datasets, one yaml per task, in task order.
if [[ -z "${TASK_YAMLS:-}" ]]; then
    TASK_YAMLS="$DATA_ROOT/task_1_cls_15/dataset.yaml $DATA_ROOT/task_2_cls_5/dataset.yaml"
fi
read -ra TASK_YAMLS <<< "$TASK_YAMLS"

# Cumulative eval/predict datasets (optional; set to () in this file to skip).
if [[ -z "${CUMULATIVE_YAMLS:-}" ]]; then
    CUMULATIVE_YAMLS="$DATA_ROOT/task_1_cls_15/dataset.yaml $DATA_ROOT/task_1-2_cls_20/dataset.yaml"
fi
read -ra CUMULATIVE_YAMLS <<< "$CUMULATIVE_YAMLS"

# Run identity.
MODEL="${MODEL:-yolo26m}"
WEIGHTS="${WEIGHTS:-yoloe-26m-seg.pt}"
METHOD="${METHOD:-naive}"
DATA_TAG="${DATA_TAG:-VOC-TINY_15+5}"
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

# Extra args forwarded to tools/train.py / tools/eval.py / tools/predict.py
# (everything after `--`). Train/eval/predict all run with NMS + agnostic_nms
# here; yolo26 additionally gets --end2end False automatically.
if [[ -z "${EXTRA_TRAIN_ARGS:-}" ]]; then
    EXTRA_TRAIN_ARGS="--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5 --freeze 10 --agnostic_nms True"
fi
read -ra EXTRA_TRAIN_ARGS <<< "$EXTRA_TRAIN_ARGS"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:---agnostic_nms True}"
read -ra EXTRA_EVAL_ARGS <<< "$EXTRA_EVAL_ARGS"
EXTRA_PREDICT_ARGS="${EXTRA_PREDICT_ARGS:---agnostic_nms True}"
read -ra EXTRA_PREDICT_ARGS <<< "$EXTRA_PREDICT_ARGS"

# Train knobs (read from the environment by scripts/train.sh).
export EPOCHS="${EPOCHS:-10}"
export DEVICE="${DEVICE:-0}"
