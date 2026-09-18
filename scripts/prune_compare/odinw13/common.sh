#!/bin/bash
# Shared parameters for prune-vs-random comparison.
# To migrate to another dataset, edit only the first block of this file;
# run.sh is an identical template in every prune_compare copy. Paths are
# repo-relative: the stage script cds to the repo root.
#
# The dataset block is read from this file only. Every scalar further down
# can also be overridden for one launch with an environment variable of the
# same name, e.g.:
#   EPOCHS=1 RUN_ROOT=runs/smoke bash <this-dir>/run.sh

# ============================================================================
# Dataset / comparison — the ONLY block to edit when migrating.
#   SRC_DATA           source YOLO dataset yaml (full original data)
#   DATA_TAG           tag used in output / run directory names
#   EVAL_YAML          yaml used for the final test (usually the same as SRC_DATA)
#   EVAL_SPLIT         split of EVAL_YAML to score (test, else val)
#   RATIOS             keep-ratio grid; each split is pruned independently
#   SPLITS             which source splits to compress into the subset
#   RANDOM_SPLITS      among SPLITS, uniformly sampled (empty = coreset all of SPLITS)
#   ATTACH_SOURCE_SPLITS
#                      extra source splits linked into the subset yaml by
#                      relative path without compression (typically val test,
#                      so train-time val and the original test stay full-size)
#   PREP_CMDS          optional commands run once before pruning (empty to skip)
# ============================================================================
SRC_DATA="data/OdinW-13-yolo/VehiclesOpenImages/dataset.yaml"
DATA_TAG="OdinW-13-VehiclesOpenImages"
EVAL_YAML="$SRC_DATA"
EVAL_SPLIT="test"
RATIOS=(0.15 0.20)
SPLITS=(train)
RANDOM_SPLITS=()
# After pruning, attach these source splits by relative path so train-time
# validation and any extra eval splits stay on the full original data.
ATTACH_SOURCE_SPLITS=(val test)
PREP_CMDS=(
    "[[ -f data/OdinW-13-yolo/VehiclesOpenImages/dataset.yaml ]] || python tools/convert_odinw13_to_yolo.py --src_dir data/OdinW-13 --out_dir data/OdinW-13-yolo --domains VehiclesOpenImages"
)

# Run identity.
MODEL="${MODEL:-yolo26n}"
WEIGHTS="${WEIGHTS:-yoloe-26n-seg.pt}"
METHOD="${METHOD:-naive}"
BACKBONE="${BACKBONE:-none}"
SEED="${SEED:-0}"
MIN_INTERVAL="${MIN_INTERVAL:-2}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-data/prune_subsets/${DATA_TAG}/feature_cache}"

# Output locations (large artifacts live under data/ -> /hy-tmp via the data symlink).
SUBSET_ROOT="${SUBSET_ROOT:-data/prune_subsets/${DATA_TAG}}"
if [[ -z "${RUN_ROOT:-}" ]]; then
    RUN_ROOT="runs/prune_compare/${DATA_TAG}"
fi
SUMMARY_CSV="${SUMMARY_CSV:-$RUN_ROOT/summary.csv}"

# Decode / NMS knobs forwarded to train and eval.
END2END="${END2END:-False}"
AGNOSTIC_NMS="${AGNOSTIC_NMS:-True}"
MAX_DET="${MAX_DET:-300}"
export END2END

DECODE_ARGS="--end2end $END2END --agnostic_nms $AGNOSTIC_NMS --max_det $MAX_DET"

if [[ -z "${EXTRA_TRAIN_ARGS:-}" ]]; then
    EXTRA_TRAIN_ARGS="--optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5 --freeze 10 --val False $DECODE_ARGS"
fi
read -ra EXTRA_TRAIN_ARGS <<< "$EXTRA_TRAIN_ARGS"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-$DECODE_ARGS}"
read -ra EXTRA_EVAL_ARGS <<< "$EXTRA_EVAL_ARGS"

# Skip the class-agnostic eval pass; this comparison only needs class-aware mAP50.
export CLASS_AGNOSTIC_EVAL="${CLASS_AGNOSTIC_EVAL:-0}"

# Train knobs (read from the environment by scripts/train.sh).
export EPOCHS="${EPOCHS:-20}"
export DEVICE="${DEVICE:-0}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
