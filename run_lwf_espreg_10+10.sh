#!/bin/bash
# Offline experiment: YOLO-LwF distillation + ESPReg (raw eigenvalues, Frobenius form)
# on the VOC 10+10 class-incremental protocol.
#
# This is one cell of the 2x2 comparison {"LwF", "prad+prd"} x {"no ESPReg", "ESPReg"}:
# it answers "what does ESPReg do to the LwF baseline" and, together with the full method,
# "what does the prototype distillation add under the same regularizer".
#
# Everything is repo-relative and offline:
#   - the split is sliced from data/VOC-YOLO when data/VOC_10+10 is missing;
#   - the base stage (task 1) is reused from an already trained checkpoint, never retrained;
#   - the ESPReg PCA cache of the base stage is rebuilt on the fly from that checkpoint.
#
# Usage (copy this file into the repository root or into scripts/, then run it):
#   bash run_lwf_espreg_10+10.sh --prepare-only      # pre-flight: build/verify split + base model
#   bash run_lwf_espreg_10+10.sh                     # full 100-epoch run + eval + txt summary
#   EPOCHS=1 DEVICE=1 bash run_lwf_espreg_10+10.sh   # quick smoke on another GPU
#
# Output: runs/yolo26m_VOC_10+10_pretrained-from-yolo26m-cls_lwf+espreg-raw
#         results/offline_voc_10+10_lwf+espreg.txt          <- copy this file back
set -euo pipefail

# Resolve the repository root: the directory that holds scripts/offline_experiment.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/scripts/offline_experiment.sh" ]]; do
    if [[ "$ROOT" == "/" ]]; then
        echo "Cannot locate the repository root (scripts/offline_experiment.sh not found)" >&2
        exit 1
    fi
    ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"

# The tools under tools/ import the `ultralytics` package; put the repository root on the
# import path so the run works without an editable install of the package.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- experiment configuration (identical to the local run of this cell) -------------------
export METHOD="lwf+espreg"
export ESPREG_WEIGHT_MODE="raw"        # raw eigenvalues: no elbow normalisation, no truncation
export ESPREG_FORM="frobenius"         # squared eigen-weighted projection (Tr(dW U diag(w) U^T dW^T))
export ESPREG_SCOPE="all"              # backbone + neck
export ESPREG_LOSS_WEIGHT="0.01"
export LWF_MODE="yolo_lwf"
export LWF_CLS_LOSS_WEIGHT="4000.0"
export LWF_REG_LOSS_WEIGHT="500.0"
export LWF_TEMPERATURE="2.0"
export LWF_WCE="True"
export LWF_CLS_IOU="True"
export LWF_MASK_OLD="True"

DATASET="voc"
SPLIT="10_10"
OUTPUT="runs/yolo26m_VOC_10+10_pretrained-from-yolo26m-cls_lwf+espreg-raw"
SUMMARY="results/offline_voc_10+10_lwf+espreg.txt"
BASE_MODEL="runs/yolo26m_VOC_10+10_pretrained-from-yolo26m-cls_naive/task-1/best.pt"

# ---- pass-through flags (--prepare-only, --train-base, ...) -------------------------------
EXTRA=()
for arg in "$@"; do
    EXTRA+=("$arg")
done

if [[ ! -f "$BASE_MODEL" ]]; then
    echo "Base-stage checkpoint not found: $BASE_MODEL" >&2
    echo "Copy it over, pass its path with --base-model, or add --train-base to train it here." >&2
    exit 1
fi

bash scripts/offline_experiment.sh \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --method "$METHOD" \
    --base-model "$BASE_MODEL" \
    --output "$OUTPUT" \
    --summary "$SUMMARY" \
    --skip-report \
    --epochs "${EPOCHS:-100}" \
    --batch "${BATCH_SIZE:-16}" \
    --workers "${WORKERS:-8}" \
    --device "${DEVICE:-0}" \
    "${EXTRA[@]}"
