#!/bin/bash
# Demo predict stage: labeled inference with PREDICT_MODEL on the task datasets
# and the cumulative datasets (test split, val if absent), in the same order as
# eval. Produces RUN_DIR/predictions (per-dataset metrics.csv plus TP/FP/FN
# visualizations). Tunables live in common.sh.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up from this script to the repo root (the dir holding scripts/train.sh),
# so this demo works when copied to any depth under the repo.
REPO_ROOT="$DEMO_DIR"
while [[ ! -f "$REPO_ROOT/scripts/train.sh" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -f "$REPO_ROOT/scripts/train.sh" ]]; then
    echo "Cannot locate the repository root above $DEMO_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT"
source "$DEMO_DIR/common.sh"

if [[ ! -f "$PREDICT_MODEL" ]]; then
    echo "Model checkpoint not found: $PREDICT_MODEL" >&2
    echo "Train first: bash $DEMO_DIR/train.sh" >&2
    exit 1
fi

bash scripts/predict.sh \
    --model "$PREDICT_MODEL" \
    --tasks "${TASK_YAMLS[@]}" \
    --cumulative "${CUMULATIVE_YAMLS[@]}" \
    --save-path "$RUN_DIR/predictions" \
    -- \
    "${EXTRA_PREDICT_ARGS[@]}"
