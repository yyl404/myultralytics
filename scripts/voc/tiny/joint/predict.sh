#!/bin/bash
# Demo predict stage: labeled inference with PREDICT_MODEL on the independent
# eval sequence (EVAL_YAMLS) and the cumulative eval sequence
# (CUMULATIVE_YAMLS, skipped when empty), in the same order as eval; test
# split, val if absent. Produces RUN_DIR/predictions (per-dataset metrics.csv
# plus TP/FP/FN visualizations). Tunables live in common.sh.

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

predict_args=(--model "$PREDICT_MODEL" --tasks "${EVAL_YAMLS[@]}" --save-path "$RUN_DIR/predictions")
if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
    predict_args+=(--cumulative "${CUMULATIVE_YAMLS[@]}")
fi
bash scripts/predict.sh "${predict_args[@]}" -- "${EXTRA_PREDICT_ARGS[@]}"
