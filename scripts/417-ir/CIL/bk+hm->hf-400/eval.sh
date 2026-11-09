#!/bin/bash
# Demo eval stage: evaluate every task-k/best.pt under RUN_DIR on the
# independent eval sequence (EVAL_YAMLS) and the cumulative eval sequence
# (CUMULATIVE_YAMLS, skipped when empty); test split, val if absent.
# Produces RUN_DIR/evaluation_results. Tunables live in common.sh.

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

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Run directory not found: $RUN_DIR" >&2
    echo "Train first: bash $DEMO_DIR/train.sh" >&2
    exit 1
fi

eval_args=(--run "$RUN_DIR" --tasks "${EVAL_YAMLS[@]}")
if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
    eval_args+=(--cumulative "${CUMULATIVE_YAMLS[@]}")
fi
bash scripts/eval.sh "${eval_args[@]}" -- "${EXTRA_EVAL_ARGS[@]}"
