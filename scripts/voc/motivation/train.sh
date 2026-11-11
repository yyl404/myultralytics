#!/bin/bash
# Motivation train stage: incremental training on the configured stage yaml
# sequences. Produces task-1, task-2, ... under each protocol's run directory.
# Tunables live in common.sh.

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

train_protocol() {
    local tag="$1" run_dir="$2"
    shift 2
    local yamls=("$@")
    local yaml

    for yaml in "${yamls[@]}"; do
        if [[ ! -f "$yaml" ]]; then
            echo "Train yaml not found: $yaml" >&2
            echo "Create the dataset first with bash $DEMO_DIR/create.sh, or fix common.sh" >&2
            exit 1
        fi
    done
    echo "=========================================="
    echo "Motivation train: ${tag}"
    echo "  stages : ${#yamls[@]}"
    echo "  run    : ${run_dir}"
    echo "=========================================="
    bash scripts/train.sh \
        --tasks "${yamls[@]}" \
        --model "$MODEL" \
        --method "$METHOD" \
        --weights "$WEIGHTS" \
        --tag "$tag" \
        --output "$run_dir" \
        -- \
        "${EXTRA_TRAIN_ARGS[@]}"
}

train_two_stage() {
    train_protocol "$TWO_STAGE_TAG" "$RUN_DIR_TWO" "${TWO_STAGE_YAMLS[@]}"
}

train_four_stage() {
    train_protocol "$FOUR_STAGE_TAG" "$RUN_DIR_FOUR" "${FOUR_STAGE_YAMLS[@]}"
}

case "$PROTOCOL" in
    two)
        train_two_stage
        ;;
    four)
        train_four_stage
        ;;
    all)
        train_two_stage
        train_four_stage
        ;;
    *)
        echo "Unknown PROTOCOL '$PROTOCOL' (expected two, four, or all)" >&2
        exit 1
        ;;
esac
