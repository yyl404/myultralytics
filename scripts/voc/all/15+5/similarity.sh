#!/bin/bash
# Demo similarity stage: N x N image-domain deep-feature similarity matrix
# across the task yaml sequence, computed with a frozen pretrained backbone.
# Runs standalone (no trained checkpoint needed). Tunables live in common.sh.

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

for yaml in "${TASK_YAMLS[@]}"; do
    if [[ ! -f "$yaml" ]]; then
        echo "Task yaml not found: $yaml" >&2
        echo "Create the dataset first, or fix TASK_YAMLS in common.sh" >&2
        exit 1
    fi
done

bash scripts/dataset_similarity.sh \
    --tasks "${TASK_YAMLS[@]}" \
    --weights "$SIMILARITY_WEIGHTS"
