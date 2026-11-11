#!/bin/bash
# Motivation dataset stage: build the two-stage and multi-stage class-incremental
# splits if they are missing. Each stage dataset holds every sample containing at
# least one class of that stage and has its own train / val / test split.
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

create_split() {
    local data_root="$1" split_id="$2" first_yaml="$3" tag="$4"
    if [[ -f "$first_yaml" ]]; then
        echo "Dataset already present: ${data_root} (${tag})"
        return
    fi
    if [[ -e "$data_root" ]]; then
        echo "Dataset directory exists but ${first_yaml} is missing: ${data_root}" >&2
        echo "Remove ${data_root} or rebuild it with scripts/create.sh before rerunning." >&2
        exit 1
    fi
    bash scripts/create.sh voc "$split_id"
}

create_split "$TWO_STAGE_ROOT" "$TWO_STAGE_SPLIT_ID" "${TWO_STAGE_YAMLS[0]}" "$TWO_STAGE_TAG"
create_split "$FOUR_STAGE_ROOT" "$FOUR_STAGE_SPLIT_ID" "${FOUR_STAGE_YAMLS[0]}" "$FOUR_STAGE_TAG"
