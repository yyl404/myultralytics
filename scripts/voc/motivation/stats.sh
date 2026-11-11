#!/bin/bash
# Motivation statistics stage: per-stage class object counts and stage sample
# overlap over the full class-space annotation. Runs standalone (no trained
# checkpoint needed). Tunables live in common.sh.

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

stats_protocol() {
    local tag="$1" run_dir="$2"
    shift 2
    local yamls=("$@")
    local yaml out_dir

    for yaml in "${yamls[@]}"; do
        if [[ ! -f "$yaml" ]]; then
            echo "Stage yaml not found: $yaml" >&2
            echo "Create the dataset first with bash $DEMO_DIR/create.sh, or fix common.sh" >&2
            exit 1
        fi
    done
    out_dir="${run_dir}/stage_stats"
    mkdir -p "$out_dir"
    echo "=========================================="
    echo "Motivation stage statistics: ${tag}"
    echo "  stages : ${#yamls[@]}"
    echo "  split  : ${ANALYSIS_SPLIT}"
    echo "  output : ${out_dir}"
    echo "=========================================="
    bash scripts/analyze.sh stage_stats \
        --stages "${yamls[@]}" \
        --source_cfg "$SOURCE_CFG" \
        --split "$ANALYSIS_SPLIT" \
        --output_dir "$out_dir"
}

stats_two_stage() {
    stats_protocol "$TWO_STAGE_TAG" "$RUN_DIR_TWO" "${TWO_STAGE_YAMLS[@]}"
}

stats_four_stage() {
    stats_protocol "$FOUR_STAGE_TAG" "$RUN_DIR_FOUR" "${FOUR_STAGE_YAMLS[@]}"
}

case "$PROTOCOL" in
    two)
        stats_two_stage
        ;;
    four)
        stats_four_stage
        ;;
    all)
        stats_two_stage
        stats_four_stage
        ;;
    *)
        echo "Unknown PROTOCOL '$PROTOCOL' (expected two, four, or all)" >&2
        exit 1
        ;;
esac
