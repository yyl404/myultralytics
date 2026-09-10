#!/bin/bash
# Demo similarity stage: pairwise image-domain feature-distribution
# similarity across the TRAIN_YAMLS x EVAL_YAMLS sequences (the deduplicated
# union matrix covers every train x eval pair), computed for EVERY feature
# extractor x every compatible distribution metric
# (scripts/domain_similarity.sh -> tools/domain_similarity.py).
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

# Cross the train and eval sequences: dedup preserving order, so the union
# matrix covers every train x eval pair exactly once.
SIM_YAMLS=()
declare -A SEEN_YAMLS=()
for yaml in "${TRAIN_YAMLS[@]}" "${EVAL_YAMLS[@]}"; do
    if [[ -z "${SEEN_YAMLS[$yaml]:-}" ]]; then
        SEEN_YAMLS[$yaml]=1
        SIM_YAMLS+=("$yaml")
    fi
done

for yaml in "${SIM_YAMLS[@]}"; do
    if [[ ! -f "$yaml" ]]; then
        echo "Dataset yaml not found: $yaml" >&2
        echo "Create the dataset first, or fix TRAIN_YAMLS / EVAL_YAMLS in common.sh" >&2
        exit 1
    fi
done

# Every available feature block, one run per block with --metrics all.
# Single histogram blocks (color_hist, lbp, *_bovw) additionally get the
# histogram-only metrics (js_divergence, hist_intersection, chi2_distance).
# Override for one launch with SIMILARITY_FEATURES="backbone color_hist".
SIMILARITY_FEATURES="${SIMILARITY_FEATURES:-backbone pixel_pca color_hist color_moments lbp hog gabor glcm sift_bovw dense_sift_bovw}"
OUT_DIR="runs/domain_similarity/${DATA_TAG}"

for feature in $SIMILARITY_FEATURES; do
    # hog raw dim (8100) exceeds the frechet DxD covariance limit; reduce
    # per-block before the metrics (mean_cosine still uses raw features).
    PCA_DIM=0
    if [[ "$feature" == "hog" ]]; then
        PCA_DIM="${SIMILARITY_PCA_DIM:-256}"
    fi
    echo "=== similarity: feature=${feature} metrics=all -> ${OUT_DIR}/${feature}.json ==="
    bash scripts/domain_similarity.sh \
        --tasks "${SIM_YAMLS[@]}" \
        --features "$feature" \
        --metrics all \
        --backbone "$SIMILARITY_WEIGHTS" \
        --pca-dim "$PCA_DIM" \
        --save-path "${OUT_DIR}/${feature}.json"
done
