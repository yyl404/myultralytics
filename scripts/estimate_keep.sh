#!/bin/bash
# Estimate the lowest i.i.d. keep ratio from the train-split feature distribution.
# Does not train. Inputs are the full dataset yaml and the pretrained backbone.
#
# Usage:
#   bash scripts/estimate_keep.sh --src_data data/VOC-YOLO/VOC.yaml
#
# Migration: change only these flags.
#   --src_data YAML          full saturated dataset yaml (required)
#   --split NAME             split to measure (default: train)
#   --backbone SPEC          pretrained weights used as the feature map
#                            (default: yoloe-26m-seg.pt, same as YOLO26m fine-tuning)
#   --out_dir DIR            knee.json and coverage.csv
#                            (default: runs/keep_ratio/<dataset-dir-name>)
#   --feature_cache_dir DIR  visual-feature cache (default: <out_dir>/feature_cache)
#
# Env: DEVICE (default 0).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Estimate the i.i.d. learning-curve knee from backbone features (no training).

  bash scripts/estimate_keep.sh --src_data data/VOC-YOLO/VOC.yaml

Options:
  --src_data YAML          Source dataset yaml (required)
  --split NAME             Split to measure (default: train)
  --backbone SPEC          Pretrained backbone (default: yoloe-26m-seg.pt)
  --out_dir DIR            knee.json and coverage.csv
  --feature_cache_dir DIR  Feature cache directory
EOF
}

SRC_DATA=""
SPLIT="train"
BACKBONE="yoloe-26m-seg.pt"
OUT_DIR=""
FEATURE_CACHE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --src_data)
            SRC_DATA="${2:?}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?}"
            shift 2
            ;;
        --backbone)
            BACKBONE="${2:?}"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="${2:?}"
            shift 2
            ;;
        --feature_cache_dir)
            FEATURE_CACHE_DIR="${2:?}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -n "$SRC_DATA" ]] || {
    usage >&2
    echo "Need --src_data" >&2
    exit 1
}
[[ -f "$SRC_DATA" ]] || {
    echo "Source yaml not found: $SRC_DATA" >&2
    exit 1
}
if [[ -z "$OUT_DIR" ]]; then
    tag="$(basename "$(dirname "$SRC_DATA")")"
    OUT_DIR="runs/keep_ratio/${tag}"
fi
if [[ -z "$FEATURE_CACHE_DIR" ]]; then
    FEATURE_CACHE_DIR="${OUT_DIR}/feature_cache"
fi

mkdir -p "$OUT_DIR" "$FEATURE_CACHE_DIR"
python tools/estimate_keep_ratio.py \
    --src_data "$SRC_DATA" \
    --split "$SPLIT" \
    --backbone "$BACKBONE" \
    --device "${DEVICE:-0}" \
    --feature_cache_dir "$FEATURE_CACHE_DIR" \
    --json_out "${OUT_DIR}/knee.json" \
    --curve_csv "${OUT_DIR}/coverage.csv"
