#!/bin/bash
# Dataset pruning: keep a ratio or an absolute count of existing samples.
#
# Usage:
#   bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned --ratio 0.1
#   bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned-500 --num_out 500
#
# Options:
#   --src_data YAML          Source dataset yaml (required)
#   --out_dir DIR            Output dataset directory (required)
#   --ratio FLOAT            Keep ratio per compressed split, in (0, 1]
#   --num_out N              Absolute keep count per split (overrides --ratio)
#   --splits a b c           Splits to write (default: train val test)
#   --random_splits a b      Among --splits, uniformly sampled (default: val test)
#   --seed N                 RNG seed (default: 0)
#   --backbone SPEC          Visual backbone or 'none' (default: none)
#   --feature_cache_dir DIR  Per-split visual feature cache directory
#   --                       Extra flags forwarded to tools/prune_dataset.py
#
# Env: DEVICE (feature extraction GPU, default 0).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Prune a YOLO dataset to a keep ratio or an absolute count of existing samples.

  bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned --ratio 0.1
  bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned-500 --num_out 500

Options:
  --src_data YAML          Source dataset yaml (required)
  --out_dir DIR            Output dataset directory (required)
  --ratio FLOAT            Keep ratio per compressed split
  --num_out N              Absolute keep count per split (overrides --ratio)
  --splits a b c           Splits to write (default: train val test)
  --random_splits a b      Uniformly sampled splits (default: val test)
  --seed N                 RNG seed (default: 0)
  --backbone SPEC          Visual backbone or 'none'
  --                       Extra flags forwarded to tools/prune_dataset.py

Env: DEVICE (default 0).
EOF
}

SRC_DATA=""
OUT_DIR=""
RATIO=""
NUM_OUT=""
SEED="0"
BACKBONE="none"
SPLITS=()
RANDOM_SPLITS=()
RANDOM_SPLITS_SET=0
PASSTHROUGH=()

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
        --out_dir)
            OUT_DIR="${2:?}"
            shift 2
            ;;
        --ratio)
            RATIO="${2:?}"
            shift 2
            ;;
        --num_out)
            NUM_OUT="${2:?}"
            shift 2
            ;;
        --splits)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SPLITS+=("$1")
                shift
            done
            ;;
        --random_splits)
            RANDOM_SPLITS_SET=1
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RANDOM_SPLITS+=("$1")
                shift
            done
            ;;
        --seed)
            SEED="${2:?}"
            shift 2
            ;;
        --backbone)
            BACKBONE="${2:?}"
            shift 2
            ;;
        --feature_cache_dir)
            FEATURE_CACHE_DIR="${2:?}"
            shift 2
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -n "$SRC_DATA" && -n "$OUT_DIR" ]] || {
    usage >&2
    echo "Need --src_data and --out_dir" >&2
    exit 1
}
[[ -n "$RATIO" || -n "$NUM_OUT" ]] || {
    usage >&2
    echo "Need --ratio or --num_out" >&2
    exit 1
}

cmd=(python tools/prune_dataset.py --src_data "$SRC_DATA" --out_dir "$OUT_DIR" --seed "$SEED" --backbone "$BACKBONE" --device "${DEVICE:-0}")
if [[ -n "${FEATURE_CACHE_DIR:-}" ]]; then
    cmd+=(--feature_cache_dir "$FEATURE_CACHE_DIR")
fi
if [[ -n "$NUM_OUT" ]]; then
    cmd+=(--num_out "$NUM_OUT")
else
    cmd+=(--ratio "$RATIO")
fi
if (( ${#SPLITS[@]} > 0 )); then
    cmd+=(--splits "${SPLITS[@]}")
fi
if (( RANDOM_SPLITS_SET )); then
    cmd+=(--random_splits "${RANDOM_SPLITS[@]}")
fi
if (( ${#PASSTHROUGH[@]} > 0 )); then
    cmd+=("${PASSTHROUGH[@]}")
fi
"${cmd[@]}"
