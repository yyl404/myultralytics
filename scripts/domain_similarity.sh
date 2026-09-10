#!/bin/bash
# Image-domain feature-distribution similarity with pluggable feature
# extractors and distribution metrics (tools/domain_similarity.py).
# Datasets are given either as a yaml sequence (--tasks, each yaml
# contributes one split of images) or as image directories (--images).
#
# Usage:
#   bash scripts/domain_similarity.sh --tasks a.yaml b.yaml [c.yaml ...]
#   bash scripts/domain_similarity.sh --images DIR_A DIR_B [DIR_C ...]
#   bash scripts/domain_similarity.sh --tasks a.yaml b.yaml \
#       --features backbone+color_hist --metrics all --save-path out/sim.json
#
# Options:
#   --tasks yaml [yaml ...]   Dataset yaml sequence (mode 1; order = matrix order)
#   --images DIR [DIR ...]    Image directories (mode 2; mutually exclusive with --tasks)
#   --split auto|test|val|... Split to sample with --tasks (default: auto = test, else val)
#   --features SPEC           '+'-joined feature blocks (default: backbone):
#                             backbone, pixel_pca, color_hist, color_moments, lbp,
#                             hog, gabor, glcm, sift_bovw, dense_sift_bovw
#   --backbone/--weights SPEC Backbone for the 'backbone' block: local YOLO .pt path |
#                             torchvision:<name> | timm:<name> | bare timm name
#                             (default: torchvision:resnet50)
#   --metrics SPEC            '+'-joined metrics or 'all' (default: all)
#   --pca-dim N               Per-block PCA dim before concatenation (default: 0 = off)
#   --max-images N            Cap images per dataset (default: 0 = all)
#   --save-path FILE          Result JSON (default: runs/domain_similarity/<data-tag>.json);
#                             a <stem>_pairs.csv is written alongside
#
# Env: DEVICE / TOOL_DEVICE (single GPU, default 0), BATCH_SIZE, IMGSZ, SEED.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Image-domain feature-distribution similarity between datasets.

  bash scripts/domain_similarity.sh --tasks a.yaml b.yaml [--features SPEC] [--metrics SPEC] [--save-path FILE]
  bash scripts/domain_similarity.sh --images DIR_A DIR_B [--features SPEC] [--metrics SPEC] [--save-path FILE]
EOF
}

FEATURES="backbone"
BACKBONE="torchvision:resnet50"
METRICS="all"
SPLIT="auto"
PCA_DIM=0
MAX_IMAGES=0
SAVE_PATH=""
TASK_YAMLS=()
IMAGE_DIRS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --images)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            IMAGE_DIRS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --features)
            FEATURES="${2:?--features needs a value}"
            shift 2
            ;;
        --backbone|--weights)
            BACKBONE="${2:?--backbone needs a value}"
            shift 2
            ;;
        --metrics)
            METRICS="${2:?--metrics needs a value}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?--split needs a value}"
            shift 2
            ;;
        --pca-dim|--pca_dim)
            PCA_DIM="${2:?--pca-dim needs a value}"
            shift 2
            ;;
        --max-images|--max_images)
            MAX_IMAGES="${2:?--max-images needs a value}"
            shift 2
            ;;
        --save-path|--save_path)
            SAVE_PATH="${2:?--save-path needs a value}"
            shift 2
            ;;
        --*)
            experiment_die "Unknown option: $1"
            ;;
        *)
            experiment_die "Unexpected argument: $1"
            ;;
    esac
done

if (( ${#TASK_YAMLS[@]} > 0 && ${#IMAGE_DIRS[@]} > 0 )); then
    experiment_die "--tasks and --images are mutually exclusive"
fi
if (( ${#TASK_YAMLS[@]} > 0 )); then
    experiment_load_custom_tasks "${TASK_YAMLS[@]}"
    N_DATASETS=${#TASK_YAMLS[@]}
elif (( ${#IMAGE_DIRS[@]} >= 1 )); then
    for dir in "${IMAGE_DIRS[@]}"; do
        [[ -d "$dir" ]] || experiment_die "Image dir not found: $dir"
    done
    DATA_TAG="$(printf '%s\n' "${IMAGE_DIRS[@]}" | xargs -n1 basename | paste -sd+ -)"
    N_DATASETS=${#IMAGE_DIRS[@]}
else
    usage >&2
    experiment_die "Need --tasks <yaml...> or --images <DIR_A> <DIR_B> [DIR_C ...]"
fi
if [[ "$BACKBONE" == *.pt && ! -f "$BACKBONE" ]]; then
    experiment_die "Backbone weights not found: $BACKBONE"
fi

SAVE_PATH="${SAVE_PATH:-runs/domain_similarity/${DATA_TAG}.json}"
mkdir -p "$(dirname "$SAVE_PATH")"

# Single-GPU tool: first GPU of DEVICE unless TOOL_DEVICE overrides it.
DEVICE="${TOOL_DEVICE:-${DEVICE:-0}}"
DEVICE="${DEVICE%%,*}"

echo "=========================================="
echo "Domain similarity"
echo "  features: ${FEATURES}   backbone: ${BACKBONE}"
echo "  metrics : ${METRICS}"
echo "  datasets: ${N_DATASETS}  split=${SPLIT}  device=${DEVICE}"
echo "  output  : ${SAVE_PATH}"
echo "=========================================="

if (( ${#TASK_YAMLS[@]} > 0 )); then
    python tools/domain_similarity.py \
        --data "${TASK_DATASETS[@]}" \
        --split "$SPLIT" \
        --features "$FEATURES" \
        --backbone "$BACKBONE" \
        --metrics "$METRICS" \
        --pca-dim "$PCA_DIM" \
        --max-images "$MAX_IMAGES" \
        --batch "${BATCH_SIZE:-16}" \
        --imgsz "${IMGSZ:-0}" \
        --device "$DEVICE" \
        --seed "${SEED:-0}" \
        --save-path "$SAVE_PATH"
else
    python tools/domain_similarity.py \
        --images "${IMAGE_DIRS[@]}" \
        --features "$FEATURES" \
        --backbone "$BACKBONE" \
        --metrics "$METRICS" \
        --pca-dim "$PCA_DIM" \
        --max-images "$MAX_IMAGES" \
        --batch "${BATCH_SIZE:-16}" \
        --imgsz "${IMGSZ:-0}" \
        --device "$DEVICE" \
        --seed "${SEED:-0}" \
        --save-path "$SAVE_PATH"
fi
