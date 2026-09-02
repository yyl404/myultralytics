#!/bin/bash
# Unified incremental train entry: any dataset × model × IOD method.
#
# Usage:
#   bash scripts/train.sh --dataset voc-tiny --split 15_5 --model yolo26 --method pseudo_label+dist+espreg
#   bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg
#
# Options:
#   --dataset --split --model --method   Identity of the run (also accepted as 4 positionals)
#   --size n|s|m|l|x                     Override default size (voc-tiny: m, yoloe-v8: l, else x)
#   --weights FILE                       Override default init weights
#   --from-scratch                       Train without pretrained weights
#   --output DIR                         Override runs/<model>_<data>_..._<method>
#   --                                Extra flags forwarded to tools/train.py
#
# Env (same as before): EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE, START_TASK, END_TASK,
# DIST_LOSS_WEIGHT, DIST_TOPK, ESPREG_LOSS_WEIGHT, YOLO26_DEFAULT_HYPS=0 to disable yolo26 hyps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=lib/experiment.sh
source scripts/lib/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental train: any dataset × model × IOD method.

  bash scripts/train.sh --dataset voc-tiny --split 15_5 --model yolo26 --method pseudo_label+dist+espreg
  bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg

Options:
  --dataset --split --model --method   Identity of the run (also 4 positionals)
  --size n|s|m|l|x                     Override default size (voc-tiny: m, yoloe-v8: l, else x)
  --weights FILE                       Override default init weights
  --from-scratch                       Train without pretrained weights
  --output DIR                         Override the default runs/ directory
  --                                   Extra flags forwarded to tools/train.py

Env: EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE, START_TASK, END_TASK,
     DIST_LOSS_WEIGHT, DIST_TOPK, ESPREG_LOSS_WEIGHT,
     YOLO26_DEFAULT_HYPS=0 to disable yolo26 hyps.
EOF
    echo ""
    echo "Datasets:  $(experiment_known_datasets)"
    echo "Models:    $(experiment_known_model_families)  (optional size suffix, e.g. yolo26m)"
    echo "Methods:   +joined components: $(experiment_known_method_components)"
}

DATASET=""
SPLIT=""
MODEL_SPEC=""
METHOD=""
FROM_SCRATCH=""
OUTPUT_DIR_OVERRIDE=""
MODEL_SIZE_FLAG=""
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --dataset)
            DATASET="${2:?--dataset needs a value}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?--split needs a value}"
            shift 2
            ;;
        --model)
            MODEL_SPEC="${2:?--model needs a value}"
            shift 2
            ;;
        --method)
            METHOD="${2:?--method needs a value}"
            shift 2
            ;;
        --size)
            MODEL_SIZE_FLAG="${2:?--size needs a value}"
            shift 2
            ;;
        --weights)
            [[ $# -ge 2 ]] || experiment_die "--weights needs a value (use --from-scratch for none)"
            MODEL_WEIGHTS_OVERRIDE="$2"
            shift 2
            ;;
        --from-scratch)
            FROM_SCRATCH=1
            shift
            ;;
        --output)
            OUTPUT_DIR_OVERRIDE="${2:?--output needs a value}"
            shift 2
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        --*)
            experiment_die "Unknown option: $1"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL[@]} > 0 )); then
    DATASET="${DATASET:-${POSITIONAL[0]:-}}"
    SPLIT="${SPLIT:-${POSITIONAL[1]:-}}"
    MODEL_SPEC="${MODEL_SPEC:-${POSITIONAL[2]:-}}"
    METHOD="${METHOD:-${POSITIONAL[3]:-}}"
    (( ${#POSITIONAL[@]} <= 4 )) || experiment_die "Unexpected extra arguments: ${POSITIONAL[*]:4}"
fi

[[ -n "$DATASET" && -n "$SPLIT" && -n "$MODEL_SPEC" && -n "$METHOD" ]] || {
    usage >&2
    experiment_die "Need dataset, split, model, and method"
}

experiment_load_dataset "$DATASET" "$SPLIT"
if [[ -n "$MODEL_SIZE_FLAG" ]]; then
    experiment_parse_model_spec "$MODEL_SPEC"
    MODEL_SPEC="${MODEL_FAMILY}${MODEL_SIZE_FLAG}"
fi
experiment_load_model "$MODEL_SPEC"
if (( ${#PASSTHROUGH[@]} > 0 )); then
    EXTRA_TRAIN_ARGS+=("${PASSTHROUGH[@]}")
fi
experiment_set_output_paths "$METHOD"

echo "=========================================="
echo "Incremental train"
echo "  dataset : ${DATASET}/${SPLIT}  (${DATA_TAG}, ${INCREMENTAL_SETTING})"
echo "  model   : ${MODEL_ID}  config=${MODEL_CONFIG}  weights=${MODEL_WEIGHTS:-<from scratch>}"
echo "  method  : ${METHOD}"
echo "  output  : ${OUTPUT_DIR}"
echo "  epochs  : ${EPOCHS}  batch=${BATCH_SIZE}  device=${DEVICE}"
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    echo "  extra   : ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "=========================================="

source scripts/run_incremental.sh
