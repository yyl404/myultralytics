#!/bin/bash
# Incremental inference: run tools/detect.py on each task yaml (registered split or --tasks).
#
# Usage:
#   bash scripts/detect.sh --dataset voc-tiny --split 15_5 --model runs/<run>/task-2/best.pt
#   bash scripts/detect.sh voc-tiny 15_5 --model runs/<run>/task-2/best.pt
#   bash scripts/detect.sh --tasks t1.yaml t2.yaml --model runs/<run>/task-2/best.pt
#   bash scripts/detect.sh --run runs/<run> --model runs/<run>/task-2/best.pt
#
# Class IDs are aligned to the model (same conversion as eval) before dumping detections.
# Extra flags after -- are forwarded to tools/detect.py (e.g. -- --conf 0.25 --imgsz 640).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Incremental inference over a yaml sequence. Dumps boxes and GT-match folders per task.

  bash scripts/detect.sh --dataset voc-tiny --split 15_5 --model runs/<run>/task-2/best.pt
  bash scripts/detect.sh voc-tiny 15_5 --model runs/<run>/task-2/best.pt
  bash scripts/detect.sh --tasks t1.yaml t2.yaml --model runs/<run>/task-2/best.pt
  bash scripts/detect.sh --run runs/<run> --model runs/<run>/task-2/best.pt

Options:
  --dataset --split     Registered experiment identity
  --tasks yaml [yaml ...]
  --run DIR             Infer yaml sequence from the run manifest / folder name
  --model FILE          Detector checkpoint (.pt)
  --save-path DIR       Root output dir (default: <model-dir>/detect)
  --pred-split val|test|train   Image split forwarded to detect.py (default: val)
  --                      Extra flags forwarded to tools/detect.py

When --tasks/--dataset/--split are omitted, --run (or a positional run dir) is used
to recover the yaml sequence written at train time.
EOF
}

DATASET=""
SPLIT=""
MODEL=""
SAVE_PATH=""
PRED_SPLIT="val"
RUN_DIR=""
DATA_TAG_OVERRIDE=""
TASK_YAMLS=()
EVAL_YAMLS=()
CUMULATIVE_YAMLS=()
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --dataset)
            DATASET="${2:?}"
            shift 2
            ;;
        --split)
            SPLIT="${2:?}"
            shift 2
            ;;
        --model)
            MODEL="${2:?}"
            shift 2
            ;;
        --save-path|--save_path)
            SAVE_PATH="${2:?}"
            shift 2
            ;;
        --pred-split|--pred_split)
            PRED_SPLIT="${2:?}"
            shift 2
            ;;
        --run)
            RUN_DIR="${2:?}"
            shift 2
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#TASK_YAMLS[@]} > 0 )) || experiment_die "--tasks needs at least one yaml"
            ;;
        --tag)
            DATA_TAG_OVERRIDE="${2:?}"
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

pos_i=0
if (( ${#TASK_YAMLS[@]} > 0 )); then
    if [[ -z "$MODEL" && $pos_i -lt ${#POSITIONAL[@]} ]]; then
        MODEL="${POSITIONAL[pos_i]}"
        pos_i=$((pos_i + 1))
    fi
    if [[ -z "$SAVE_PATH" && $pos_i -lt ${#POSITIONAL[@]} ]]; then
        SAVE_PATH="${POSITIONAL[pos_i]}"
        pos_i=$((pos_i + 1))
    fi
    (( pos_i == ${#POSITIONAL[@]} )) || experiment_die "Unexpected extra arguments with --tasks: ${POSITIONAL[*]:pos_i}"
elif [[ -n "$DATASET" || -n "$SPLIT" ]]; then
    DATASET="${DATASET:-${POSITIONAL[0]:-}}"
    SPLIT="${SPLIT:-${POSITIONAL[1]:-}}"
    MODEL="${MODEL:-${POSITIONAL[2]:-}}"
    SAVE_PATH="${SAVE_PATH:-${POSITIONAL[3]:-}}"
elif (( ${#POSITIONAL[@]} >= 2 )) && [[ "${POSITIONAL[1]}" =~ ^[0-9]+(_[0-9]+)*$ ]]; then
    DATASET="${POSITIONAL[0]}"
    SPLIT="${POSITIONAL[1]}"
    MODEL="${MODEL:-${POSITIONAL[2]:-}}"
    SAVE_PATH="${SAVE_PATH:-${POSITIONAL[3]:-}}"
else
    RUN_DIR="${RUN_DIR:-${POSITIONAL[0]:-}}"
    MODEL="${MODEL:-${POSITIONAL[1]:-}}"
    SAVE_PATH="${SAVE_PATH:-${POSITIONAL[2]:-}}"
fi

[[ -n "$MODEL" ]] || {
    usage >&2
    experiment_die "Need --model"
}
[[ -f "$MODEL" ]] || experiment_die "Model checkpoint not found: $MODEL"

if (( ${#TASK_YAMLS[@]} > 0 )) || [[ -n "$DATASET" && -n "$SPLIT" ]]; then
    experiment_resolve_dataset
else
    [[ -n "$RUN_DIR" ]] || {
        usage >&2
        experiment_die "Need --dataset/--split, --tasks, or --run"
    }
    experiment_resolve_eval_dataset "$RUN_DIR"
fi

SAVE_PATH="${SAVE_PATH:-$(dirname "$MODEL")/detect}"
mkdir -p "$SAVE_PATH"

echo "=========================================="
echo "Incremental detect (${DATA_TAG}, ${INCREMENTAL_SETTING})"
echo "  model   : ${MODEL}"
echo "  tasks   : ${#TASK_DATASETS[@]}"
echo "  split   : ${PRED_SPLIT}"
echo "  output  : ${SAVE_PATH}"
echo "=========================================="

for task_index in "${!TASK_DATASETS[@]}"; do
    TASK_ID=$((task_index + 1))
    DATASET_PATH="${TASK_DATASETS[$task_index]}"
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Warning: Dataset not found: $DATASET_PATH" >&2
        continue
    fi
    TASK_OUT="${SAVE_PATH}/task-${TASK_ID}"
    CONVERTED_DIR="${TASK_OUT}/converted"
    mkdir -p "$TASK_OUT"
    python tools/convert_dataset_class_ids.py \
        --model "$MODEL" --dataset "$DATASET_PATH" \
        --output_dir "$CONVERTED_DIR" --splits "$PRED_SPLIT"
    detect_args=(
        python tools/detect.py
        --model "$MODEL"
        --data "${CONVERTED_DIR}/dataset.yaml"
        --split "$PRED_SPLIT"
        --save_path "$TASK_OUT"
        --project "$TASK_OUT"
        --device "${DEVICE:-0}"
    )
    if (( ${#PASSTHROUGH[@]} > 0 )); then
        detect_args+=("${PASSTHROUGH[@]}")
    fi
    "${detect_args[@]}"
done
