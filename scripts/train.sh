#!/bin/bash
# Unified incremental train entry: an explicit task yaml sequence x model x IOD method.
#
# Usage:
#   bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26m --method naive
#   bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26 --method pseudo_label+dist+espreg
#
# Options:
#   --tasks yaml [yaml ...]              Incremental train yaml sequence, one yaml per task
#                                        (a single comma-separated argument also works)
#   --model --method                     Model family (optional size suffix) and method
#                                        components joined with '+'; also 2 positionals
#   --tag NAME                           Override the auto-derived DATA_TAG (run naming)
#   --size n|s|m|l|x                     Override default size (yoloe-v8: l, else x)
#   --weights FILE                       Override default init weights
#   --from-scratch                       Train without pretrained weights
#   --output DIR                         Override runs/<model>_<data>_..._<method>
#   --                                Extra flags forwarded to tools/train.py
#
# Training is fully decoupled from evaluation: it only consumes the train yaml
# sequence and produces task-1, task-2, ... under the output directory. Evaluate
# afterwards with scripts/eval.sh on any yaml sequence.
#
# Env: EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE, START_TASK, END_TASK,
#      DIST_LOSS_WEIGHT, DIST_TOPK, ESPREG_LOSS_WEIGHT,
#      YOLO26_DEFAULT_HYPS=0 to disable yolo26 hyps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental train: task yaml sequence x model x IOD method.

  bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26m --method naive

Options:
  --tasks yaml [yaml ...]              Incremental train yaml sequence (required)
  --model --method                     Model and method (also 2 positionals)
  --tag NAME                           Override the auto-derived DATA_TAG
  --size n|s|m|l|x                     Override default size (yoloe-v8: l, else x)
  --weights FILE                       Override default init weights
  --from-scratch                       Train without pretrained weights
  --output DIR                         Override the default runs/ directory
  --                                   Extra flags forwarded to tools/train.py

Env: EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE, START_TASK, END_TASK,
     DIST_LOSS_WEIGHT, DIST_TOPK, ESPREG_LOSS_WEIGHT,
     YOLO26_DEFAULT_HYPS=0 to disable yolo26 hyps.
EOF
    echo ""
    echo "Models:    $(experiment_known_model_families)  (optional size suffix, e.g. yolo26m)"
    echo "Methods:   +joined components: $(experiment_known_method_components)"
}

MODEL_SPEC=""
METHOD=""
FROM_SCRATCH=""
OUTPUT_DIR_OVERRIDE=""
MODEL_SIZE_FLAG=""
DATA_TAG_OVERRIDE=""
TASK_YAMLS=()
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --model)
            MODEL_SPEC="${2:?--model needs a value}"
            shift 2
            ;;
        --method)
            METHOD="${2:?--method needs a value}"
            shift 2
            ;;
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            ;;
        --tag)
            DATA_TAG_OVERRIDE="${2:?--tag needs a value}"
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

MODEL_SPEC="${MODEL_SPEC:-${POSITIONAL[0]:-}}"
METHOD="${METHOD:-${POSITIONAL[1]:-}}"
(( ${#POSITIONAL[@]} <= 2 )) || experiment_die "Unexpected extra arguments: ${POSITIONAL[*]:2}"

(( ${#TASK_YAMLS[@]} > 0 )) || {
    usage >&2
    experiment_die "Need --tasks <yaml...>"
}
[[ -n "$MODEL_SPEC" && -n "$METHOD" ]] || {
    usage >&2
    experiment_die "Need model and method"
}

experiment_load_custom_tasks "${TASK_YAMLS[@]}"
experiment_apply_run_defaults
if [[ -n "$MODEL_SIZE_FLAG" ]]; then
    experiment_parse_model_spec "$MODEL_SPEC"
    MODEL_SPEC="${MODEL_FAMILY}${MODEL_SIZE_FLAG}"
fi
experiment_load_model "$MODEL_SPEC"
if (( ${#PASSTHROUGH[@]} > 0 )); then
    EXTRA_TRAIN_ARGS+=("${PASSTHROUGH[@]}")
fi
experiment_set_output_paths "$METHOD"
# Absolute paths: a relative ultralytics --project would be re-rooted under runs/detect/.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "=========================================="
echo "Incremental train"
echo "  data    : ${DATA_TAG}  (${#TASK_DATASETS[@]} tasks)"
echo "  model   : ${MODEL_ID}  config=${MODEL_CONFIG}  weights=${MODEL_WEIGHTS:-<from scratch>}"
echo "  method  : ${METHOD}"
echo "  output  : ${OUTPUT_DIR}"
echo "  epochs  : ${EPOCHS}  batch=${BATCH_SIZE}  device=${DEVICE}"
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    echo "  extra   : ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "=========================================="

source scripts/run_incremental.sh
