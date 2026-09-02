#!/bin/bash
# Unified incremental train entry: any dataset × model × IOD method, or an
# explicit yaml sequence via --tasks.
#
# Usage:
#   bash scripts/train.sh --dataset voc-tiny --split 15_5 --model yolo26 --method pseudo_label+dist+espreg
#   bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg
#   bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26 --method naive
#
# Options:
#   --dataset --split --model --method   Identity of the run (dataset/split/model/method
#                                        also accepted as 4 positionals; with --tasks the
#                                        positionals are just model and method)
#   --tasks yaml [yaml ...]              Explicit incremental train yaml sequence
#                                        (replaces --dataset/--split)
#   --eval-tasks yaml [yaml ...]         Per-task eval yamls (default: the train sequence)
#   --cumulative yaml [yaml ...]         Cumulative eval yamls, one per task (optional;
#                                        omitting it disables cumulative evaluation)
#   --tag NAME                           Override the auto-derived DATA_TAG (run naming)
#   --size n|s|m|l|x                     Override default size (voc-tiny: m, yoloe-v8: l, else x)
#   --weights FILE                       Override default init weights
#   --from-scratch                       Train without pretrained weights
#   --output DIR                         Override runs/<model>_<data>_..._<method>
#   --                                Extra flags forwarded to tools/train.py
#
# The resolved yaml sequences are written to <output>/task_yamls.txt (+ eval_yamls.txt,
# cumulative_yamls.txt, experiment.meta) so eval.sh / detect.sh can recover them.
#
# Env (same as before): EPOCHS, BATCH_SIZE, IMGSZ, WORKERS, DEVICE, START_TASK, END_TASK,
# DIST_LOSS_WEIGHT, DIST_TOPK, ESPREG_LOSS_WEIGHT, YOLO26_DEFAULT_HYPS=0 to disable yolo26 hyps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=libexec/experiment.sh
source scripts/libexec/experiment.sh

usage() {
    cat <<'EOF'
Unified incremental train: any dataset × model × IOD method.

  bash scripts/train.sh --dataset voc-tiny --split 15_5 --model yolo26 --method pseudo_label+dist+espreg
  bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg
  bash scripts/train.sh --tasks t1.yaml t2.yaml --model yolo26 --method naive

Options:
  --dataset --split --model --method   Identity of the run (also 4 positionals;
                                       with --tasks the positionals are model method)
  --tasks yaml [yaml ...]              Explicit incremental train yaml sequence
                                       (replaces --dataset/--split)
  --eval-tasks yaml [yaml ...]         Per-task eval yamls (default: the train sequence)
  --cumulative yaml [yaml ...]         Cumulative eval yamls, one per task (optional)
  --tag NAME                           Override the auto-derived DATA_TAG
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
        --tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            TASK_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#TASK_YAMLS[@]} > 0 )) || experiment_die "--tasks needs at least one yaml"
            ;;
        --eval-tasks)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            EVAL_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#EVAL_YAMLS[@]} > 0 )) || experiment_die "--eval-tasks needs at least one yaml"
            ;;
        --cumulative)
            shift
            experiment_collect_yaml_args "$@"
            shift "$EXPERIMENT_CONSUMED"
            CUMULATIVE_YAMLS=("${EXPERIMENT_YAML_ARGS[@]}")
            (( ${#CUMULATIVE_YAMLS[@]} > 0 )) || experiment_die "--cumulative needs at least one yaml"
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

if (( ${#TASK_YAMLS[@]} > 0 )); then
    # Explicit yaml sequence: positionals are just model and method.
    MODEL_SPEC="${MODEL_SPEC:-${POSITIONAL[0]:-}}"
    METHOD="${METHOD:-${POSITIONAL[1]:-}}"
    (( ${#POSITIONAL[@]} <= 2 )) || experiment_die "Unexpected extra arguments: ${POSITIONAL[*]:2}"
elif (( ${#POSITIONAL[@]} > 0 )); then
    DATASET="${DATASET:-${POSITIONAL[0]:-}}"
    SPLIT="${SPLIT:-${POSITIONAL[1]:-}}"
    MODEL_SPEC="${MODEL_SPEC:-${POSITIONAL[2]:-}}"
    METHOD="${METHOD:-${POSITIONAL[3]:-}}"
    (( ${#POSITIONAL[@]} <= 4 )) || experiment_die "Unexpected extra arguments: ${POSITIONAL[*]:4}"
fi

[[ -n "$MODEL_SPEC" && -n "$METHOD" ]] || {
    usage >&2
    experiment_die "Need model and method"
}

experiment_resolve_dataset
if [[ -n "$MODEL_SIZE_FLAG" ]]; then
    experiment_parse_model_spec "$MODEL_SPEC"
    MODEL_SPEC="${MODEL_FAMILY}${MODEL_SIZE_FLAG}"
fi
experiment_load_model "$MODEL_SPEC"
if (( ${#PASSTHROUGH[@]} > 0 )); then
    EXTRA_TRAIN_ARGS+=("${PASSTHROUGH[@]}")
fi
experiment_set_output_paths "$METHOD"
experiment_write_manifest "$OUTPUT_DIR"

echo "=========================================="
echo "Incremental train"
echo "  dataset : ${DATASET}/${SPLIT}  (${DATA_TAG}, ${INCREMENTAL_SETTING})"
echo "  tasks   : ${#TASK_DATASETS[@]} train / ${#EVAL_DATASETS[@]} eval / ${#CUMULATIVE_DATASETS[@]} cumulative"
echo "  model   : ${MODEL_ID}  config=${MODEL_CONFIG}  weights=${MODEL_WEIGHTS:-<from scratch>}"
echo "  method  : ${METHOD}"
echo "  output  : ${OUTPUT_DIR}"
echo "  epochs  : ${EPOCHS}  batch=${BATCH_SIZE}  device=${DEVICE}"
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    echo "  extra   : ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "=========================================="

source scripts/run_incremental.sh
