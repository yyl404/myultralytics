#!/bin/bash
# Shared dataset / model / output resolution for the unified experiment entry points.
# Sourced from the repo root (scripts/train.sh, eval.sh, predict.sh, create.sh,
# feature_drift.sh).
#
# Dataset identity knobs (--dataset/--split) exist ONLY for create.sh (slicing a
# full single-stage dataset into class-incremental task datasets). Train, eval,
# and predict take explicit yaml sequences.

experiment_known_datasets() {
    echo "voc voc-tiny coco odinw-13"
}

experiment_known_model_families() {
    echo "yolo26 yolov8 yoloe-v8"
}

experiment_known_method_components() {
    echo "naive bpf pseudo_label ewc l2 dist espreg nsgp repre replay"
}

experiment_die() {
    echo "$*" >&2
    exit 1
}

# Parse "15_5" / "10_2_2_2_2_2" into CLASS_COUNTS.
experiment_parse_split_counts() {
    local split="$1"
    local part
    CLASS_COUNTS=()
    [[ "$split" =~ ^[0-9]+(_[0-9]+)*$ ]] || experiment_die "CIL split must be underscore-joined class counts, got '$split'"
    IFS='_' read -ra CLASS_COUNTS <<< "$split"
    for part in "${CLASS_COUNTS[@]}"; do
        (( part > 0 )) || experiment_die "Split class counts must be positive integers, got '$split'"
    done
}

# Validate a yaml sequence: at least one entry, every file must exist.
experiment_check_yamls() {
    (( $# >= 1 )) || experiment_die "Need at least one yaml"
    local yaml
    for yaml in "$@"; do
        [[ -f "$yaml" ]] || experiment_die "Yaml not found: $yaml"
    done
}

# Pick the split of one dataset yaml to evaluate / predict on.
# Usage: split=$(experiment_resolve_split <yaml> <auto|test|val|train>)
# "auto" prefers test and falls back to val.
experiment_resolve_split() {
    local yaml="${1:?yaml}" requested="${2:?split}"
    if [[ "$requested" != "auto" ]]; then
        grep -qE "^${requested}:" "$yaml" || experiment_die "Split '$requested' not found in $yaml"
        echo "$requested"
        return
    fi
    if grep -qE '^test:' "$yaml"; then
        echo "test"
        return
    fi
    grep -qE '^val:' "$yaml" || experiment_die "Neither test nor val split found in $yaml"
    echo "val"
}

# Echo which of the train/val/test split keys a dataset yaml defines, in
# canonical order (space-separated).
experiment_yaml_splits() {
    local yaml="${1:?yaml}" key
    local -a found=()
    for key in train val test; do
        grep -qE "^${key}:" "$yaml" && found+=("$key")
    done
    (( ${#found[@]} > 0 )) || experiment_die "No train/val/test split found in $yaml"
    echo "${found[@]}"
}

# Fill EXPERIMENT_MODEL_TASKS with the numerically sorted task ids that have a
# best.pt under a run directory. Fails when the run holds no model at all.
experiment_list_model_tasks() {
    local run="${1:?run dir}"
    local dir
    EXPERIMENT_MODEL_TASKS=()
    for dir in "$run"/task-*; do
        [[ -f "$dir/best.pt" ]] || continue
        EXPERIMENT_MODEL_TASKS+=("${dir##*/task-}")
    done
    (( ${#EXPERIMENT_MODEL_TASKS[@]} > 0 )) || experiment_die "No task-*/best.pt found under $run"
    IFS=$'\n' EXPERIMENT_MODEL_TASKS=($(sort -n <<<"${EXPERIMENT_MODEL_TASKS[*]}"))
    unset IFS
}

# Registered CIL split recipe, used by create.sh only: DATA_TAG, DATA_ROOT,
# SOURCE_CFG, CLASS_COUNTS, plus the optional subsample prerequisite.
experiment_load_dataset() {
    local dataset="${1:?dataset id}"
    local split="${2:?split id}"
    DATASET="$dataset"
    SPLIT="$split"
    CREATE_PREREQ_CMDS=()
    SOURCE_CFG=""
    CREATE_OVERWRITE_DEFAULT=""

    case "$dataset" in
        voc)
            INCREMENTAL_SETTING="cil"
            DATA_TAG="VOC_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/VOC-YOLO/VOC.yaml"
            ;;
        voc-tiny)
            INCREMENTAL_SETTING="cil"
            DATA_TAG="VOC-TINY_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/VOC-TINY-YOLO/VOC.yaml"
            CREATE_OVERWRITE_DEFAULT="1"
            CREATE_PREREQ_CMDS=(
                "python tools/subsample_dataset.py --source_cfg data/VOC-YOLO/VOC.yaml --output_dir data/VOC-TINY-YOLO --fraction ${TINY_FRACTION:-0.1} --seed ${SEED:-0}"
            )
            ;;
        coco)
            INCREMENTAL_SETTING="cil"
            DATA_TAG="COCO_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/coco-yolo/coco.yaml"
            ;;
        odinw-13)
            INCREMENTAL_SETTING="til"
            DATA_TAG="OdinW-13-yolo"
            DATA_ROOT="data/${DATA_TAG}"
            [[ "$split" == "13" ]] || experiment_die "odinw-13 protocol split must be '13', got '$split'"
            ;;
        *)
            experiment_die "Unknown dataset '$dataset'. Known: $(experiment_known_datasets)"
            ;;
    esac

    if [[ "$INCREMENTAL_SETTING" == "cil" ]]; then
        experiment_parse_split_counts "$split"
    fi
}

# Shared train knobs.
experiment_apply_run_defaults() {
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    IMGSZ="${IMGSZ:-640}"
    WORKERS="${WORKERS:-8}"
    DEVICE="${DEVICE:-0}"
}

# Consume non-flag args into EXPERIMENT_YAML_ARGS. Stops at the next --option.
# A single comma-separated argument is also accepted: a.yaml,b.yaml,c.yaml
# Caller: shift; experiment_collect_yaml_args "$@"; shift "$EXPERIMENT_CONSUMED"
experiment_collect_yaml_args() {
    EXPERIMENT_YAML_ARGS=()
    EXPERIMENT_CONSUMED=0
    while [[ $# -gt 0 && "$1" != --* ]]; do
        EXPERIMENT_YAML_ARGS+=("$1")
        EXPERIMENT_CONSUMED=$((EXPERIMENT_CONSUMED + 1))
        shift
    done
    if (( ${#EXPERIMENT_YAML_ARGS[@]} == 1 )) && [[ "${EXPERIMENT_YAML_ARGS[0]}" == *,* ]]; then
        local csv="${EXPERIMENT_YAML_ARGS[0]}"
        local part
        local -a _yaml_csv
        EXPERIMENT_YAML_ARGS=()
        IFS=',' read -ra _yaml_csv <<< "$csv"
        for part in "${_yaml_csv[@]}"; do
            part="${part#"${part%%[![:space:]]*}"}"
            part="${part%"${part##*[![:space:]]}"}"
            [[ -n "$part" ]] && EXPERIMENT_YAML_ARGS+=("$part")
        done
    fi
}

# Fill TASK_DATASETS from an explicit yaml sequence. DATA_TAG is derived from
# the parent directory names joined with '+' unless --tag overrides it.
experiment_load_custom_tasks() {
    experiment_check_yamls "$@"
    local yaml parent
    local -a tag_parts=()
    local seen="|"

    TASK_DATASETS=()
    for yaml in "$@"; do
        TASK_DATASETS+=("$yaml")
        parent="$(basename "$(dirname "$yaml")")"
        if [[ "$parent" == "." ]]; then
            parent="$(basename "${yaml%.*}")"
        fi
        if [[ "$seen" != *"|$parent|"* ]]; then
            tag_parts+=("$parent")
            seen="${seen}${parent}|"
        fi
    done

    local IFS=+
    DATA_TAG="${DATA_TAG_OVERRIDE:-${tag_parts[*]}}"
    unset IFS
}

# Parse yolo26 / yolo26m / yoloe-v8l into MODEL_FAMILY + MODEL_SIZE (size may still be empty).
experiment_parse_model_spec() {
    local spec="$1"
    MODEL_FAMILY=""
    MODEL_SIZE=""
    case "$spec" in
        yoloe-v8[nslmx])
            MODEL_FAMILY="yoloe-v8"
            MODEL_SIZE="${spec: -1}"
            ;;
        yoloe-v8)
            MODEL_FAMILY="yoloe-v8"
            ;;
        yolo26[nslmx])
            MODEL_FAMILY="yolo26"
            MODEL_SIZE="${spec: -1}"
            ;;
        yolo26)
            MODEL_FAMILY="yolo26"
            ;;
        yolov8[nslmx])
            MODEL_FAMILY="yolov8"
            MODEL_SIZE="${spec: -1}"
            ;;
        yolov8)
            MODEL_FAMILY="yolov8"
            ;;
        *)
            experiment_die "Unknown model '$spec'. Known families: $(experiment_known_model_families) (optional size suffix n/s/m/l/x)"
            ;;
    esac
}

experiment_default_size() {
    if [[ "$MODEL_FAMILY" == "yoloe-v8" ]]; then
        echo "l"
    else
        echo "x"
    fi
}

experiment_default_weights() {
    case "$MODEL_FAMILY" in
        yolo26)
            if [[ "$MODEL_SIZE" == "m" ]]; then
                echo "yoloe-26m-seg.pt"
            else
                echo "yolo26${MODEL_SIZE}.pt"
            fi
            ;;
        yolov8)
            if [[ "$MODEL_SIZE" == "m" ]]; then
                echo "yoloe-v8m-seg.pt"
            else
                echo "yolov8${MODEL_SIZE}-cls.pt"
            fi
            ;;
        yoloe-v8)
            echo "yoloe-v8${MODEL_SIZE}-seg.pt"
            ;;
    esac
}

# YOLO26 train hyps: end2end=False on every yolo26 run (NMS / one2many).
experiment_apply_yolo26_hyps() {
    [[ "$MODEL_FAMILY" == "yolo26" ]] || return 0
    [[ "${YOLO26_DEFAULT_HYPS:-1}" == "1" ]] || return 0
    EXTRA_TRAIN_ARGS+=(
        --end2end "${END2END:-False}"
    )
}

# Resolve --model into MODEL_ID / MODEL_CONFIG / MODEL_WEIGHTS / EXTRA_TRAIN_ARGS.
experiment_load_model() {
    local spec="${1:?model spec}"
    MODEL_ADAPTER="${MODEL_ADAPTER:-scripts/model_adapters/ultralytics.sh}"
    experiment_parse_model_spec "$spec"
    MODEL_SIZE="${MODEL_SIZE:-$(experiment_default_size)}"

    case "$MODEL_FAMILY" in
        yolo26)
            MODEL_ID="yolo26${MODEL_SIZE}"
            MODEL_CONFIG="yolo26${MODEL_SIZE}.yaml"
            ;;
        yolov8|yoloe-v8)
            MODEL_ID="yolov8${MODEL_SIZE}"          # yoloe-v8 keeps yolov8 id for run-name compat
            MODEL_CONFIG="yolov8${MODEL_SIZE}.yaml"
            ;;
    esac

    if [[ -n "${FROM_SCRATCH:-}" ]]; then
        MODEL_WEIGHTS=""
    elif [[ -v MODEL_WEIGHTS_OVERRIDE ]]; then
        MODEL_WEIGHTS="$MODEL_WEIGHTS_OVERRIDE"
    else
        MODEL_WEIGHTS="$(experiment_default_weights)"
    fi

    if ! declare -p EXTRA_TRAIN_ARGS >/dev/null 2>&1; then
        EXTRA_TRAIN_ARGS=()
    fi
    experiment_apply_yolo26_hyps
}

experiment_set_output_paths() {
    local method="${1:-}"
    if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
        OUTPUT_PREFIX="runs/${MODEL_ID}_${DATA_TAG}_pretrained-from-${MODEL_WEIGHTS%.pt}"
    else
        OUTPUT_PREFIX="runs/${MODEL_ID}_${DATA_TAG}_fromscratch"
    fi
    if [[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]]; then
        OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
    elif [[ -n "$method" ]]; then
        OUTPUT_DIR="${OUTPUT_PREFIX}_${method}"
    else
        OUTPUT_DIR="$OUTPUT_PREFIX"
    fi
}
