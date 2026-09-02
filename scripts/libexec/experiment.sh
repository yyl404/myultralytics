#!/bin/bash
# Shared dataset / model / output resolution for the unified experiment entry points.
# Sourced from the repo root (scripts/train.sh, eval.sh, create.sh, feature_drift.sh).

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

# Fill TASK_DATASETS / EVAL_DATASETS / CUMULATIVE_DATASETS from DATA_ROOT /
# INCREMENTAL_SETTING / SPLIT. The per-task eval sequence defaults to the train sequence.
experiment_fill_task_datasets() {
    TASK_DATASETS=()
    CUMULATIVE_DATASETS=()

    if [[ "$INCREMENTAL_SETTING" == "til" ]]; then
        [[ -d "$DATA_ROOT" ]] || experiment_die "TIL data root not found: $DATA_ROOT"
        local dir yaml
        while IFS= read -r dir; do
            yaml="${dir}/${TASK_YAML_NAME}"
            [[ -f "$yaml" ]] || experiment_die "Task yaml not found: $yaml"
            TASK_DATASETS+=("$yaml")
        done < <(LC_ALL=C find -L "$DATA_ROOT" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | LC_ALL=C sort)
        (( ${#TASK_DATASETS[@]} > 0 )) || experiment_die "No task directories under $DATA_ROOT"
        EVAL_DATASETS=("${TASK_DATASETS[@]}")
        return
    fi

    experiment_parse_split_counts "$SPLIT"
    local i k n sum=0 range
    for i in "${!CLASS_COUNTS[@]}"; do
        n="${CLASS_COUNTS[$i]}"
        k=$((i + 1))
        TASK_DATASETS+=("${DATA_ROOT}/task_${k}_cls_${n}/${TASK_YAML_NAME}")
        sum=$((sum + n))
        if (( k == 1 )); then
            CUMULATIVE_DATASETS+=("${DATA_ROOT}/task_1_cls_${n}/${TASK_YAML_NAME}")
        else
            range="1-${k}"
            CUMULATIVE_DATASETS+=("${DATA_ROOT}/task_${range}_cls_${sum}/${TASK_YAML_NAME}")
        fi
    done
    EVAL_DATASETS=("${TASK_DATASETS[@]}")
}

# Resolve --dataset / --split into DATA_TAG, TASK_DATASETS, default EPOCHS, create recipe.
experiment_load_dataset() {
    local dataset="${1:?dataset id}"
    local split="${2:?split id}"
    DATASET="$dataset"
    SPLIT="$split"
    CREATE_PREREQ_CMDS=()
    EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-}"
    TASK_YAML_NAME="dataset.yaml"
    SOURCE_CFG=""
    CREATE_OVERWRITE_DEFAULT=""

    case "$dataset" in
        voc)
            DATASET_FAMILY="voc"
            INCREMENTAL_SETTING="cil"
            DATA_TAG="VOC_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/VOC-YOLO/VOC.yaml"
            DEFAULT_EPOCHS=100
            ;;
        voc-tiny)
            DATASET_FAMILY="voc"
            INCREMENTAL_SETTING="cil"
            DATA_TAG="VOC-TINY_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/VOC-TINY-YOLO/VOC.yaml"
            DEFAULT_EPOCHS=10
            CREATE_OVERWRITE_DEFAULT="1"
            CREATE_PREREQ_CMDS=(
                "python tools/subsample_dataset.py --source_cfg data/VOC-YOLO/VOC.yaml --output_dir data/VOC-TINY-YOLO --fraction ${TINY_FRACTION:-0.1} --seed ${SEED:-0}"
            )
            ;;
        coco)
            DATASET_FAMILY="coco"
            INCREMENTAL_SETTING="cil"
            DATA_TAG="COCO_${split//_/+}"
            DATA_ROOT="data/${DATA_TAG}"
            SOURCE_CFG="data/coco-yolo/coco.yaml"
            DEFAULT_EPOCHS=12
            EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.75}"
            ;;
        odinw-13)
            DATASET_FAMILY="odinw"
            INCREMENTAL_SETTING="til"
            DATA_TAG="OdinW-13-yolo"
            DATA_ROOT="data/${DATA_TAG}"
            DEFAULT_EPOCHS=100
            TASK_YAML_NAME="data.yaml"
            [[ "$split" == "13" ]] || experiment_die "odinw-13 protocol split must be '13', got '$split'"
            ;;
        *)
            experiment_die "Unknown dataset '$dataset'. Known: $(experiment_known_datasets)"
            ;;
    esac

    experiment_apply_run_defaults
    experiment_fill_task_datasets
}

# Shared train/eval knobs. DEFAULT_EPOCHS is set by the dataset loader (100 if unset).
experiment_apply_run_defaults() {
    EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS:-100}}"
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

# Fill TASK_DATASETS from an explicit yaml sequence. The per-task eval sequence
# defaults to the train sequence; cumulative eval stays off unless
# experiment_set_eval_datasets / experiment_set_cumulative_datasets are called.
experiment_load_custom_tasks() {
    (( $# >= 1 )) || experiment_die "Need at least one task yaml"
    local yaml parent
    local -a tag_parts=()
    local seen="|"

    TASK_DATASETS=()
    CUMULATIVE_DATASETS=()
    DATASET="${DATASET:-custom}"
    SPLIT="${SPLIT:-custom}"
    DATASET_FAMILY="${DATASET_FAMILY:-custom}"
    INCREMENTAL_SETTING="til"
    SOURCE_CFG=""
    DATA_ROOT=""
    TASK_YAML_NAME=""
    CREATE_PREREQ_CMDS=()
    CREATE_OVERWRITE_DEFAULT=""
    EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-}"
    DEFAULT_EPOCHS="${DEFAULT_EPOCHS:-100}"

    for yaml in "$@"; do
        [[ -f "$yaml" ]] || experiment_die "Task yaml not found: $yaml"
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
    EVAL_DATASETS=("${TASK_DATASETS[@]}")
    experiment_apply_run_defaults
}

# Override the per-task eval yaml sequence. One yaml per task; defaults to the
# train sequence when never called.
experiment_set_eval_datasets() {
    (( $# >= 1 )) || experiment_die "--eval-tasks needs at least one yaml"
    (( $# == ${#TASK_DATASETS[@]} )) || experiment_die \
        "--eval-tasks must list one yaml per task (got $# yamls, ${#TASK_DATASETS[@]} tasks)"
    local yaml
    EVAL_DATASETS=()
    for yaml in "$@"; do
        [[ -f "$yaml" ]] || experiment_die "Eval yaml not found: $yaml"
        EVAL_DATASETS+=("$yaml")
    done
}

# Enable cumulative eval with one yaml per task. Optional: no cumulative yamls
# means no cumulative evaluation, regardless of the incremental setting.
experiment_set_cumulative_datasets() {
    (( $# >= 1 )) || experiment_die "--cumulative needs at least one yaml"
    (( $# == ${#TASK_DATASETS[@]} )) || experiment_die \
        "--cumulative must list one yaml per task (got $# yamls, ${#TASK_DATASETS[@]} tasks)"
    local yaml
    CUMULATIVE_DATASETS=()
    for yaml in "$@"; do
        [[ -f "$yaml" ]] || experiment_die "Cumulative yaml not found: $yaml"
        CUMULATIVE_DATASETS+=("$yaml")
    done
}

# Resolve either the registered dataset/split or an explicit --tasks yaml list,
# then apply the optional --eval-tasks / --cumulative overrides on top.
experiment_resolve_dataset() {
    if (( ${#TASK_YAMLS[@]} > 0 )); then
        if [[ -n "${DATASET:-}" || -n "${SPLIT:-}" ]]; then
            experiment_die "Use either --dataset/--split or --tasks, not both"
        fi
        experiment_load_custom_tasks "${TASK_YAMLS[@]}"
    else
        [[ -n "${DATASET:-}" && -n "${SPLIT:-}" ]] || experiment_die "Need --dataset and --split, or --tasks <yaml...>"
        experiment_load_dataset "$DATASET" "$SPLIT"
    fi
    if (( ${#EVAL_YAMLS[@]} > 0 )); then
        experiment_set_eval_datasets "${EVAL_YAMLS[@]}"
    fi
    if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
        experiment_set_cumulative_datasets "${CUMULATIVE_YAMLS[@]}"
    fi
    if [[ -n "${DATA_TAG_OVERRIDE:-}" ]]; then
        DATA_TAG="$DATA_TAG_OVERRIDE"
    fi
}

# Persist the yaml sequences next to the run so eval/detect can recover them.
experiment_write_manifest() {
    local run="${1:-${OUTPUT_DIR:?}}"
    mkdir -p "$run"
    printf '%s\n' "${TASK_DATASETS[@]}" > "${run}/task_yamls.txt"
    printf '%s\n' "${EVAL_DATASETS[@]}" > "${run}/eval_yamls.txt"
    if (( ${#CUMULATIVE_DATASETS[@]} > 0 )); then
        printf '%s\n' "${CUMULATIVE_DATASETS[@]}" > "${run}/cumulative_yamls.txt"
    else
        rm -f "${run}/cumulative_yamls.txt"
    fi
    cat > "${run}/experiment.meta" <<EOF
INCREMENTAL_SETTING=${INCREMENTAL_SETTING}
DATA_TAG=${DATA_TAG}
DATASET=${DATASET}
SPLIT=${SPLIT}
EVAL_IOU_THRESHOLD=${EVAL_IOU_THRESHOLD:-}
EOF
}

experiment_read_yaml_file() {
    local file="$1"
    local line
    EXPERIMENT_YAML_ARGS=()
    [[ -f "$file" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        EXPERIMENT_YAML_ARGS+=("$line")
    done < "$file"
    (( ${#EXPERIMENT_YAML_ARGS[@]} > 0 ))
}

# Load the yaml sequences from a previous run's manifest. Returns 1 if none exists.
experiment_try_load_manifest() {
    local run="${1:?run dir}"
    experiment_read_yaml_file "${run}/task_yamls.txt" || return 1
    experiment_load_custom_tasks "${EXPERIMENT_YAML_ARGS[@]}"
    if experiment_read_yaml_file "${run}/eval_yamls.txt"; then
        experiment_set_eval_datasets "${EXPERIMENT_YAML_ARGS[@]}"
    fi
    if experiment_read_yaml_file "${run}/cumulative_yamls.txt"; then
        experiment_set_cumulative_datasets "${EXPERIMENT_YAML_ARGS[@]}"
    fi
    if [[ -f "${run}/experiment.meta" ]]; then
        local k v
        while IFS='=' read -r k v; do
            case "$k" in
                INCREMENTAL_SETTING|DATA_TAG|DATASET|SPLIT)
                    printf -v "$k" '%s' "$v"
                    ;;
                EVAL_IOU_THRESHOLD)
                    [[ -n "$v" ]] && EVAL_IOU_THRESHOLD="$v"
                    ;;
            esac
        done < "${run}/experiment.meta"
    fi
    if [[ -n "${DATA_TAG_OVERRIDE:-}" ]]; then
        DATA_TAG="$DATA_TAG_OVERRIDE"
    fi
    return 0
}

# Eval/detect: --tasks, else --dataset/--split, else run manifest, else infer from
# run name; explicit --eval-tasks / --cumulative / --tag override whichever source won.
experiment_resolve_eval_dataset() {
    local run="${1:-}"
    if (( ${#TASK_YAMLS[@]} > 0 )); then
        if [[ -n "${DATASET:-}" || -n "${SPLIT:-}" ]]; then
            experiment_die "Use either --dataset/--split or --tasks, not both"
        fi
        experiment_load_custom_tasks "${TASK_YAMLS[@]}"
    elif [[ -n "${DATASET:-}" || -n "${SPLIT:-}" ]]; then
        [[ -n "${DATASET:-}" && -n "${SPLIT:-}" ]] || experiment_die "Pass both --dataset and --split, or neither"
        experiment_load_dataset "$DATASET" "$SPLIT"
    else
        [[ -n "$run" ]] || experiment_die "Need --dataset/--split, --tasks, or a run directory"
        if ! experiment_try_load_manifest "$run"; then
            experiment_infer_dataset_from_run "$run"
        fi
    fi
    if (( ${#EVAL_YAMLS[@]} > 0 )); then
        experiment_set_eval_datasets "${EVAL_YAMLS[@]}"
    fi
    if (( ${#CUMULATIVE_YAMLS[@]} > 0 )); then
        experiment_set_cumulative_datasets "${CUMULATIVE_YAMLS[@]}"
    fi
    if [[ -n "${DATA_TAG_OVERRIDE:-}" ]]; then
        DATA_TAG="$DATA_TAG_OVERRIDE"
    fi
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
    elif [[ "${DATASET:-}" == "voc-tiny" ]]; then
        echo "m"
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

# YOLO26 train hyps. end2end=False on every yolo26 run (NMS / one2many).
# voc-tiny keeps the small-data fine-tune knobs from the previous yolo26 config.
experiment_apply_yolo26_hyps() {
    [[ "$MODEL_FAMILY" == "yolo26" ]] || return 0
    [[ "${YOLO26_DEFAULT_HYPS:-1}" == "1" ]] || return 0
    EXTRA_TRAIN_ARGS+=(
        --end2end "${END2END:-False}"
    )
    if [[ "${DATASET:-}" == "voc-tiny" ]]; then
        EXTRA_TRAIN_ARGS+=(
            --optimizer "${OPTIMIZER:-AdamW}"
            --lr0 "${LR0:-0.001}"
            --warmup_bias_lr "${WARMUP_BIAS_LR:-0.0}"
            --mosaic "${MOSAIC:-0.5}"
            --freeze "${FREEZE:-10}"
        )
    fi
}

# Resolve --model into MODEL_ID / MODEL_CONFIG / MODEL_WEIGHTS / EXTRA_TRAIN_ARGS.
# Call after experiment_load_dataset so voc-tiny size and yolo26 hyps can depend on DATASET.
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
            MODEL_ID="yolov8${MODEL_SIZE}"
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

# Infer dataset + split from a training run directory name.
experiment_infer_dataset_from_run() {
    local name
    name="$(basename "${1:?run dir}")"
    if [[ "$name" =~ VOC-TINY_([0-9+]+) ]]; then
        experiment_load_dataset "voc-tiny" "${BASH_REMATCH[1]//+/_}"
    elif [[ "$name" =~ COCO_([0-9+]+) ]]; then
        experiment_load_dataset "coco" "${BASH_REMATCH[1]//+/_}"
    elif [[ "$name" == *OdinW-13-yolo* ]]; then
        experiment_load_dataset "odinw-13" "13"
    elif [[ "$name" =~ VOC_([0-9+]+) ]]; then
        experiment_load_dataset "voc" "${BASH_REMATCH[1]//+/_}"
    else
        experiment_die "Cannot infer dataset/split from run name '$name'. Pass --dataset and --split."
    fi
}
