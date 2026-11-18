#!/bin/bash
# Motivation incremental-error stage: run each incrementally trained stage model
# on the history-stage images and measure high-confidence error rates against the
# full class-space GT. Requires one trained two-stage run per method in
# INCREMENTAL_METHODS; the paper's experiment 3 compares naive with YOLO-LwF
# (which uses no replay). Run it with PROTOCOL=two. Tunables live in common.sh.

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

incremental_errors_method() {
    local tag="$1" method="$2"
    shift 2
    local yamls=("$@")
    local yaml run_dir model images out_dir

    for yaml in "${yamls[@]}"; do
        if [[ ! -f "$yaml" ]]; then
            echo "Stage yaml not found: $yaml" >&2
            echo "Create the dataset first with bash $DEMO_DIR/create.sh, or fix common.sh" >&2
            exit 1
        fi
    done
    if (( INCREMENTAL_STAGE > ${#yamls[@]} )); then
        echo "INCREMENTAL_STAGE=${INCREMENTAL_STAGE} exceeds the ${#yamls[@]} stages of ${tag}" >&2
        exit 1
    fi
    run_dir="$(motivation_run_dir "$tag" "$method")"
    model="${run_dir}/task-${INCREMENTAL_STAGE}/best.pt"
    if [[ ! -f "$model" ]]; then
        echo "Incremental checkpoint not found: $model" >&2
        echo "Train the method first: METHOD=${method} bash $DEMO_DIR/train.sh" >&2
        exit 1
    fi
    if (( INCREMENTAL_IMAGES_STAGE > ${#yamls[@]} )); then
        echo "INCREMENTAL_IMAGES_STAGE=${INCREMENTAL_IMAGES_STAGE} exceeds the ${#yamls[@]} stages of ${tag}" >&2
        exit 1
    fi
    images="$(dirname "${yamls[$((INCREMENTAL_IMAGES_STAGE - 1))]}")/images/${INCREMENTAL_SPLIT}"
    if [[ ! -d "$images" ]]; then
        echo "History-stage images not found: $images" >&2
        exit 1
    fi
    out_dir="${run_dir}/incremental_errors/task-${INCREMENTAL_STAGE}-on-stage-${INCREMENTAL_IMAGES_STAGE}"
    rm -rf "${out_dir}/predictions" "${out_dir}/examples"
    mkdir -p "$out_dir"
    echo "=========================================="
    echo "Motivation incremental errors: ${tag} (${method})"
    echo "  model  : ${model}"
    echo "  stage  : ${INCREMENTAL_STAGE} on stage ${INCREMENTAL_IMAGES_STAGE} images"
    echo "  images : ${images}  split: ${INCREMENTAL_SPLIT}"
    echo "  output : ${out_dir}"
    echo "=========================================="
    python tools/predict.py \
        --model "$model" \
        --images "$images" \
        --save_path "${out_dir}/predictions" \
        --project "$out_dir" \
        --device "$DEVICE" \
        --conf "$PREDICT_CONF" \
        "${EXTRA_PREDICT_ARGS[@]}"
    bash scripts/analyze.sh incremental_errors \
        --predictions "${out_dir}/predictions" \
        --model "$model" \
        --stages "${yamls[@]}" \
        --history_stages "$INCREMENTAL_HISTORY_STAGES" \
        --source_cfg "$SOURCE_CFG" \
        --split "$INCREMENTAL_SPLIT" \
        --iou_threshold "$IOU_THRESHOLD" \
        --conf_min "$CONF_MIN" \
        --conf_max "$CONF_MAX" \
        --conf_step "$CONF_STEP" \
        --vis_dir "${out_dir}/examples" \
        --vis_conf "$VIS_CONF" \
        --output_dir "$out_dir"
}

incremental_errors_protocol() {
    local tag="$1" method
    shift 1
    for method in "${INCREMENTAL_METHODS[@]}"; do
        incremental_errors_method "$tag" "$method" "$@"
    done
}

incremental_errors_two_stage() {
    incremental_errors_protocol "$TWO_STAGE_TAG" "${TWO_STAGE_YAMLS[@]}"
}

incremental_errors_four_stage() {
    incremental_errors_protocol "$FOUR_STAGE_TAG" "${FOUR_STAGE_YAMLS[@]}"
}

case "$PROTOCOL" in
    two)
        incremental_errors_two_stage
        ;;
    four)
        incremental_errors_four_stage
        ;;
    all)
        incremental_errors_two_stage
        incremental_errors_four_stage
        ;;
    *)
        echo "Unknown PROTOCOL '$PROTOCOL' (expected two, four, or all)" >&2
        exit 1
        ;;
esac
