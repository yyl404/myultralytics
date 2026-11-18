#!/bin/bash
# Motivation teacher-error stage: run the stage-1 teacher on the target-stage
# training images and measure high-confidence error rates against the full
# class-space GT. Requires the protocol's task-1 checkpoint. Tunables live in
# common.sh.

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

teacher_errors_protocol() {
    local tag="$1" run_dir="$2"
    shift 2
    local yamls=("$@")
    local yaml teacher images target_yaml out_dir

    for yaml in "${yamls[@]}"; do
        if [[ ! -f "$yaml" ]]; then
            echo "Stage yaml not found: $yaml" >&2
            echo "Create the dataset first with bash $DEMO_DIR/create.sh, or fix common.sh" >&2
            exit 1
        fi
    done
    if (( TARGET_STAGE > ${#yamls[@]} )); then
        echo "TARGET_STAGE=${TARGET_STAGE} exceeds the ${#yamls[@]} stages of ${tag}" >&2
        exit 1
    fi
    teacher="${run_dir}/task-1/best.pt"
    if [[ ! -f "$teacher" ]]; then
        echo "Teacher checkpoint not found: $teacher" >&2
        echo "Train the protocol first with bash $DEMO_DIR/train.sh" >&2
        exit 1
    fi
    target_yaml="${yamls[$((TARGET_STAGE - 1))]}"
    images="$(dirname "$target_yaml")/images/${ANALYSIS_SPLIT}"
    out_dir="${run_dir}/teacher_errors/task_${TARGET_STAGE}"

    if [[ ! -d "$images" ]]; then
        echo "Target-stage images not found: $images" >&2
        exit 1
    fi
    rm -rf "${out_dir}/predictions" "${out_dir}/examples"
    mkdir -p "$out_dir"
    echo "=========================================="
    echo "Motivation teacher errors: ${tag}"
    echo "  teacher: ${teacher}"
    echo "  stage  : ${TARGET_STAGE}  split: ${ANALYSIS_SPLIT}"
    echo "  images : ${images}"
    echo "  output : ${out_dir}"
    echo "=========================================="
    python tools/predict.py \
        --model "$teacher" \
        --images "$images" \
        --save_path "${out_dir}/predictions" \
        --project "$out_dir" \
        --device "$DEVICE" \
        --conf "$PREDICT_CONF" \
        "${EXTRA_PREDICT_ARGS[@]}"
    bash scripts/analyze.sh teacher_errors \
        --predictions "${out_dir}/predictions" \
        --model "$teacher" \
        --stages "${yamls[@]}" \
        --target_stage "$TARGET_STAGE" \
        --source_cfg "$SOURCE_CFG" \
        --split "$ANALYSIS_SPLIT" \
        --iou_threshold "$IOU_THRESHOLD" \
        --conf_min "$CONF_MIN" \
        --conf_max "$CONF_MAX" \
        --conf_step "$CONF_STEP" \
        --vis_dir "${out_dir}/examples" \
        --vis_conf "$VIS_CONF" \
        --output_dir "$out_dir"
}

teacher_errors_two_stage() {
    teacher_errors_protocol "$TWO_STAGE_TAG" "$RUN_DIR_TWO" "${TWO_STAGE_YAMLS[@]}"
}

teacher_errors_four_stage() {
    teacher_errors_protocol "$FOUR_STAGE_TAG" "$RUN_DIR_FOUR" "${FOUR_STAGE_YAMLS[@]}"
}

case "$PROTOCOL" in
    two)
        teacher_errors_two_stage
        ;;
    four)
        teacher_errors_four_stage
        ;;
    all)
        teacher_errors_two_stage
        teacher_errors_four_stage
        ;;
    *)
        echo "Unknown PROTOCOL '$PROTOCOL' (expected two, four, or all)" >&2
        exit 1
        ;;
esac
