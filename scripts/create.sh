#!/bin/bash
# Unified CIL split creation. TIL datasets (odinw-13) are pre-bundled and have no create step.
#
# Usage:
#   bash scripts/create.sh --dataset voc --split 15_5
#   bash scripts/create.sh voc-tiny 15_5
#
# voc-tiny first writes a seeded subsample under data/VOC-TINY-YOLO, then the incremental split.
# Env: WORKERS, TINY_FRACTION (default 0.25), SEED (default 0).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=lib/experiment.sh
source scripts/lib/experiment.sh

usage() {
    cat <<'EOF'
Unified CIL split creation. TIL datasets (odinw-13) are pre-bundled.

  bash scripts/create.sh --dataset voc --split 15_5
  bash scripts/create.sh voc-tiny 15_5

voc-tiny first writes a seeded subsample under data/VOC-TINY-YOLO.
Env: WORKERS, TINY_FRACTION (default 0.25), SEED (default 0).
EOF
}

DATASET=""
SPLIT=""
OVERWRITE=""
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
        --overwrite)
            OVERWRITE=1
            shift
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

DATASET="${DATASET:-${POSITIONAL[0]:-}}"
SPLIT="${SPLIT:-${POSITIONAL[1]:-}}"
[[ -n "$DATASET" && -n "$SPLIT" ]] || {
    usage >&2
    experiment_die "Need dataset and split"
}

experiment_load_dataset "$DATASET" "$SPLIT"
[[ "$INCREMENTAL_SETTING" == "cil" ]] || experiment_die "Dataset '$DATASET' is ${INCREMENTAL_SETTING}; there is no create step"
[[ -n "$SOURCE_CFG" ]] || experiment_die "No source yaml configured for '$DATASET'"

if (( ${#CREATE_PREREQ_CMDS[@]} > 0 )); then
    for cmd in "${CREATE_PREREQ_CMDS[@]}"; do
        eval "$cmd"
    done
fi

create_args=(
    python tools/create_incremental_dataset.py
    --source_cfg "$SOURCE_CFG"
    --output_dir "$DATA_ROOT"
    --n_classes "${CLASS_COUNTS[@]}"
    --workers "${WORKERS:-8}"
)
if [[ -n "$OVERWRITE" || -n "$CREATE_OVERWRITE_DEFAULT" ]]; then
    create_args+=(--overwrite)
fi
"${create_args[@]}"
