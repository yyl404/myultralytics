#!/bin/bash

set -euo pipefail

source "scripts/voc/15_5/yolov8/config.sh"

NAIVE_TASK1="${OUTPUT_PREFIX}_naive/task-1"
START_TASK=2
export START_TASK

if [[ ! -f "${NAIVE_TASK1}/best.pt" ]]; then
    echo "Common naive Task 1 checkpoint not found: ${NAIVE_TASK1}/best.pt" >&2
    exit 1
fi

copy_naive_task1() {
    local method="$1"
    local target_task1="${OUTPUT_PREFIX}_${method}/task-1"

    mkdir -p "$target_task1"
    cp -a "${NAIVE_TASK1}/." "${target_task1}/"
    echo "Copied common naive Task 1 to: ${target_task1}"
}

copy_naive_task1 "pseudo_label+nsgp"
bash scripts/voc/15_5/yolov8/train_pseudo_label+nsgp.sh

copy_naive_task1 "pseudo_label+nsgp+repre"
bash scripts/voc/15_5/yolov8/train_pseudo_label+nsgp+repre.sh

copy_naive_task1 "pseudo_label+ewc"
bash scripts/voc/15_5/yolov8/train_pseudo_label+ewc.sh

copy_naive_task1 "bpf"
bash scripts/voc/15_5/yolov8/train_bpf.sh
