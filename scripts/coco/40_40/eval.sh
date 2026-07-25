#!/bin/bash
set -euo pipefail

output_dir="${1:?Usage: bash scripts/coco/40_40/eval.sh <training-output-dir>}"
device="${DEVICE:-0}"
datasets=(
    "data/COCO_40+40/task_1_cls_40/dataset.yaml"
    "data/COCO_40+40/task_1-2_cls_80/dataset.yaml"
)
mkdir -p "${output_dir}/evaluation"

for task_id in 1 2; do
    model="${output_dir}/task-${task_id}/best.pt"
    python tools/eval.py \
        --model "$model" \
        --data "${datasets[$((task_id - 1))]}" \
        --device "$device" \
        --save_path "${output_dir}/evaluation/task-${task_id}.csv" \
        --confusion_matrix_path "${output_dir}/evaluation/task-${task_id}-confusion.csv" \
        --project "${output_dir}/evaluation/task-${task_id}"
done
