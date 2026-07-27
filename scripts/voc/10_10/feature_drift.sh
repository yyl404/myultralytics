#!/bin/bash
# Feature drift between two checkpoints on the task-1 sample set of VOC 10+10.
# The dataset is fixed for this split; the two model checkpoints are passed in.
# Usage: bash scripts/voc/10_10/feature_drift.sh MODEL1_CKPT MODEL2_CKPT [SAVE_PATH]
# Example: bash scripts/voc/10_10/feature_drift.sh \
#     runs/<run>/task-1/train/weights/last.pt \
#     runs/<run>/task-2/train/weights/last.pt

set -euo pipefail

MODEL1="${1:?Pass the task-1 model checkpoint path}"
MODEL2="${2:?Pass the task-2 model checkpoint path}"
SAVE_PATH="${3:-$(dirname "$MODEL2")/feature_drift_task1_to_task2.json}"

python tools/feature_drift.py \
    --data "data/VOC_10+10/task_1_cls_10/dataset.yaml" \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --save_path "$SAVE_PATH"
