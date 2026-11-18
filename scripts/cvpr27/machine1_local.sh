#!/bin/bash
# Machine 1 (local): train and publish the two shared stage-1 checkpoints, then
# run the 15+5 ESP-YOLO main row (CAD top-1).
#
# Usage:
#   bash scripts/cvpr27/machine1_local.sh
#   PLAN_ONLY=1 bash scripts/cvpr27/machine1_local.sh
#
# The other two machines require the stage-1 files printed at the end of each
# stage-1 run; copy them to the same path on those machines before running.
set -euo pipefail

MACHINE_ID=1
TRAIN_STAGE1=1
STAGE1_SPLITS=("15+5" "19+1")
RUNS=(
    "15+5|pseudo_label+espreg+dist|esp-yolo-cad-top1|1|0"
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
