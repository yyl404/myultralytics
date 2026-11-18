#!/bin/bash
# Machine 3 (offline): 19+1 ESP-YOLO and EWC / NSGP / NSGP-RePRE / BPF baselines.
# Requires the shared 19+1 stage-1 checkpoint produced by machine 1; exits if missing.
#
# Usage:
#   bash scripts/cvpr27/machine3_offline.sh
#   PLAN_ONLY=1 bash scripts/cvpr27/machine3_offline.sh
set -euo pipefail

MACHINE_ID=3
TRAIN_STAGE1=0
STAGE1_SPLITS=("19+1")
RUNS=(
    "19+1|pseudo_label+espreg+dist|esp-yolo-cad-top1|1|0"
    "19+1|pseudo_label+ewc|ewc||0"
    "19+1|pseudo_label+nsgp|nsgp||0"
    "19+1|pseudo_label+nsgp+repre|nsgp-repre||0"
    "19+1|pseudo_label+bpf|bpf||0"
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
