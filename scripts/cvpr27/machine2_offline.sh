#!/bin/bash
# Machine 2 (offline): 15+5 CAD channel ablation, top-3 / top-5 / all.
# Requires the shared 15+5 stage-1 checkpoint produced by machine 1; exits if missing.
#
# Usage:
#   bash scripts/cvpr27/machine2_offline.sh
#   PLAN_ONLY=1 bash scripts/cvpr27/machine2_offline.sh
set -euo pipefail

MACHINE_ID=2
TRAIN_STAGE1=0
STAGE1_SPLITS=("15+5")
RUNS=(
    "15+5|pseudo_label+espreg+dist|cad-top3|3|0"
    "15+5|pseudo_label+espreg+dist|cad-top5|5|0"
    "15+5|pseudo_label+espreg+dist|cad-top-all|-1|0"
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
