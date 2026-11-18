#!/bin/bash
# 15+5 CAD controls with ESPReg raw + PL + shared stage-1:
#   cad-uniform    all old-class channels, all anchors, no confidence mask
#   cad-foreground all old-class channels, anchors inside GT boxes, no confidence mask
set -euo pipefail

MACHINE_ID=1
TRAIN_STAGE1=0
STAGE1_SPLITS=("15+5")
RUNS=(
    "15+5|pseudo_label+espreg+dist|cad-uniform|-1|0|uniform"
    "15+5|pseudo_label+espreg+dist|cad-foreground|-1|0|foreground"
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
