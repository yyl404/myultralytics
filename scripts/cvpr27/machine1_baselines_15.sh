#!/bin/bash
# Local 15+5 baselines: EWC / BPF / NSGP / NSGP-RePre with Pseudo Label only.
# Shared 15+5 stage-1, YOLOv8m, SGD 0.01, 100 epochs, no LwF, no CAD.
# Each method is trained, evaluated, and appended to results/cvpr27/machine1.csv.
set -euo pipefail

MACHINE_ID=1
TRAIN_STAGE1=0
STAGE1_SPLITS=("15+5")
RUNS=(
    "15+5|pseudo_label+ewc|ewc||0"
    "15+5|pseudo_label+bpf|bpf||0"
    "15+5|pseudo_label+nsgp|nsgp||0"
    "15+5|pseudo_label+nsgp+repre|nsgp-repre||0"
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
