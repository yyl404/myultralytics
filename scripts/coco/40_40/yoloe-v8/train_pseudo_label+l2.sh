#!/bin/bash

set -euo pipefail

source "scripts/coco/40_40/yoloe-v8/config.sh"
METHOD="pseudo_label+l2"
L2_LOSS_WEIGHT="${L2_LOSS_WEIGHT:-100.0}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
