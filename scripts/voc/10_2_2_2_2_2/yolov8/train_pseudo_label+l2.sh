#!/bin/bash

set -euo pipefail

source "scripts/voc/10_2_2_2_2_2/yolov8/config.sh"
METHOD="pseudo_label+l2"
L2_LOSS_WEIGHT="${L2_LOSS_WEIGHT:-100.0}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
