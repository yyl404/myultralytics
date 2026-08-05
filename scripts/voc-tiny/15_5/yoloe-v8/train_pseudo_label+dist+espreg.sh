#!/bin/bash

set -euo pipefail

source "scripts/voc-tiny/15_5/yoloe-v8/config.sh"
METHOD="pseudo_label+dist+espreg"
DIST_LOSS_WEIGHT="${DIST_LOSS_WEIGHT:-100.0}"
DIST_TOPK="${DIST_TOPK:-1}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
