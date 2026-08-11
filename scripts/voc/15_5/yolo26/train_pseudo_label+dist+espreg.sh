#!/bin/bash

set -euo pipefail

source "scripts/voc/15_5/yolo26/config.sh"
METHOD="pseudo_label+dist+espreg"
DIST_TOPK="${DIST_TOPK:-1}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
