#!/bin/bash

set -euo pipefail

source "scripts/voc/15_5/yolov8/config.sh"
METHOD="pseudo_label+dist+espreg"
DIST_TOPK="${DIST_TOPK:-20}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
