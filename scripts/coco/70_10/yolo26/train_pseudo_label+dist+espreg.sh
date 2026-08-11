#!/bin/bash

set -euo pipefail

source "scripts/coco/70_10/yolo26/config.sh"
METHOD="pseudo_label+dist+espreg"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
