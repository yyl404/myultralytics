#!/bin/bash

set -euo pipefail

source "scripts/voc/5_5_5_5/yolo26/config.sh"
METHOD="pseudo_label+dist+espreg"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
