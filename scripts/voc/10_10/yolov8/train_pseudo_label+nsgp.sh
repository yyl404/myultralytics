#!/bin/bash

set -euo pipefail

source "scripts/voc/10_10/yolov8/config.sh"
METHOD="pseudo_label+nsgp"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
