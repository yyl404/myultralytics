#!/bin/bash

set -euo pipefail

source "scripts/voc/10_5_5/yolov8/config.sh"
METHOD="pseudo_label"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
