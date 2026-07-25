#!/bin/bash

set -euo pipefail

source "scripts/voc/19_1/yolov8/config.sh"
METHOD="naive"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
