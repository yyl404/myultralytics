#!/bin/bash

set -euo pipefail

source "scripts/odinw-13/13/yolov8/config.sh"
METHOD="naive"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
