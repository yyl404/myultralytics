#!/bin/bash

set -euo pipefail

source "scripts/coco/70_10/yolov8/config.sh"
METHOD="naive"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
