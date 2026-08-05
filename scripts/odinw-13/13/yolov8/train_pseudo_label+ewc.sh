#!/bin/bash

set -euo pipefail

source "scripts/odinw-13/13/yolov8/config.sh"
METHOD="pseudo_label+ewc"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
