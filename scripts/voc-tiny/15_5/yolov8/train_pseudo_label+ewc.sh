#!/bin/bash

set -euo pipefail

source "scripts/voc-tiny/15_5/yolov8/config.sh"
METHOD="pseudo_label+ewc"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
