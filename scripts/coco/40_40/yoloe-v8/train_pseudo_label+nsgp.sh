#!/bin/bash

set -euo pipefail

source "scripts/coco/40_40/yoloe-v8/config.sh"
METHOD="pseudo_label+nsgp"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
