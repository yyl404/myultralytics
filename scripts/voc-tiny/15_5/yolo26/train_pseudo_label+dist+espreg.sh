#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
METHOD="pseudo_label+dist+espreg"
DIST_LOSS_WEIGHT="${DIST_LOSS_WEIGHT:-100.0}"
DIST_TOPK="${DIST_TOPK:-1}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
