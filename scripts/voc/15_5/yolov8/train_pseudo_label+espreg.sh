#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
METHOD="pseudo_label+espreg"
ESPREG_LOSS_WEIGHT="${ESPREG_LOSS_WEIGHT:-100.0}"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
