#!/bin/bash

set -euo pipefail

source "scripts/odinw-13/13/yoloe-v8/config.sh"
METHOD="pseudo_label+espreg"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
