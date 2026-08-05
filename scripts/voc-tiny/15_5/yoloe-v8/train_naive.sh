#!/bin/bash

set -euo pipefail

source "scripts/voc-tiny/15_5/yoloe-v8/config.sh"
METHOD="naive"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
