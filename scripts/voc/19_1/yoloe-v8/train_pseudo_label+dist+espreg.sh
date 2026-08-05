#!/bin/bash

set -euo pipefail

source "scripts/voc/19_1/yoloe-v8/config.sh"
METHOD="pseudo_label+dist+espreg"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
