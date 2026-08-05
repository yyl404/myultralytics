#!/bin/bash

set -euo pipefail

source "scripts/voc/5_5_5_5/yoloe-v8/config.sh"
METHOD="pseudo_label+nsgp+repre"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
