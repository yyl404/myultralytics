#!/bin/bash

set -euo pipefail

source "scripts/voc/10_2_2_2_2_2/yolov8/config.sh"
METHOD="bpf"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
