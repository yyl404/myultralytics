#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
METHOD="pseudo_label+nsgp+repre"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
