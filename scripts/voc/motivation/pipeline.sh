#!/bin/bash
# Motivation pipeline: build splits -> stage statistics -> incremental training
# -> teacher high-confidence error analysis. Each stage can also run on its own;
# tunables live in common.sh.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DEMO_DIR/create.sh"
bash "$DEMO_DIR/stats.sh"
bash "$DEMO_DIR/train.sh"
bash "$DEMO_DIR/teacher_errors.sh"
