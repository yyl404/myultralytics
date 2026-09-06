#!/bin/bash
# Demo pipeline: train -> eval -> labeled predict.
# Each stage can also run on its own; tunables live in common.sh.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DEMO_DIR/train.sh"
bash "$DEMO_DIR/eval.sh"
bash "$DEMO_DIR/predict.sh"
