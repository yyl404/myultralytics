#!/bin/bash

set -euo pipefail

source "scripts/voc-tiny/15_5/yolov8/config.sh"
BPF_OBJECT_TOPK="${BPF_OBJECT_TOPK:-0.2}"
BPF_ATTENTION_TOPK="${BPF_ATTENTION_TOPK:-0.2}"
METHOD="bpf"
OUTPUT_DIR="${OUTPUT_PREFIX}_${METHOD}"

source scripts/run_incremental.sh
