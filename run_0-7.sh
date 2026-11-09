#!/bin/bash
# 快速启动：域增量（DIL）实验队列，GPU 0-7。
#
# 用法：bash run_0-7.sh
# 环境变量透传：
#   RUN_ROOT=...   结果根目录（默认 runs/417-ir）
#   FORCE=1        忽略已有断点，从头重跑
#   DRY_RUN=1      只打印将要执行的命令
#   DEVICE=...     覆盖 GPU 列表（默认 0,1,2,3,4,5,6,7）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/417-ir/run_DIL_0-7.sh"
