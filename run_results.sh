#!/bin/bash
# 快速查看：汇总 417-ir 全部实验结果并显示在终端（缺失结果会标记并跳过）。
#
# 用法：bash run_results.sh [--root DIR]
# 环境变量透传：
#   RUN_ROOT=...   结果根目录（默认 runs/417-ir）
#   FORCE / DRY_RUN 对本脚本无效
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/417-ir/get_results.sh" "$@"
