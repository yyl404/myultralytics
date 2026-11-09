#!/bin/bash
# 汇总 417-ir 全部实验结果并显示在终端上；缺失的结果标记为“缺失”并跳过计算。
#
# 用法：bash scripts/417-ir/get_results.sh [--root DIR]
# 环境变量：RUN_ROOT（默认 runs/417-ir）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec python tools/ir_campaign.py report --root "${RUN_ROOT:-runs/417-ir}" "$@"
