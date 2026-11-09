#!/bin/bash
# 域增量（DIL）实验队列：使用 GPU 0-7。
#
# 串行跑两条五阶段链：naive 链和 espreg+replay（增量模块）链。每条链的
# 前缀模型即各阶段统计所需的记录项（例如第 k 阶段的第 3 项就是增量链的
# task-k），因此不需要为每个阶段单独训练。
#
# 用法：bash scripts/417-ir/run_DIL_0-7.sh
# 环境变量：RUN_ROOT（默认 runs/417-ir）、FORCE=1 从头重跑、DEVICE 覆盖
#           GPU 列表，其余训练参数透传给各实验的 common.sh。
# 断点继承：已有的 task-*/best.pt 不会重训；中断的阶段从自己的
#           train/weights/last.pt 续训；缺失的 PCA cache / 回放数据集由
#           模型适配器重新生成。
set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_queue.sh" dil-0-7 "${DEVICE:-0,1,2,3,4,5,6,7}"
