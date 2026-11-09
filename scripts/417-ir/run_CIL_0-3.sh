#!/bin/bash
# 类增量（CIL）实验队列：使用 GPU 0-3。
#
# 队列按第二阶段的数据集族切分：本脚本只跑 hf 族实验（hf、hf-400、
# hf-pruned-400、bk+hm-400->hf-400），联合族实验由 run_CIL_4-7.sh 在
# GPU 4-7 上同时跑，因此两组进程不会并行读取同一批训练数据（第一阶段
# 的 bk+hm 是两条队列共有的，无法避免）。
#
# 用法：bash scripts/417-ir/run_CIL_0-3.sh
# 环境变量：RUN_ROOT（默认 runs/417-ir）、FORCE=1 从头重跑、DEVICE 覆盖
#           GPU 列表，其余训练参数透传给各实验的 common.sh。
# 断点继承：已有的 task-*/best.pt 不会重训；中断的阶段从自己的
#           train/weights/last.pt 续训；缺失的 PCA cache / 回放数据集由
#           模型适配器重新生成。
set -euo pipefail

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_queue.sh" cil-0-3 "${DEVICE:-0,1,2,3}"
