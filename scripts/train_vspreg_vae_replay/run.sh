#!/usr/bin/env bash

set -euo pipefail

# 可覆盖变量
CFG_PATH=${CFG_PATH:-/hy-tmp/myultralytics/configs/train_naive_voc_15_5/train_2.yaml}
SAVE_DIR=${SAVE_DIR:-/hy-tmp/myultralytics/runs/exp-vspreg-vae-replay}
CHECKPOINT=${CHECKPOINT:-}
VAE_CKPT=${VAE_CKPT:-/root/vae-search/logs/best.pt}
VAE_ARCH=${VAE_ARCH:-vq} # vq | vanilla
REPLAY_SOURCE_IMAGES=${REPLAY_SOURCE_IMAGES:-/hy-tmp/myultralytics/data/VOC_inc_15_5/task_1_cls_15/images/train}
REPLAY_CONF=${REPLAY_CONF:-0.25}

python /hy-tmp/myultralytics/tools/train_incremental.py \
  --cfg "$CFG_PATH" \
  --save_dir "$SAVE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --replay_enable \
  --vae_ckpt "$VAE_CKPT" \
  --vae_arch "$VAE_ARCH" \
  --replay_source_images "$REPLAY_SOURCE_IMAGES" \
  --replay_conf_threshold "$REPLAY_CONF"


