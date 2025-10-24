#!/usr/bin/env bash

set -euo pipefail

# 域增量顺序：VOC -> CLIPART -> WATERCOLOR -> COMIC
# 依赖：tools/train_incremental.py 已支持 --replay_enable 与 VAE 回放参数

# 可覆盖变量
ROOT=${ROOT:-/hy-tmp}
PROJ=${PROJ:-/hy-tmp/myultralytics}
SAVE_ROOT=${SAVE_ROOT:-$PROJ/runs/domain-incremental}
VAE_CKPT=${VAE_CKPT:-/root/vae-search/logs/best.pt}
VAE_ARCH=${VAE_ARCH:-vq} # vq | vanilla
REPLAY_CONF=${REPLAY_CONF:-0.25}

# 可选断点：未提供则为空，避免 set -u 报错
CHECKPOINT=${CHECKPOINT:-}

# vocseries 数据根（四个域都在此目录）
VOCSERIES_ROOT=${VOCSERIES_ROOT:-$ROOT/vocseries}

# 四个域的数据配置（按实际目录名：voc/clipart/watercolor/comic）
CFG_VOC=${CFG_VOC:-$VOCSERIES_ROOT/voc/dataset.yaml}
CFG_CLIPART=${CFG_CLIPART:-$VOCSERIES_ROOT/clipart/dataset.yaml}
CFG_WATERCOLOR=${CFG_WATERCOLOR:-$VOCSERIES_ROOT/watercolor/dataset.yaml}
CFG_COMIC=${CFG_COMIC:-$VOCSERIES_ROOT/comic/dataset.yaml}

# 模型结构配置与初始teacher（可根据需要调整）
MODEL_CFG=${MODEL_CFG:-yolov8l.yaml}
# 阶段1 teacher 默认用官方预训练（将自动下载）；如有自有权重可覆盖
INIT_TEACHER=${INIT_TEACHER:-yolov8l.pt}
# 阶段1 base（可选）：若提供将作为增量起点；默认从零开始（base_model=null）
INIT_BASE=${INIT_BASE:-}

# 每个阶段VSPReg/PCA配置（可选）。如已有 YAML，可直接传 --cfg 使用。
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
DEVICE=${DEVICE:-0}

mkdir -p "$SAVE_ROOT"

train_stage() {
  local STAGE_NAME=$1
  local DATA_CFG=$2
  local SAVE_DIR=$3
  local BASE_MODEL=$4
  local TEACHER=$5
  local REPLAY_SRC=$6

  mkdir -p "$SAVE_DIR"

  # 生成一个临时 YAML 配置，复用现有 train_incremental.py 的配置读取逻辑
  local TMP_CFG="$SAVE_DIR/train_cfg.yaml"

  # 处理 base_model 字段：若为空或文件不存在，则写入 null
  local BASE_FIELD
  if [[ -z "${BASE_MODEL}" ]]; then
    BASE_FIELD=null
  elif [[ -f "${BASE_MODEL}" || "${BASE_MODEL}" == *.pt ]]; then
    BASE_FIELD=${BASE_MODEL}
  else
    BASE_FIELD=null
  fi

  cat > "$TMP_CFG" <<EOF
base_model: ${BASE_FIELD}
model_cfg: ${MODEL_CFG}
data_cfg: ${DATA_CFG}
teacher_model: ${TEACHER}
epochs: ${EPOCHS}
batch: ${BATCH}
workers: ${WORKERS}
device: ${DEVICE}
frozen_layers: [0,1,2,3,4,5,6,7,8,9]
pca_sample_num: 100
projection_layers:
pca_sample_images:
pca_sample_labels:
pca_cache_save_path:
pca_cache_load_path:
EOF

  # 阶段训练：若提供 REPLAY_SRC 则启用 VAE 回放
  if [[ -n "${REPLAY_SRC}" && -f "${VAE_CKPT}" ]]; then
    CMD=(python "$PROJ/tools/train_incremental.py" --cfg "$TMP_CFG" --save_dir "$SAVE_DIR" --replay_enable \
      --vae_ckpt "$VAE_CKPT" --vae_arch "$VAE_ARCH" --replay_source_images "$REPLAY_SRC" --replay_conf_threshold "$REPLAY_CONF")
    if [[ -n "${CHECKPOINT}" ]]; then CMD+=(--checkpoint "$CHECKPOINT"); fi
    "${CMD[@]}"
  else
    CMD=(python "$PROJ/tools/train_incremental.py" --cfg "$TMP_CFG" --save_dir "$SAVE_DIR")
    if [[ -n "${CHECKPOINT}" ]]; then CMD+=(--checkpoint "$CHECKPOINT"); fi
    "${CMD[@]}"
  fi
}

# 阶段1：VOC（无回放；base=null 或使用 INIT_BASE 覆盖）
STAGE1_DIR="$SAVE_ROOT/stage1_voc"
train_stage "VOC" "$CFG_VOC" "$STAGE1_DIR" "${INIT_BASE}" "$INIT_TEACHER" ""

# 更新上一阶段最优权重作为下一阶段 base/teacher
S1_BEST="$STAGE1_DIR/best.pt"

# 阶段2：CLIPART（开启VAE回放，回放源使用上一域的训练图像）
STAGE2_DIR="$SAVE_ROOT/stage2_clipart"
# 推断上一域train图目录（dataset.yaml -> path/train）
CLIPART_REPLAY_SRC="$(python - <<PY
from ultralytics.utils import YAML
import os
cfg=YAML.load('$CFG_VOC')
root=os.path.dirname('$CFG_VOC')
img=os.path.join(cfg.get('path', root), cfg['train']) if 'path' in cfg else os.path.join(root,cfg['train'])
print(img)
PY
)"
train_stage "CLIPART" "$CFG_CLIPART" "$STAGE2_DIR" "$S1_BEST" "$S1_BEST" "$CLIPART_REPLAY_SRC"

S2_BEST="$STAGE2_DIR/best.pt"

# 阶段3：WATERCOLOR（回放源=前一域CLIPART的train图）
STAGE3_DIR="$SAVE_ROOT/stage3_watercolor"
WATERCOLOR_REPLAY_SRC="$(python - <<PY
from ultralytics.utils import YAML
import os
cfg=YAML.load('$CFG_CLIPART')
root=os.path.dirname('$CFG_CLIPART')
img=os.path.join(cfg.get('path', root), cfg['train']) if 'path' in cfg else os.path.join(root,cfg['train'])
print(img)
PY
)"
train_stage "WATERCOLOR" "$CFG_WATERCOLOR" "$STAGE3_DIR" "$S2_BEST" "$S2_BEST" "$WATERCOLOR_REPLAY_SRC"

S3_BEST="$STAGE3_DIR/best.pt"

# 阶段4：COMIC（回放源=前一域WATERCOLOR的train图）
STAGE4_DIR="$SAVE_ROOT/stage4_comic"
COMIC_REPLAY_SRC="$(python - <<PY
from ultralytics.utils import YAML
import os
cfg=YAML.load('$CFG_WATERCOLOR')
root=os.path.dirname('$CFG_WATERCOLOR')
img=os.path.join(cfg.get('path', root), cfg['train']) if 'path' in cfg else os.path.join(root,cfg['train'])
print(img)
PY
)"
train_stage "COMIC" "$CFG_COMIC" "$STAGE4_DIR" "$S3_BEST" "$S3_BEST" "$COMIC_REPLAY_SRC"

echo "Domain incremental training finished. Best checkpoints:"
echo "  VOC       : $S1_BEST"
echo "  CLIPART   : $S2_BEST"
echo "  WATERCOLOR: $S3_BEST"
echo "  COMIC     : $STAGE4_DIR/best.pt"


