# myultralytics：YOLO 增量学习实验框架

基于 [Ultralytics](https://github.com/ultralytics/ultralytics)（8.3.x）修改的增量学习（Incremental Learning / Incremental Object Detection）研究代码库。在 YOLOv8 / YOLOE-v8 检测器上实现并支持以下抗遗忘方法及其组合：

- **naive**：直接微调（下界基线）
- **pseudo_label**：旧模型伪标签与 GT 合并训练
- **ewc**：Elastic Weight Consolidation（逐任务对角 Fisher 二次惩罚）
- **l2**：向参考模型的朴素参数距离正则
- **espreg**：特征值缩放投影正则（基于 PCA 缓存的特征投影正则）
- **nsgp**：零空间梯度投影（可搭配 **repre** 区域原型回放）
- **distillation**（dist）：教师 top-k 类别通道 KL 蒸馏
- **bpf**：Bridge Past and Future（伪标签分档加权 + Bridge Future 忽略掩码 + DwF 蒸馏）

支持设定：类增量（CIL，VOC / COCO / VOC-TINY 多种 split）与任务增量（TIL，OdinW-13）。

---

## 1. 环境配置（conda，python=3.9，cuda=11.8）

```bash
# 创建环境（默认环境名 yolo）
conda create -n yolo python=3.9 -y
conda activate yolo

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 以可编辑模式安装本仓库（含 ultralytics 全部依赖）
cd myultralytics
pip install -e .

# 增量学习管线与分析工具的额外依赖
pip install joblib seaborn

# （可选）运行单元测试
pip install pytest
```

要求：NVIDIA GPU + 兼容 CUDA 11.8 的驱动。全部命令均需在**仓库根目录**下执行。

---

## 2. 数据集准备

所有数据集放在 `data/` 目录下（可以是真实目录或符号链接），每个数据集一个子目录。

### 2.1 类增量数据集（VOC / COCO / VOC-TINY）

先准备完整类别的标准 YOLO 格式数据集（`images/{train,val,test}` + `labels/{train,val,test}` + 含 `train/val/test/names` 字段的 yaml），然后用各 split 的创建脚本切分为多阶段类增量数据集：

```bash
# 以 VOC-TINY 15+5 为例：源数据集为 data/VOC-TINY-YOLO/（yaml 为 VOC.yaml）
bash scripts/voc-tiny/15_5/create_voc-tiny_15_5.sh
# 输出：data/VOC-TINY_15+5/task_1_cls_15、task_2_cls_5、task_1-2_cls_20（累积集）
```

其余 split 同理（源数据集路径在各 `create_*.sh` 内指定，可按需修改）：

| 数据集 | split | 创建脚本 |
|---|---|---|
| VOC | 10+10 / 15+5 / 19+1 / 10+5+5 / 5+5+5+5 / 10+2+2+2+2+2 | `scripts/voc/<split>/create_voc_<split>.sh` |
| VOC-TINY | 15+5 | `scripts/voc-tiny/15_5/create_voc-tiny_15_5.sh` |
| COCO | 40+40 / 70+10 | `scripts/coco/<split>/create_coco_<split>.sh` |

### 2.2 任务增量数据集（OdinW-13）

OdinW-13 为预打包数据集，无需创建脚本，直接将 `OdinW-13-yolo/`（13 个子域目录，各含 `data.yaml`）放入 `data/` 即可。任务顺序为子域名的词典序。

### 2.3 预训练权重

将以下权重文件放在**仓库根目录**（训练脚本按相对路径引用）：

- `yolov8x-cls.pt`：ImageNet 分类预训练权重，`yolov8` baseline 的检测初始化
- `yoloe-v8l-seg.pt`：YOLOE 分割预训练权重，`yoloe-v8` baseline 的初始化
- `yolo26x.pt`：COCO 检测预训练权重，`yolo26` baseline（yolo26x）的初始化

---

## 3. 训练与评估

### 3.1 训练

每个 `<dataset>/<split>/<baseline>` 目录下每个方法一个启动脚本：

```bash
bash scripts/<dataset>/<split>/<baseline>/train_<method>.sh
```

`<method>` 取值（9 个）：`naive`、`pseudo_label`、`pseudo_label+ewc`、`pseudo_label+l2`、`pseudo_label+espreg`、`pseudo_label+dist+espreg`、`pseudo_label+nsgp`、`pseudo_label+nsgp+repre`、`bpf`。

注：`yolo26` baseline（end2end 检测头）当前仅提供 `pseudo_label+dist+espreg` 启动脚本。

示例：

```bash
# VOC-TINY 15+5，yolov8 baseline，naive 微调
bash scripts/voc-tiny/15_5/yolov8/train_naive.sh

# VOC 15+5，yoloe-v8 baseline，伪标签 + EWC
bash scripts/voc/15_5/yoloe-v8/train_pseudo_label+ewc.sh

# OdinW-13 任务增量，伪标签 + NSGP + RePRE
bash scripts/odinw-13/13/yolov8/train_pseudo_label+nsgp+repre.sh
```

通用环境变量（在各 `config.sh` / `run_incremental.sh` 中生效）：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `EPOCHS` | 每任务训练轮数 | 各 config.sh 内指定 |
| `BATCH_SIZE` / `IMGSZ` / `WORKERS` / `DEVICE` | 训练超参 | 16 / 640 / 8 / 0 |
| `START_TASK` | 从第几个任务开始（断点续跑） | 1 |
| `END_TASK` | 到第几个任务结束（部分运行/调试） | 任务总数 |

```bash
# 示例：只跑前 2 个任务、每任务 1 个 epoch（冒烟调试）
EPOCHS=1 END_TASK=2 bash scripts/odinw-13/13/yolov8/train_naive.sh
```

训练产物保存在 `runs/<OUTPUT_PREFIX>_<method>/task-<k>/`（`best.pt`、EWC 的 `importance.pth`、ESPReg/NSGP 的 `pca_cache.pkl`、RePRE 的 `repre_prototypes.pt` 等）。

### 3.2 评估

评估脚本是 split 级、模型无关的，传入任意训练 run 的输出目录即可：

```bash
bash scripts/<dataset>/<split>/eval.sh runs/<OUTPUT_PREFIX>_<method>
```

对每个任务的 `best.pt` 评估其已见各任务（CIL 另含累积数据集），结果写入 `<run>/evaluation_results/`：逐类指标 CSV、混淆矩阵 CSV、`individual_datasets_eval.csv`、`cumulative_datasets_eval.csv` 与按任务汇总的 mAP 表。

---

## 4. 分析工具

### 4.1 特征漂移（feature drift）

量化相邻任务 checkpoint 间 backbone 特征的漂移（方向/幅度分解），数据集按 split 固定：

```bash
bash scripts/<dataset>/<split>/feature_drift.sh \
    runs/<run>/task-1/best.pt runs/<run>/task-2/best.pt [save_path]
```

### 4.2 其余分析工具（统一入口 `scripts/analyze.sh`）

```bash
bash scripts/analyze.sh <analysis> [工具参数...]
```

| analysis | 功能 | 所需产物 |
|---|---|---|
| `pca_projection` | 核更新/值偏移在 PCA 主成分上的投影分析（4 个 stage，`tools/vis.py`） | pca_cache + 两个 checkpoint |
| `kernel_projection` | 核更新在主成分上的投影曲线（含方差分布、拐点标注） | pca_cache + 两个 checkpoint |
| `eigen_adjust` | ESPReg 特征值 log+sigmoid 调整可视化 | pca_cache |
| `prototypes` | RePRE 原型经检测头回放的可视化验证 | checkpoint + repre_prototypes.pt |
| `confusion_matrix` | 混淆矩阵聚合为 old/new/background 三态并出图（需 seaborn） | 评估产物的 confusion matrix CSV |

示例：

```bash
# stage 1/2（方差/核更新投影直方图、极坐标图）：直接使用管线产出的 unfold 模式 pca_cache，需加 --unfold
bash scripts/analyze.sh pca_projection \
    --pca_cache_path runs/<run>/task-1/pca_cache.pkl \
    --base_model runs/<run>/task-1/best.pt \
    --incremental_model runs/<run>/task-2/best.pt \
    --save_dir runs/<run>/vis_task2_on_task1 \
    --stages 1 2 --unfold

# stage 3/4（值偏移、输入/核投影长度）：需要 fold 模式的 pca_cache，先用 pca.py 生成
python tools/pca.py --model runs/<run>/task-1/best.pt \
    --dataset data/VOC-TINY_15+5/task_1_cls_15/dataset.yaml \
    --save_path runs/<run>/task-1/pca_cache_fold.pkl --exclude_head --mode fold
bash scripts/analyze.sh pca_projection \
    --pca_cache_path runs/<run>/task-1/pca_cache_fold.pkl \
    --base_model runs/<run>/task-1/best.pt \
    --incremental_model runs/<run>/task-2/best.pt \
    --save_dir runs/<run>/vis_task2_on_task1 \
    --stages 3 4 \
    --sample_dir data/VOC-TINY_15+5/task_1_cls_15/images/val \
    --label_dir data/VOC-TINY_15+5/task_1_cls_15/labels/val

# ESPReg 特征值调整可视化
bash scripts/analyze.sh eigen_adjust \
    --pca_cache runs/<run>/task-1/pca_cache.pkl --save_dir runs/<run>/eigen_adjust

# RePRE 原型回放可视化
bash scripts/analyze.sh prototypes \
    --model runs/<run>/task-2/best.pt \
    --prototypes runs/<run>/task-1/repre_prototypes.pt \
    --output runs/<run>/prototypes_vis

# 混淆矩阵三态聚合（VOC 15+5 示例：前 15 类为 old，后 5 类为 new）
bash scripts/analyze.sh confusion_matrix \
    --confusion_matrix_path runs/<run>/evaluation_results/model_2_eval_cumulative_confusion_matrix.csv \
    --old_classes aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person \
    --new_classes pottedplant sheep sofa train tvmonitor \
    --save_dir runs/<run>/confusion_analysis
```

各工具的完整参数可用 `python tools/<对应工具>.py --help` 查看。

---

## 5. 测试与代码结构

```bash
pytest tests/test_bpf.py tests/test_ewc.py
```

仓库结构与重构记录见 [REFACTORING.md](REFACTORING.md)；脚本目录规范见 [skills/scripts_structure_skill.md](skills/scripts_structure_skill.md)；项目设计文档与实验记录归档在 [docs/project/](docs/project/)。

## License

本仓库基于 Ultralytics 修改，遵循 [AGPL-3.0](LICENSE) 许可。
