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

先准备完整类别的标准 YOLO 格式数据集（`images/{train,val,test}` + `labels/{train,val,test}` + 含 `train/val/test/names` 字段的 yaml），然后用统一入口切分：

```bash
# VOC-TINY 15+5：先从 VOC 抽 25% 子样本，再切分
bash scripts/create.sh voc-tiny 15_5
# 输出：data/VOC-TINY_15+5/task_1_cls_15、task_2_cls_5、task_1-2_cls_20（累积集）

bash scripts/create.sh voc 15_5
bash scripts/create.sh coco 70_10
```

`--split` 为下划线连接的每任务类别数，对应 `data/` 下的 `+` 目录名（`15_5` → `VOC_15+5`）。任意正整数序列均可，不必再加脚本。

| 数据集 | `--dataset` | 常用 split | 源 yaml |
|---|---|---|---|
| VOC | `voc` | `10_10` / `15_5` / `19_1` / `10_5_5` / `5_5_5_5` / `10_2_2_2_2_2` | `data/VOC-YOLO/VOC.yaml` |
| VOC-TINY | `voc-tiny` | `15_5` | 先 subsample 到 `data/VOC-TINY-YOLO/` |
| COCO | `coco` | `40_40` / `70_10` | `data/coco-yolo/coco.yaml` |

### 2.2 任务增量数据集（OdinW-13）

OdinW-13 为预打包数据集，无需创建脚本，直接将 `OdinW-13-yolo/`（13 个子域目录，各含 `data.yaml`）放入 `data/` 即可。任务顺序为子域名的词典序。

### 2.3 任意增量 yaml 序列

不注册数据集家族，也可以直接用三串 yaml 序列驱动实验（`data-split` 两级划分只是生成这些序列的一种方式）：

- `--tasks`：增量训练数据集序列（每任务一个 yaml，必需）；
- `--eval-tasks`：单任务评估数据集序列（可选，默认与训练序列相同）；
- `--cumulative`：累积任务评估数据集序列（可选，每任务一个；不提供则不做累积评估）。

三串序列长度都必须等于任务数。`train.sh` / `eval.sh` / `detect.sh` / `feature_drift.sh` 均接受 `--tasks`。训练时会把解析后的序列写入 `<output>/task_yamls.txt`、`eval_yamls.txt`、`cumulative_yamls.txt` 与 `experiment.meta`，之后 `eval.sh runs/<run>` 无需任何参数即可恢复。

### 2.4 预训练权重

将以下权重文件放在**仓库根目录**（训练脚本按相对路径引用）：

- `yolov8x-cls.pt`：ImageNet 分类预训练，默认 `yolov8`（size x）
- `yoloe-v8m-seg.pt` / `yoloe-v8l-seg.pt`：YOLOE 分割预训练（voc-tiny 的 yolov8m / `yoloe-v8`）
- `yoloe-26m-seg.pt`：YOLOE-26 分割预训练，默认 voc-tiny 的 `yolo26`
- `yolo26x.pt`：COCO 检测预训练，默认非 tiny 的 `yolo26`（size x）

---

## 3. 训练与评估

### 3.1 训练

统一入口，任意数据集 × 模型 × IOD 方法：

```bash
bash scripts/train.sh --dataset <ds> --split <split> --model <model> --method <method>
bash scripts/train.sh <ds> <split> <model> <method>

# 或者直接给任意 yaml 序列（--tasks 与 --dataset/--split 互斥，位置参数只剩 model、method）
bash scripts/train.sh --tasks t1.yaml t2.yaml t3.yaml \
    --eval-tasks e1.yaml e2.yaml e3.yaml \
    --cumulative c1.yaml c2.yaml c3.yaml \
    --tag my-exp --model yolo26 --method pseudo_label+dist
```

`--eval-tasks` 省略时逐任务评估复用训练序列；`--cumulative` 省略则不做累积评估。`--tag` 覆盖自动生成的 `DATA_TAG`（影响输出目录名）。

`--method` 为 `+` 连接的组件，可任意组合：`naive`、`bpf`、`pseudo_label`、`ewc`、`l2`、`dist`、`espreg`、`nsgp`、`repre`、`replay`。

`--model` 为族名，可带尺寸后缀（`yolo26` / `yolo26m` / `yolov8x` / `yoloe-v8`）。voc-tiny 默认 size `m`，`yoloe-v8` 默认 `l`，其余默认 `x`。

YOLO26 会自动加上 `--end2end False`；在 voc-tiny 上另外使用 AdamW、`lr0=0.001`、`mosaic=0.5`、`freeze=10`（可用环境变量覆盖，或 `YOLO26_DEFAULT_HYPS=0` 关掉）。

示例：

```bash
# VOC-TINY 15+5，yolo26m + yoloe-26m-seg，伪标签 + dist + espreg
bash scripts/train.sh voc-tiny 15_5 yolo26 pseudo_label+dist+espreg

# VOC 15+5，yoloe-v8，伪标签 + EWC
bash scripts/train.sh voc 15_5 yoloe-v8 pseudo_label+ewc

# OdinW-13 任务增量，伪标签 + NSGP + RePRE
bash scripts/train.sh odinw-13 13 yolov8 pseudo_label+nsgp+repre
```

| 变量 | 含义 | 默认值 |
|---|---|---|
| `EPOCHS` | 每任务训练轮数 | voc 100 / voc-tiny 10 / coco 12 / odinw-13 100 |
| `BATCH_SIZE` / `IMGSZ` / `WORKERS` / `DEVICE` | 训练超参 | 16 / 640 / 8 / 0 |
| `START_TASK` | 从第几个任务开始（断点续跑） | 1 |
| `END_TASK` | 到第几个任务结束（部分运行/调试） | 任务总数 |

`DEVICE` 可以是多卡列表（如 `0,1`），训练走 DDP；统计类工具（importance / PCA / 原型）自动只用首卡（可用 `TOOL_DEVICE` 覆盖）。

```bash
# 示例：只跑前 2 个任务、每任务 1 个 epoch（冒烟调试）
EPOCHS=1 END_TASK=2 bash scripts/train.sh odinw-13 13 yolov8 naive
```

训练产物保存在 `runs/<MODEL_ID>_<DATA_TAG>_pretrained-from-<weights>_<method>/task-<k>/`（`best.pt`、EWC 的 `importance.pth`、ESPReg/NSGP 的 `pca_cache.pkl`、RePRE 的 `repre_prototypes.pt` 等）。

类别空间约定：任务 `k>1` 开始时由 `tools/expand_model_head.py` 扩展检测头——既有类别的 id 与在检测头中的顺序保持不变，新数据集中未见过的类别按其 yaml 中的顺序追加在最后；若新数据集含有与既有类别同名的类别，则不新增通道，其标注由 `tools/convert_dataset_class_ids.py` 按类别名统一对齐到既有 id。训练、评估与推理用到的数据集都会先按类别名对齐到当前模型的类别空间，DDP 各 rank 加载同一扩展权重与同一转换后数据集，类别空间天然一致。

### 3.2 评估

传入任意训练 run 目录即可（优先读训练时写入的 manifest，其次从目录名推断注册数据集/split）：

```bash
bash scripts/eval.sh runs/<run>
bash scripts/eval.sh --dataset voc-tiny --split 15_5 --run runs/<run>
# 自定义序列也可以显式覆盖（三个 flag 可独立使用）
bash scripts/eval.sh --tasks t1.yaml t2.yaml --cumulative c1.yaml c2.yaml --run runs/<run>
```

对每个任务的 `best.pt` 评估其已见各任务（有序列时另含累积数据集），结果写入 `<run>/evaluation_results/`：逐类指标 CSV、混淆矩阵 CSV、`individual_datasets_eval.csv`、`cumulative_datasets_eval.csv` 与按任务汇总的 mAP 表。

---

## 4. 分析工具

### 4.1 特征漂移（feature drift）

量化相邻任务 checkpoint 间 backbone 特征的漂移（方向/幅度分解），在 task-1 图像上计算：

```bash
bash scripts/feature_drift.sh voc-tiny 15_5 \
    runs/<run>/task-1/best.pt runs/<run>/task-2/best.pt [save_path]

# 自定义序列：用 --tasks 提供 yaml（其余参数走 flag）
bash scripts/feature_drift.sh --tasks t1.yaml t2.yaml \
    --model1 runs/<run>/task-1/best.pt --model2 runs/<run>/task-2/best.pt
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
