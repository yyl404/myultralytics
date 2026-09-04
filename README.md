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

OdinW-13 为预打包数据集，无需创建脚本，直接将 `OdinW-13-yolo/`（13 个子域目录，各含 `data.yaml`）放入 `data/` 即可。训练 / 评估时显式给出任务序列，如 `--tasks data/OdinW-13-yolo/*/data.yaml`（shell glob 按词典序展开）。

### 2.3 任意增量 yaml 序列

训练与评估彻底解耦，两者都只吃显式 yaml 序列（`--dataset` + `--split` 仅用于上面的 create 切分流程）：

- **训练**：`--tasks` 增量训练数据集序列（每任务一个 yaml，必需）；
- **评估**：`--tasks` 单任务评估数据集序列（必需）+ `--cumulative` 累积评估数据集序列（可选，不提供则不做累积评估）。

训练序列与评估序列不必相同，也不必等长：评估矩阵严格按 run 目录里实际的 `task-*/best.pt` 数量 × 实际给定的评估 yaml 数量构建（全 cross product）。`train.sh` / `eval.sh` / `predict.sh` / `feature_drift.sh` 均接受 `--tasks`（单个逗号分隔参数也可以）。

### 2.4 预训练权重

将以下权重文件放在**仓库根目录**（训练脚本按相对路径引用）：

- `yolov8x-cls.pt`：ImageNet 分类预训练，默认 `yolov8`（size x）
- `yoloe-v8m-seg.pt` / `yoloe-v8l-seg.pt`：YOLOE 分割预训练（`yolov8m` / `yoloe-v8`）
- `yoloe-26m-seg.pt`：YOLOE-26 分割预训练，默认 `yolo26m`
- `yolo26x.pt`：COCO 检测预训练，默认 `yolo26`（size x）

---

## 3. 训练与评估

训练与评估彻底解耦：训练只消费训练 yaml 序列并产出 `task-1`、`task-2`、…；评估在任意 yaml 序列上评测任意训练产物。`--dataset` + `--split` 不作为训练或评估入口（仅 `create.sh` 使用）。

### 3.0 快速上手：VOC-TINY 15+5 示范

`scripts/voc/tiny/15+5/` 提供一条龙管线与各环节的独立脚本（参数集中在 `common.sh`，迁移到其它增量数据集只需改这一个文件；所有参数也可用**同名环境变量**临时覆盖，列表参数用空格分隔字符串）：

```bash
# 前置：数据集已就绪（否则先 bash scripts/create.sh voc-tiny 15_5）
bash scripts/voc/tiny/15+5/pipeline.sh   # 训练 → 评估 → 有标签推理

# 或逐环节独立运行
bash scripts/voc/tiny/15+5/train.sh
bash scripts/voc/tiny/15+5/eval.sh
bash scripts/voc/tiny/15+5/predict.sh

# 环境变量覆盖示例（冒烟调试 / 临时换设置，无需改文件）
EPOCHS=1 MODEL=yolo26x RUN_DIR=runs/smoke bash scripts/voc/tiny/15+5/pipeline.sh
TASK_YAMLS="data/A/t1.yaml data/A/t2.yaml" bash scripts/voc/tiny/15+5/train.sh
```

约定：`yolo26m` + `yoloe-26m-seg.pt` 预训练；训练 / 评估 / 推理均开启 NMS 且为 `agnostic_nms`；评估与推理按任务数据集、任务累积数据集的顺序，对各 yaml 的 `test` 分割运行（无 `test` 则用 `val`）。

### 3.1 训练

统一入口，任意训练 yaml 序列 × 模型 × IOD 方法：

```bash
bash scripts/train.sh --tasks t1.yaml t2.yaml t3.yaml \
    --tag my-exp --model yolo26 --method pseudo_label+dist
```

`--tasks` 为唯一的数据入口（必需，每任务一个 yaml）；`--tag` 覆盖自动生成的 `DATA_TAG`（影响输出目录名）。训练不接收任何评估数据集参数。

`--method` 为 `+` 连接的组件，可任意组合：`naive`、`bpf`、`pseudo_label`、`ewc`、`l2`、`dist`、`espreg`、`nsgp`、`repre`、`replay`。

`--model` 为族名，可带尺寸后缀（`yolo26` / `yolo26m` / `yolov8x` / `yoloe-v8`）。`yoloe-v8` 默认 `l`，其余默认 `x`。

YOLO26 会自动加上 `--end2end False`（训练中验证走 one2many + NMS；`YOLO26_DEFAULT_HYPS=0` 关掉）。小数据集微调超参（如 voc-tiny 的 AdamW、`lr0=0.001`、`mosaic=0.5`、`freeze=10`）不再自动套用，按需在 `--` 后显式传入，参见 `scripts/voc/tiny/15+5/train.sh`。

示例：

```bash
# VOC-TINY 15+5，yolo26m + yoloe-26m-seg，伪标签 + dist + espreg
bash scripts/train.sh \
    --tasks data/VOC-TINY_15+5/task_1_cls_15/dataset.yaml data/VOC-TINY_15+5/task_2_cls_5/dataset.yaml \
    --tag VOC-TINY_15+5 --model yolo26m --method pseudo_label+dist+espreg

# OdinW-13 任务增量，伪标签 + NSGP + RePRE
bash scripts/train.sh --tasks data/OdinW-13-yolo/*/data.yaml --model yolov8 --method pseudo_label+nsgp+repre
```

| 变量 | 含义 | 默认值 |
|---|---|---|
| `EPOCHS` | 每任务训练轮数 | 100 |
| `BATCH_SIZE` / `IMGSZ` / `WORKERS` / `DEVICE` | 训练超参 | 16 / 640 / 8 / 0 |
| `START_TASK` | 从第几个任务开始（断点续跑） | 1 |
| `END_TASK` | 到第几个任务结束（部分运行/调试） | 任务总数 |

`DEVICE` 可以是多卡列表（如 `0,1`），训练走 DDP；统计类工具（importance / PCA / 原型）自动只用首卡（可用 `TOOL_DEVICE` 覆盖）。

```bash
# 示例：只跑前 2 个任务、每任务 1 个 epoch（冒烟调试）
EPOCHS=1 END_TASK=2 bash scripts/train.sh --tasks data/OdinW-13-yolo/*/data.yaml --model yolov8 --method naive
```

训练产物保存在 `runs/<MODEL_ID>_<DATA_TAG>_pretrained-from-<weights>_<method>/task-<k>/`（`best.pt`、EWC 的 `importance.pth`、ESPReg/NSGP 的 `pca_cache.pkl`、RePRE 的 `repre_prototypes.pt` 等）。训练器中间产物（`weights/`、`results.csv`、曲线图等）保存在 `task-<k>/train/` 下，重跑同一任务前会清空重建，不会累积 `train2/train3/...`。

类别空间约定：任务 `k>1` 开始时由 `tools/expand_model_head.py` 扩展检测头——既有类别的 id 与在检测头中的顺序保持不变，新数据集中未见过的类别按其 yaml 中的顺序追加在最后；若新数据集含有与既有类别同名的类别，则不新增通道，其标注由 `tools/convert_dataset_class_ids.py` 按类别名统一对齐到既有 id。训练、评估与推理用到的数据集都会先按类别名对齐到当前模型的类别空间，DDP 各 rank 加载同一扩展权重与同一转换后数据集，类别空间天然一致。此外，每个 `best.pt` 都以模块属性 `incremental_history`（`[{"task": k, "names": [...]}]`，每增量阶段一条）携带自身经历的类别空间历史：任务 1 由 `tools/train.py` 在保存时写入，后续任务由 `expand_model_head.py` 追加；评估侧因此无需假设评估用任务数据集与训练时一致。

解码配置一致性：`end2end` / `agnostic_nms` / `max_det` 是 Detect 头上的 Python 属性，不在 state_dict 中；从 yaml 重建检测头会回落到 yaml 默认值（`yolo26*.yaml` 为 `end2end=True`），若不加处理，扩展后的模型会静默改走未训练的 one2one 分支（伪标签、抗遗忘随之失效）。管线各环节以「当前阶段实际生效的训练/评估参数 + 上一阶段 checkpoint 保存的属性」为准：`tools/expand_model_head.py` 扩头时把源模型检测头的这三个属性随权重一起复制到扩展模型；所有"yaml 重建 + 权重迁移"的路径（训练器 `setup_model`、`Model.train` 内部预重建、蒸馏教师重建）经 `BaseModel.load`（`ultralytics/nn/tasks.py`）从源模型继承这些属性，显式传入的 `--end2end` 等训练参数仍在其后优先生效；AntiForget/BPF 的冻结教师与参考模型加载后套用当前训练参数（`ultralytics/engine/anti_forget.py` 的 `_apply_train_head_args`），且教师的 `end2end` 与学生不一致时立即报错（给出期望 vs 实际），不会静默回退到 yaml 默认。评估与推理沿用 checkpoint 中保存的属性，经 `--` 显式透传的 `--end2end` / `--agnostic_nms` / `--max_det` 优先。

### 3.2 评估

传入训练 run 目录 + 显式评估 yaml 序列（评估不读训练产物中的任何数据清单，序列完全由评估命令决定）：

```bash
bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml
bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml --cumulative c1.yaml c2.yaml
# COCO 等需要额外 IoU 阈值列时：
bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml --iou-threshold 0.75
# NMS / agnostic 等预测参数经 -- 透传给 tools/eval.py：
bash scripts/eval.sh runs/<run> --tasks e1.yaml e2.yaml -- --agnostic_nms True
```

结果矩阵严格按实际构建：run 目录中每个 `task-k/best.pt` × 每个评估 yaml（per-task 与 cumulative 序列各自做全 cross product），两个序列的长度与顺序都不必与训练序列一致。每个 yaml 默认在 `test` 分割上评估，无 `test` 键则用 `val`（`--split` 可覆盖）。模型类空间与某评估数据集完全不相交的格子产出空 CSV，在表中标为 `N/A`。

结果写入 `<run>/evaluation_results/`：逐类指标 CSV（`model_<k>_eval_task_<j>.csv` / `model_<k>_eval_cumulative_<j>.csv`）、混淆矩阵 CSV、矩阵表 `individual_datasets_eval.csv` 与 `cumulative_datasets_eval.csv`，以及按增量阶段聚合的 mAP 表——`tools/stage_task_map.py` 对每个评估 CSV 用对应 checkpoint 自带的 `incremental_history` 划分阶段类别空间，产出 `<同名>_stage_mAP.csv` 与汇总 `stage_mAP_sequence.csv`，不依赖评估时的任务数据集划分。评估器中间产物（`<eval>/model_*_eval_*/`）在每次重跑评估前清空重建。

### 3.3 推理（predict）

```bash
# 任意图像目录，纯推理（无 GT 比较）
python tools/predict.py --model runs/<run>/task-2/best.pt --images some/images

# 提供标签目录：输出 TP/FP/FN/Precision/Recall/F1（metrics.csv）并按 TP/FP/FN 分类可视化
python tools/predict.py --model runs/<run>/task-2/best.pt --images some/images --labels some/labels

# 对一串数据集 yaml 批量推理（类别 id 先按名对齐到模型空间；默认 test 分割，无 test 用 val）
bash scripts/predict.sh --model runs/<run>/task-2/best.pt --tasks t1.yaml t2.yaml --cumulative c1.yaml c2.yaml
```

---

## 4. 分析工具

### 4.1 特征漂移（feature drift）

量化相邻任务 checkpoint 间 backbone 特征的漂移（方向/幅度分解），在 task-1 图像上计算：

```bash
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
    --confusion_matrix_path runs/<run>/evaluation_results/model_2_eval_cumulative_2_confusion_matrix.csv \
    --old_classes aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person \
    --new_classes pottedplant sheep sofa train tvmonitor \
    --save_dir runs/<run>/confusion_analysis
```

各工具的完整参数可用 `python tools/<对应工具>.py --help` 查看。

---

## 5. 测试与代码结构

```bash
pytest tests/test_anti_forget.py tests/test_bpf.py tests/test_ewc.py
```

仓库结构与重构记录见 [REFACTORING.md](REFACTORING.md)；脚本目录规范见 [skills/scripts_structure_skill.md](skills/scripts_structure_skill.md)；项目设计文档与实验记录归档在 [docs/project/](docs/project/)。

## License

本仓库基于 Ultralytics 修改，遵循 [AGPL-3.0](LICENSE) 许可。
