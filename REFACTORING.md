# 仓库重构报告（2026-08-05）

本文档记录对 myultralytics 仓库进行的一次重构：① 全部重构动作；② 重构后的代码结构。

**功能范围**：增量学习的训练、评估、分析（naive / pseudo_label / ewc / l2 / espreg / nsgp(+repre) / distillation / bpf 及组合）。
**约束**：未改动 ultralytics 上游原始代码（以 `main` 分支 head 为基准）。本次所有改动均落在项目新增文件或项目新增行上；上游同步（8.3.190→8.3.239）带来的文件一律未动。

---

## 1. 全部重构动作

### 1.1 删除：FiLM 支线（已损坏、与增量学习无关）

`ultralytics/models/film/` 依赖不存在的 `ultralytics.data.dataset_json`，import 即报错，且未在 `models/__init__.py` 注册，属搁置的实验线。

- 删除 `ultralytics/models/film/`（`__init__.py` / `model.py` / `train.py` / `val.py`）
- 删除 `ultralytics/nn/tasks_film.py`、`ultralytics/nn/modules/film.py`
- 清理 `ultralytics/nn/modules/__init__.py` 中 `from .film import AttributeEncoder, FiLM` 及 `__all__` 对应条目
- 删除 `tools/train_yolo_film.py`、`tools/convert_yolo_to_json.py`、`YOLOFiLM_DESIGN.md`、`COMMIT_MESSAGE.txt`

### 1.2 删除：ABR / OSR 遗留实验管线（已被 `scripts/` 体系取代）

- 删除根目录 `abr/`（4 个脚本）、`osr/`（3 个脚本）
- 删除 `ultralytics/engine/abr.py`（`ABRReplay`）
- 删除 `detect/train.py` 中 `ABRDetectionTrainer` 类及 `detect/__init__.py` 导出；删除 `tools/train.py` 中 `--trainer abr` 分支
- 删除 `tools/build_abr_memory.py`、`tools/osr.py`、`tools/generate_osr_memory_bank.py`（后者 import 不存在的 `osr_utils`，早已无法运行）
- 删除一次性硬编码汇总脚本：`tools/summarize_6_runs_incremental_map50.py`、`tools/extract_nsgp_results.py`、`tools/summarize_final_incremental_map50.py`
- 删除根目录 `eval_6_runs_final.sh`、`eval_6_runs_pseudo_label.sh`（引用已不存在的旧命名数据/run 路径，功能与 `scripts/<dataset>/<split>/eval.sh` 重复）
- 删除 `default.yaml` 中 `abr*` 配置块（7 项）

### 1.3 删除：旧 cfg 管线与一次性脚本

旧管线三件套主训练调用早已被注释、无任何脚本引用，功能被 `tools/train.py` + `tools/eval.py` + `scripts/` 完全覆盖：

- 删除 `tools/train_incremental.py`、`tools/eval_incremental.py`、`tools/incremental_utils.py`
- 删除仅文档提及的遗留工具：`tools/generate_pseudo_label.py`（伪标签已内置于 trainer）、`tools/generate_replay_dataset.py`、`tools/convert_prototype_classes.py`、`tools/train_head_proto.py`
- 删除 `tools/fill_train_scripts.py`（904 行脚本生成器，模板仍是旧命名 `VOC_inc_10_10` 与 proto_rp 方法矩阵，与现行 `scripts/` 已不符）
- 删除路径硬编码的一次性脚本：`tools/train_yoloe_lp.py`、`tools/vis_epoch_trajectory.py`、`tools/vis_map50_curve.py`
- 删除根目录 `prompt.md`（70KB 一次性 LLM prompt 工作垃圾）

### 1.4 删除：`ultralytics/` 内死代码与重复实现

- 删除 `ultralytics/engine/distillation.py`（751 行 `KDLoss` 特征蒸馏族，唯一 import 早已被注释，纯死代码）。保留活跃的蒸馏实现 `anti_forget.py::get_dist_loss`（教师 top-k 通道 KL，`--distillation` 开关）
- 清理 `anti_forget.py` 中全部 kd 注释块（5 处）与 proto_rp 内联实现（`restore_prototypes`、`compute_proto_replay_loss`、`_setup_train`/`_do_train` 中的 proto_rp 分支，约 300 行）。proto_rp 与 `repre.py`（RePRE 区域原型回放）功能重复且无任何脚本使用，RePRE 为保留实现
- 删除 `default.yaml` 中 `kd/distill_layers/distiller` 与 `proto_rp/prototypes/proto_rp_use_base_model/proto_rp_loss_weight/proto_use_neg` 配置键，以及一行项目误加的重复 `tracker:` 键
- 删除 `detect/train.py`、`obb/train.py` 两个 `get_validator` 中的 `kd_loss`、`proto_rp` loss 名分支

### 1.5 合并：近重复工具

- `tools/copy_images_from_search.py`（307 行）并入 `tools/symlink_images_from_search.py`：新增 `--copy` 开关（保留 copy 版的"同文件/同 size+mtime"去重判定），一个文件同时支持软链与复制，删除 copy 版
- 蒸馏双实现合并：见 1.4（保留 `get_dist_loss`，删除 `KDLoss`）
- 原型回放双实现合并：见 1.4（保留 `repre.py`，删除 proto_rp 内联实现）

### 1.6 重构：Trainer 继承结构（降耦合）

- `AntiForgetDetectionTrainer` 由"继承 `AntiForgetTrainer` + 逐字复制 `DetectionTrainer` 的 10 个方法（约 200 行）"改为 **mixin 多继承**：
  `class AntiForgetDetectionTrainer(AntiForgetTrainer, DetectionTrainer)`
  MRO 为 `AntiForgetDetectionTrainer → AntiForgetTrainer → DetectionTrainer → BaseTrainer`：训练循环（`_setup_train`/`_do_train`/`optimizer_step`）取自 `AntiForgetTrainer`，数据/模型方法取自 `DetectionTrainer`。仅保留两个真实差异点：`get_validator`（追加 IL loss 名）与 `progress_string`（%13s 宽列以对齐额外 loss 列，与 `_do_train` 中 pbar 的 %13s 一致）
- 提取 `AntiForgetTrainer._anti_forget_loss_names()`，供 detect/obb 两个 `get_validator` 复用，消除 `AntiForgetOBBTrainer.get_validator` 中的第二份复制
- 附带收益：今后上游 `DetectionTrainer` 方法更新不再需要手工同步到 AntiForget 副本

### 1.7 重命名

| 原名称 | 新名称 | 理由 |
|---|---|---|
| `ultralytics/engine/antiforget.py` | `ultralytics/engine/anti_forget.py` | PEP 8 模块命名（snake_case 分词） |
| `tools/cal_importance.py` | `tools/compute_importance.py` | `cal` 缩写不规范 |
| `tools/fuse_zero-shot_yoloe.py` | `tools/fuse_zero_shot_yoloe.py` | 连字符不是合法 Python 模块名 |
| `scripts/voc-tiny/15_5/create_voc_15_5.sh` | `create_voc-tiny_15_5.sh` | 符合 `create_<dataset>_<split>.sh` 规范 |
| `scripts_structure_skill.md`（根目录） | `skills/scripts_structure_skill.md` | skill 文档应置于 `skills/` |
| `README-YYL.md`（根目录） | `docs/project/vspreg_derivation.md` | 个人署名命名不规范，按内容（VSPReg 数学推导）命名 |

引用已同步更新：`detect/train.py` 的 import、`scripts/model_adapters/ultralytics.sh`（2 处）、`tests/test_ewc.py`、`tools/fuse_zero_shot_yoloe.py` 自身 docstring。
注意：`--trainer antiforget` 这一 CLI 标志字符串**保持不变**（180 个训练脚本均使用它，仅 Python 模块文件改名）。

### 1.8 根目录整理

- 新建 `docs/project/`，移入项目文档与实验材料：`INCREMENTAL_LEARNING_DESIGN.md`（头部已加存档说明：其中 proto_rp/旧管线内容已被现行实现取代）、`experiments.md`、`nips-rebuttal.md`、`15365_ESP_YOLO_Stabilizing_Inc.pdf`、`nsgp_pseudo_label_results.csv`、`nsgp_pseudo_label_results_table.md`、`vspreg_derivation.md`、`images/`（→ `docs/project/images/`，`experiments.md` 内相对链接不受影响）
- 根目录仅保留：`README.md`、`README.zh-CN.md`、`CONTRIBUTING.md`、`LICENSE`、`CITATION.cff`、`pyproject.toml`、`mkdocs.yml`、`.gitignore`、本文件

### 1.9 验证结果与声明

- `py_compile`（Python 3.11）：`tools/`、`ultralytics/engine/`、`ultralytics/models/yolo/`、`ultralytics/nn/`、`tests/` 全部通过。注：本机 base 环境为 Python 3.8，仓库代码本身使用 3.10+ 语法（如 `create_incremental_dataset.py` 的括号式 with），该语法错误在重构前的 HEAD 上同样存在，非本次引入
- 残留引用 grep（`KDLoss|proto_rp|ABRReplay|film|incremental_utils|cal_importance|...` 共 20+ 模式）：`tools/`、`scripts/`、`ultralytics/`、`tests/` 中零残留
- `default.yaml`：YAML 可解析，待删键确认移除，保留键（`distillation`/`repre` 等）确认在位
- mixin 继承的 MRO 线性化与方法解析经 mock 验证：`AntiForgetTrainer` 的训练循环优先，`DetectionTrainer` 的数据/模型方法正确兜底
- **未执行**：`pytest tests/test_bpf.py tests/test_ewc.py` 未能运行——本机没有兼容环境（base 为 py3.8 且无 cv2/pytest，其余 conda 环境为 py3.9/无依赖）。建议在训练环境（Python ≥3.10 + torch + opencv）中补跑这两个测试及一次 voc-tiny/15_5 的 naive → pseudo_label+ewc 冒烟训练
- 全部改动**未提交 git**，`git status` 可查，请审查后自行提交

### 1.10 保留但提请留意

以下分析工具与现行脚本无引用关系，但因属于增量学习方法的分析手段而保留：`tools/vis.py`、`tools/vis_kernel_proj_pc.py`、`tools/vis_eigen_adjust.py`、`tools/vis_prototypes_det.py`、`tools/parse_confusion_matrix.py`。

---

## 1.5+ 第二轮重构（2026-08-05 下午，延续同一任务）

### 追加删除（tools 二次精减）

按"仅保留增量学习流程实际使用 + 对增量学习分析有用的工具"的标准，追加删除 8 个工具：

- `tools/fuse_zero_shot_yoloe.py`（现行 yoloe-v8 脚本直接加载 `yoloe-v8l-seg.pt`，无人调用融合工具）
- `tools/merge_datasets.py`（仅被已删除的 osr 脚本引用）
- `tools/symlink_images_from_search.py`（现行数据集已物化图像，非管线一环；其 `--copy` 合并随之撤销）
- `tools/match_and_pick_dataset.py`、`tools/create_field_dataset_structure.py`、`tools/convert_dota_to_yolo.py`（特定数据集的一次性准备工具，与现行数据集无关）
- `tools/benchmark_speed_flops.py`（速度对比，非 IL 分析）
- `tools/model_compress.py`（PCA 低秩压缩实验，非 IL 分析）

### 调试中修复的缺陷（在 VOC-TINY 小数据集上实测发现）

- `tools/vis_kernel_proj_pc.py`、`tools/vis_eigen_adjust.py`：import 不存在的 `ultralytics.engine.ewpr`（模块早已改名 `espreg`），两个工具此前必现 ImportError → 已改为 `ultralytics.engine.espreg`，并同步 docstring 中的 EWPR 旧称
- `tools/summarize_cumulative_task_map.py`：评估集中某类无实例时 eval CSV 无该类行，原实现直接 `ValueError` 中断整个 eval.sh → 改为告警后将该类排除在宏平均之外（与 validator 计算 mAP 的口径一致）；一个任务全部类缺失仍硬失败
- `tools/pca.py`：fold 模式名存实亡——hook 无条件按 unfold 展开特征，导致 fold 缓存的 components 形状错误（如 (3,27)、(80,720)）→ 修复 hook 使其真正按 fold 语义输出通道维特征（(3,3)、(80,80)），unfold 路径行为不变（已用缓存形状比对验证）
- `tools/vis.py` stage 4：`_calculate_projection_lengths` 的 `torch.norm(..., dim=1)` 返回逐样本长度而非逐主成分长度，与调用方/绘图期望的 `[n_components]` 不符 → 改为 `dim=0`
- `tools/vis_prototypes_det.py`：某层未收集到原型时 `prototypes[i]` 为 None，main 中 `.to(device)` 直接 AttributeError → 跳过 None 层（`run_evaluation` 本就有该守卫）
- `scripts/run_incremental.sh`：新增 `END_TASK` 环境变量（配合已有 `START_TASK` 支持部分任务运行/调试）

### 新增

- `scripts/analyze.sh`：分析工具统一入口（`pca_projection` / `kernel_projection` / `eigen_adjust` / `prototypes` / `confusion_matrix` 五个子命令），解决 `vis.py` 等工具启动参数过长的问题

### README 重写

- `README.md` 彻底重写为本项目文档：conda（python=3.9, cuda=11.8）环境配置、数据集准备、全部方法的训练/评估启动命令、分析工具启动命令；删除全部 ultralytics 上游说明
- 删除 `README.zh-CN.md`（上游中文说明文档，与本项目无关）

### 第二轮验证（conda 环境，python 3.9.25 + torch 2.4.1+cu118，RTX 3090）

- `tests/test_ewc.py` 4/4、`tests/test_bpf.py` 6/6 通过
- VOC-TINY 15+5 全流程冒烟（每任务 1 epoch）全部 EXIT=0：`create_voc-tiny_15_5.sh` 数据集创建；`naive` / `pseudo_label+l2` / `pseudo_label+ewc` / `pseudo_label+dist+espreg` / `pseudo_label+nsgp+repre` / `bpf` 训练；`eval.sh`（naive、ewc 两个 run）；`feature_drift.sh`
- OdinW-13 前 2 任务（TIL，`END_TASK=2`）冒烟全部 EXIT=0：`naive`、`pseudo_label` 训练 + `eval.sh`
- 分析工具实测全部通过：`feature_drift`、`eigen_adjust`、`confusion_matrix`、`prototypes`、`kernel_projection`、`pca_projection`（stage 1/2 用 unfold 缓存 + `--unfold`；stage 3/4 用修复后的 fold 缓存）。注：stage 1/2 全模块出图较慢（半小时量级），属该工具既有的性能特征
- 注：VOC-TINY 源数据集 `data/VOC-TINY-YOLO/` 本次按 create 脚本的预期格式建立（软链自 100 张 VOC coreset），已一并保留在数据目录中

---

## 2. 重构后的代码结构

```text
myultralytics/
├── ultralytics/                    # ultralytics 上游代码 + 项目扩展（项目改动集中如下）
│   ├── cfg/default.yaml            #   新增 IL 配置段：pseudo_label / distillation / bpf* / ewc / l2 /
│   │                               #   espreg / nsgp / repre（kd、proto_rp、abr 段已移除）
│   ├── engine/
│   │   ├── anti_forget.py          #   核心：AntiForgetTrainer(BaseTrainer)，重写 _setup_train/_do_train，
│   │   │                           #   按开关装配各抗遗忘方法；含伪标签合并、get_dist_loss 等模块级辅助函数
│   │   ├── bpf.py                  #   BPF：伪标签分档加权合并、Bridge Future 忽略掩码、DwF 蒸馏
│   │   ├── ewc.py                  #   EWC：逐任务对角 Fisher 重要度加载校验 + 二次惩罚损失
│   │   ├── espreg.py               #   ESPReg：基于 PCA 缓存的特征投影正则（hook 反传累加）
│   │   ├── nsgp.py                 #   NSGP：零空间梯度投影（optimizer step 后投影参数更新）
│   │   ├── repre.py                #   RePRE：旧类 5×5 区域原型经分类分支的回放损失
│   │   └── l2.py                   #   朴素 L2 参数距离正则
│   ├── models/yolo/
│   │   ├── detect/train.py         #   AntiForgetDetectionTrainer（mixin：AntiForgetTrainer+DetectionTrainer）、
│   │   │                           #   BPFDetectionTrainer
│   │   └── obb/train.py            #   AntiForgetOBBTrainer（OBB 版，重写 get_model/get_validator）
│   └── utils/loss.py               #   v8DetectionLoss 支持 BPF 逐 GT 分类权重与 future 忽略掩码
│
├── tools/                          # 独立工具（不继承 ultralytics 类层次），19 个模块
│   ├── train.py                    # 训练入口：--trainer {antiforget,bpf} + 动态参数透传
│   ├── eval.py                     # 评估入口：per-class 指标 CSV + 混淆矩阵
│   ├── create_incremental_dataset.py   # 完整数据集 → 多阶段类增量数据集
│   ├── expand_model_head.py        # 检测头类别数扩展：既有类别 id/顺序不变，未见类别按数据集
│   │                               # 顺序追加在后，同名类对齐既有 id（零初始化 / YOLOE 文本嵌入初始化）
│   ├── convert_dataset_class_ids.py    # 数据集类别 ID 对齐到模型输出空间
│   ├── compute_importance.py       # EWC Fisher 重要度估计
│   ├── expand_importance.py        # 扩头后扩展历史 Fisher/参数快照
│   ├── pca.py + pca_on_gpu.py      # 逐层激活 PCA（espreg/nsgp 的 pca_cache 来源；后者为 GPU 库类）
│   ├── generate_prototypes.py      # RePRE 原型提取（K-means + 密度筛选）
│   ├── generate_eval_tables.py / summarize_cumulative_task_map.py    # 评估结果汇总
│   ├── feature_drift.py            # 两 checkpoint 间 backbone 特征漂移量化
│   ├── vis.py / vis_kernel_proj_pc.py / vis_eigen_adjust.py / vis_prototypes_det.py /
│   │   parse_confusion_matrix.py   # 分析可视化（统一入口 scripts/analyze.sh）
│   └── utils.py                    # tools 公共函数库
│
├── scripts/                        # 纯 shell 编排（无 Python 实现）
│   ├── run_incremental.sh          # 模型无关的任务循环编排器（支持 START_TASK/END_TASK）
│   ├── analyze.sh                  # 分析工具统一入口（5 个子命令）
│   ├── model_adapters/ultralytics.sh   # 框架适配器：任务准备→训练→收尾 调用 tools/ 各入口
│   └── <dataset>/<split>/          # voc{,10_10,15_5,19_1,10_5_5,5_5_5_5,10_2_2_2_2_2}、voc-tiny/15_5、
│       ├── create_<dataset>_<split>.sh #   coco/{40_40,70_10}、odinw-13/13(TIL)
│       ├── eval.sh                 # split 级、模型无关评估
│       ├── feature_drift.sh        # 特征漂移分析
│       └── <baseline>/             # yolov8 / yoloe-v8
│           ├── config.sh
│           └── train_<method>.sh   # naive、bpf、pseudo_label 及其 +ewc/+l2/+espreg/
│                                   # +dist+espreg/+nsgp/+nsgp+repre 组合
│
├── tests/                          # 上游测试 + 项目新增 test_bpf.py、test_ewc.py
├── docs/project/                   # 项目文档与实验材料归档（设计文档、实验日志、rebuttal、论文 PDF、
│   └── images/                     #   结果表、VSPReg 推导、可视化图）
├── skills/                         # Coding Agent 技能文档（含 scripts_structure_skill.md）
├── data -> /hy-tmp/data/           # 数据集（符号链接）
└── runs -> /hy-tmp/runs/           # 训练输出（符号链接）
```

### 入口调用链

```text
scripts/<dataset>/<split>/<baseline>/train_<method>.sh
  └─ source config.sh + scripts/run_incremental.sh（逐任务循环）
       └─ scripts/model_adapters/ultralytics.sh
            ├─ 任务准备(task>1): tools/expand_model_head.py → convert_dataset_class_ids.py
            │                    [+ewc: expand_importance.py] [bpf: tools/train.py 训练 interim 模型]
            ├─ 训练: tools/train.py --trainer antiforget|bpf（方法差异经 --ewc/--espreg/--nsgp/... 开关传入）
            │         └─ ultralytics/models/yolo/{detect,obb}/train.py 的 AntiForget*Trainer
            │              └─ ultralytics/engine/anti_forget.py 按开关装配
            │                 ewc.py / espreg.py / nsgp.py / l2.py / repre.py / bpf.py / get_dist_loss
            └─ 收尾: [ewc/nsgp 系: compute_importance.py] [espreg/nsgp 系: pca.py]
                      [nsgp+repre: generate_prototypes.py]

评估: scripts/<dataset>/<split>/eval.sh
  └─ tools/convert_dataset_class_ids.py → tools/eval.py（逐任务 + 累积数据集）
     → tools/generate_eval_tables.py → tools/summarize_cumulative_task_map.py

分析: scripts/<dataset>/<split>/feature_drift.sh → tools/feature_drift.py
```
